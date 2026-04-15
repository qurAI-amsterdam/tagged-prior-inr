"""
Coordinate constants, utilities, and collection classes for cardiac registration.

View-name constants
-------------------
KEY_SAX_VIEW, KEY_SAX_SEG_VIEW, KEY_4CH_VIEW, KEY_4CH_SEG_VIEW,
KEY_2CH_VIEW, KEY_2CH_SEG_VIEW

Coordinate utilities
--------------------
de_normalize              – reverse [-1, 1] normalisation to voxel indices
fast_trilinear_interpolation – trilinear sampling of a 3-D tensor

Classes
-------
Coordinates         – builds, scales, and maps coordinate tensors for the SIREN network
TemporalCoordinates – extends Coordinates for 4-D cine sequences
"""

import torch
import numpy as np
from collections import defaultdict

from utils.cardiac import blur_mask, MMS2MRILabel
from utils.coords import (  # constants live here to avoid circular imports
    KEY_SAX_VIEW,
    KEY_SAX_SEG_VIEW,
    KEY_4CH_VIEW,
    KEY_4CH_SEG_VIEW,
    KEY_2CH_VIEW,
    KEY_2CH_SEG_VIEW,
)
from canonical.transforms import execute_resampling

# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------


def de_normalize(array_shape, x_indices, y_indices, z_indices, min_offset=1):
    """
    Reverse the [-1, 1] normalisation back to voxel indices.

    Parameters
    ----------
    array_shape : tuple of int  (nx, ny, nz)
    x_indices, y_indices, z_indices : torch.Tensor
    min_offset : float  (default 1)

    Returns
    -------
    x_indices, y_indices, z_indices : torch.Tensor  (voxel-index space)
    """
    x_indices = (x_indices + min_offset) * (array_shape[0] - 1) * 0.5
    y_indices = (y_indices + min_offset) * (array_shape[1] - 1) * 0.5
    z_indices = (z_indices + min_offset) * (array_shape[2] - 1) * 0.5
    return x_indices, y_indices, z_indices


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    """
    Fast trilinear interpolation of a 3-D tensor at arbitrary (x, y, z) positions.

    Parameters
    ----------
    input_array : torch.Tensor  shape (X, Y, Z)
    x_indices, y_indices, z_indices : torch.Tensor  (floating-point voxel coords)

    Returns
    -------
    output : torch.Tensor  (same length as x_indices)
    """
    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )

    return output


class Coordinates:
    """Builds, scales, and maps coordinate tensors for the registration network."""

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_coords(self):
        """Collect coordinates from all views, compute scale, and normalise to [-1, 1]."""
        self.possible_coordinate_tensor = defaultdict(dict)
        self.min_coord_offset, self.max_coord_offset = {}, {}
        self._collect_all_coords()
        self._collect_possible_coords()
        self.scale_coords()

    def _collect_possible_coords(self):
        """
        Populate *possible_coordinate_tensor* with per-view coordinate arrays
        and dilated-mask ROI indicators for both the fixed and moving images.
        """
        # SAX view
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"] = (
            self.cimage_fixed.get_sax_coords(device=self.device)
        )
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["move"] = (
            self.cimage_moving.get_sax_coords(device=self.device)
        )

        # Dilated union mask used as a loss ROI
        fixed_seg = self.cimage_fixed.get_sax_image(image_type="mask")
        moving_seg = self.cimage_moving.get_sax_image(image_type="mask")
        fixed_lv_mask = fixed_seg == MMS2MRILabel.LV.value
        moving_lv_mask = moving_seg == MMS2MRILabel.LV.value
        fixed_dilated = blur_mask(
            fixed_lv_mask, kernel_shape=(2, 2), num_dilations=1, apply_blur=False
        )
        moving_dilated = blur_mask(
            moving_lv_mask, kernel_shape=(2, 2), num_dilations=1, apply_blur=False
        )
        dilated_roi = (
            torch.from_numpy(np.logical_or(fixed_dilated, moving_dilated))
            .bool()
            .to(self.device)
        )

        self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW][
            "fixed"
        ] = self.fixed_mask.flatten().bool()
        self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW][
            "move"
        ] = self.moving_mask.flatten().bool()
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed_roi_mask"] = dilated_roi

        # Optional 4-chamber long-axis view
        if self.multiview:
            self.possible_coordinate_tensor[KEY_4CH_VIEW]["fixed"] = (
                self.cimage_fixed.get_lax_4ch_coords_in_canon(
                    device=self.device
                ).squeeze()[..., :3]
            )
            self.possible_coordinate_tensor[KEY_4CH_VIEW]["move"] = (
                self.cimage_moving.get_lax_4ch_coords_in_canon(
                    device=self.device
                ).squeeze()[..., :3]
            )
            self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW][
                "fixed"
            ] = self.fixed_mask_4ch.flatten().bool()
            self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW][
                "move"
            ] = self.moving_mask_4ch.flatten().bool()

    def _collect_all_coords(self):
        """Concatenate all view coordinates to establish the global min/max for scaling."""
        sax_coords = [
            self.cimage_fixed.get_sax_coords(device=self.device),
            self.cimage_moving.get_sax_coords(device=self.device),
        ]
        if self.multiview:
            sax_coords += [
                self.cimage_fixed.get_lax_4ch_coords_in_canon(
                    device=self.device
                ).squeeze()[..., :3],
                self.cimage_moving.get_lax_4ch_coords_in_canon(
                    device=self.device
                ).squeeze()[..., :3],
            ]
        self.all_coords = torch.cat(sax_coords, dim=0)

    # ── Coordinate scaling ───────────────────────────────────────────────────

    def scale_coords(self, frames=("move", "fixed")):
        """Linearly map all coordinate tensors to the [-1, 1] range for SIREN input."""
        max_coords, _ = torch.max(self.all_coords, dim=0)
        min_coords, _ = torch.min(self.all_coords, dim=0)
        self.min_coord_offset["ALL"] = min_coords
        self.max_coord_offset["ALL"] = max_coords
        self.coords_scale_factor = max_coords - min_coords

        coord_range = self.max_coord_offset["ALL"] - self.min_coord_offset["ALL"]
        for view_type in self.cardiac_views:
            for frame in frames:
                scaled = (
                    2
                    * (
                        (self.possible_coordinate_tensor[view_type][frame] - min_coords)
                        / coord_range
                    )
                    - 1
                )
                self.possible_coordinate_tensor[view_type][frame] = scaled
                self.min_coord_offset[view_type], _ = torch.min(scaled, dim=0)
                self.max_coord_offset[view_type], _ = torch.max(scaled, dim=0)
                if self.verbose:
                    print(
                        f"INFO - {view_type}-{frame}: min={self.min_coord_offset[view_type].detach().cpu().numpy()}"
                    )
                    print(
                        f"INFO - {view_type}-{frame}: max={self.max_coord_offset[view_type].detach().cpu().numpy()}"
                    )

        self.all_coords = (
            2 * ((self.all_coords - min_coords) / (max_coords - min_coords)) - 1
        )
        self.coords_scale_factor = torch.abs(self.coords_scale_factor)

    # ── Coordinate mapping ───────────────────────────────────────────────────

    def _model_to_image_voxel_coords(
        self,
        coordinate_tensor,
        input_scaling,
        spacing_xyz=None,
        array_shape=None,
        key_of_view=None,
    ):
        """
        Map network-output coordinates back to image voxel indices.

        Parameters
        ----------
        coordinate_tensor : torch.Tensor  (..., 3)  normalised model coordinates
        input_scaling     : str  'offset' | 'backward_2dview' | other
        spacing_xyz       : tuple  voxel spacing in mm (required for 'offset')
        array_shape       : tuple  image shape (required for fallback branch)
        key_of_view       : str   view key (required for 'backward_2dview')

        Returns
        -------
        x_indices, y_indices, z_indices : torch.Tensor
        """
        coord_range = self.max_coord_offset["ALL"] - self.min_coord_offset["ALL"]

        if input_scaling == "offset":
            new_coords = (
                (coordinate_tensor + 1) * coord_range * 0.5
            ) + self.min_coord_offset["ALL"]
            new_coords = self.cimage_fixed.de_scale_aligned_voxel_coords(
                new_coords, scale_m=spacing_xyz
            )
            return new_coords[..., 0], new_coords[..., 1], new_coords[..., 2]

        elif input_scaling == "backward_2dview":
            assert key_of_view is not None
            coords = (
                (coordinate_tensor + 1) * coord_range * 0.5
            ) + self.min_coord_offset["ALL"]
            coords = self.cimage_fixed.from_canon_to_original_view(
                key_of_view, coords
            ).squeeze()[..., :3]
            return coords[..., 0], coords[..., 1], coords[..., 2]

        else:
            assert array_shape is not None
            x, y, z = de_normalize(
                array_shape,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            )
            return x / spacing_xyz[0], y / spacing_xyz[1], z / spacing_xyz[2]

    def _interpolate(
        self,
        image,
        coordinate_tensor,
        spacing_xyz,
        input_scaling="offset",
        key_of_view=None,
        move_fixed="move",
    ):
        """Trilinear interpolation of *image* at the given model-space coordinates."""
        if self.xyz_sequence:
            image = torch.permute(image, (2, 1, 0))
        x, y, z = self._model_to_image_voxel_coords(
            coordinate_tensor,
            input_scaling,
            spacing_xyz,
            array_shape=image.shape,
            key_of_view=key_of_view,
        )
        return fast_trilinear_interpolation(image, x, y, z)

    def _torch_grid_sampling(self, image, coords, shape_zyx, mode="bilinear"):
        """Sample *image* at *coords* using PyTorch grid_sample (align_corners=True).
            Use bilinear for image and nearest mode for segmentation!"""
        if coords.shape[-1] == 4:
            coords = coords[..., :3]
        if coords.dim() == 2:
            coords = coords[None]

        if shape_zyx[0] == 1:
            shape_div = torch.tensor(
                shape_zyx[1:][::-1] + (2,), dtype=torch.float32, device=self.device
            )
        else:
            shape_div = torch.tensor(
                shape_zyx[::-1], dtype=torch.float32, device=self.device
            )

        t_coords_normed = (coords / ((shape_div[None, None] - 1) / 2)) - 1
        t_coords_normed = t_coords_normed.reshape(tuple(shape_zyx) + (3,))
        return execute_resampling(image, t_coords_normed, mode=mode)


# ── Temporal extension ───────────────────────────────────────────────────────


class TemporalCoordinates(Coordinates):
    """
    Extends Coordinates for 4-D cine sequences.

    Overrides coordinate collection to work from a single reference frame,
    adds a time dimension to produce (x, y, z, t_norm) tensors for each
    timepoint, and provides the batch-sampling helpers used during training.
    """

    # ── Sequence + coordinate initialisation ─────────────────────────────────

    def _init_sequence(self):
        """Set the reference image (last frame) and cache its segmentation mask."""
        self.reference_image = self.sequence[-1]
        self.fixed_mask = self.reference_image.get_sax_image(
            image_type="mask", device=self.device
        )

    def _collect_all_coords(self):
        """Collect global coordinates from the reference frame only."""
        ref = self.reference_image
        coords = [ref.get_sax_coords(device=self.device)]
        if self.multiview:
            coords.append(
                ref.get_lax_4ch_coords_in_canon(device=self.device).squeeze()[..., :3]
            )
        self.all_coords = torch.cat(coords, dim=0)

    def _collect_possible_coords(self):
        """
        Populate coordinate tensors from the reference frame.

        'move' is mirrored from 'fixed' so that the parent's scale_coords() works
        without modification (it expects both keys to exist).
        """
        ref = self.reference_image

        # SAX spatial coordinates
        sax_coords = ref.get_sax_coords(device=self.device)
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"] = sax_coords
        self.possible_coordinate_tensor[KEY_SAX_VIEW]["move"] = sax_coords.clone()

        # Dilated LV+LVBP mask as the loss ROI
        sax_mask_np = ref.get_sax_image(image_type="mask")
        lv_mask_np = np.isin(
            sax_mask_np, [MMS2MRILabel.LV.value, MMS2MRILabel.LVBP.value]
        )
        lvbp_mask_np = np.isin(sax_mask_np, [MMS2MRILabel.LVBP.value])
        dilated_np = blur_mask(
            lv_mask_np, kernel_shape=(2, 2), num_dilations=1, apply_blur=False
        )

        roi_flat = torch.as_tensor(dilated_np, device=self.device).bool().flatten()
        lvbp_flat = torch.as_tensor(lvbp_mask_np, device=self.device).bool().flatten()

        self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW]["fixed"] = roi_flat
        self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW]["move"] = roi_flat.clone()
        self.possible_coordinate_tensor_lvbp_mask = lvbp_flat

        # Optional 4-chamber long-axis view
        if self.multiview:
            lax_coords = ref.get_lax_4ch_coords_in_canon(device=self.device).squeeze()[
                ..., :3
            ]
            self.possible_coordinate_tensor[KEY_4CH_VIEW]["fixed"] = lax_coords
            self.possible_coordinate_tensor[KEY_4CH_VIEW]["move"] = lax_coords.clone()

            lax_mask_np = (
                ref.get_4ch_image(mask=True, device=None) == MMS2MRILabel.LV.value
            )
            lax_flat = torch.as_tensor(lax_mask_np, device=self.device).bool().flatten()
            self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW]["fixed"] = lax_flat
            self.possible_coordinate_tensor[KEY_4CH_SEG_VIEW]["move"] = lax_flat.clone()

    def _init_coords(self):
        """Override: initialise from the reference frame, then scale."""
        self.possible_coordinate_tensor = defaultdict(lambda: defaultdict(dict))
        self.min_coord_offset, self.max_coord_offset = {}, {}
        self._collect_all_coords()
        self._collect_possible_coords()
        self.scale_coords()

    def _make_temporal_encoding(self, t_norm: float, n_points: int) -> torch.Tensor:
        """Return a (N, 1) column of the normalised time value for *n_points*."""
        return torch.full((n_points, 1), t_norm, device=self.device)

    def _init_temporal_coords(self):
        """Pre-compute (x, y, z, t_norm) tensors for every timepoint."""
        self.temporal_coordinate_tensor = defaultdict(lambda: defaultdict(dict))
        sax_coords = self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"]
        N = sax_coords.shape[0]

        for t in range(self.T):
            t_norm = t / max(self.T - 1, 1)
            t_enc = self._make_temporal_encoding(t_norm, N)
            self.temporal_coordinate_tensor[KEY_SAX_VIEW][t] = torch.cat(
                [sax_coords, t_enc], dim=-1
            )

        # Cache myocardium indices once — avoids torch.nonzero every training step
        _mask = self.possible_coordinate_tensor.get(KEY_SAX_SEG_VIEW, {}).get("fixed")
        if getattr(self, "temporal_loss_in_myo", True) and _mask is not None:
            _idx = torch.nonzero(_mask, as_tuple=False).squeeze(1)
            self._myo_indices_cache = _idx if _idx.numel() > 0 else None
        else:
            self._myo_indices_cache = None

    # ── Batch sampling ────────────────────────────────────────────────────────

    def _sample_batch(self, t_idx: int, batch_size: int):
        """Return a random batch of (x, y, z, t) coords and their row indices."""
        coords_xt = self.temporal_coordinate_tensor[KEY_SAX_VIEW][t_idx]
        indices = torch.randperm(coords_xt.shape[0], device=self.device)[:batch_size]
        return coords_xt[indices], indices

    def _batch_lv_mask(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Return the LV ROI mask values at the given row indices."""
        return self.possible_coordinate_tensor[KEY_SAX_SEG_VIEW]["fixed"][batch_indices]

    def _sample_myo_indices(self, batch_size: int) -> torch.Tensor:
        """Sample spatial indices, preferring myocardium voxels when available."""
        idx_all = getattr(self, "_myo_indices_cache", None)
        if idx_all is None:
            idx_all = torch.arange(
                self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"].shape[0],
                device=self.device,
            )
        perm = torch.randperm(idx_all.numel(), device=self.device)[:batch_size]
        return idx_all[perm]

    def _coords_at(self, indices: torch.Tensor, t_idx: int) -> torch.Tensor:
        """Build (x, y, z, t_enc) for *indices* at timepoint *t_idx*.

        Uses the precomputed temporal_coordinate_tensor for t_idx in [0, T) to
        avoid recreating the temporal encoding tensor on every call.  Falls back
        to the original per-call construction for anchor timepoints (e.g.
        t_idx = T for the reference frame in the identity anchor loss).
        """
        if 0 <= t_idx < self.T:
            coords = self.temporal_coordinate_tensor[KEY_SAX_VIEW][t_idx]
            return coords[indices.to(coords.device)]
        # Anchor timepoints (t_idx outside [0, T)) — recompute t_enc on the fly
        xyz = self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"][indices]
        t_norm = 0.0 if self.T <= 1 else t_idx / (self.T - 1)
        t_enc = self._make_temporal_encoding(t_norm, xyz.shape[0])
        return torch.cat([xyz.to(self.device), t_enc], dim=-1)
