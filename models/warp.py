"""
WarpMixin: image and coordinate warping via the learned displacement field.

Provides warp (warp moving image → fixed space), warp_coords (warp arbitrary
coordinate sets), and warp_4ch_view (warp a 2-D long-axis view).
"""

import torch
import numpy as np

from models.coords import KEY_SAX_VIEW, KEY_4CH_VIEW
from canonical.transforms import execute_resampling


class Warp:
    """Warps images and coordinate tensors using the trained SIREN displacement field."""

    def warp_coords(self, coords, spacing_xyz=None, eval_dvf=True, do_scale=True):
        """
        Warp an arbitrary set of *coords* through the learned deformation.

        Parameters
        ----------
        coords      : np.ndarray | torch.Tensor  (N, 3)  voxel coordinates
        spacing_xyz : tuple  voxel spacing in mm (defaults to self.voxel_size_xyz)
        eval_dvf    : bool  whether to compute and return the Jacobian
        do_scale    : bool  whether to map coords into [-1, 1] before the network

        Returns
        -------
        forward_estimate : np.ndarray  (N, 3)  warped coordinates in voxel space
        dvf_jacobian     : np.ndarray  (N, 3, 3)  only when eval_dvf=True
        """
        if spacing_xyz is None:
            spacing_xyz = self.voxel_size_xyz
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        if coords.device != self.device:
            coords = coords.to(self.device)

        if do_scale:
            coord_range = self.max_coord_offset["ALL"] - self.min_coord_offset["ALL"]
            coords = 2 * ((coords - self.min_coord_offset["ALL"]) / coord_range) - 1

        transform_rel, dvf_jacobian = self._predict_displacement(
            coords, chunk_size=10_000, eval_dvf=eval_dvf
        )
        if transform_rel.device != self.device:
            transform_rel = transform_rel.to(self.device)

        forward_estimate = coords + transform_rel
        x, y, z = self._model_to_image_voxel_coords(
            forward_estimate, "offset", spacing_xyz
        )
        forward_estimate = torch.stack([x, y, z], dim=-1)[None]
        forward_estimate = forward_estimate.detach().cpu().numpy().squeeze()

        if eval_dvf:
            return forward_estimate, dvf_jacobian.detach().cpu().numpy()
        return forward_estimate

    def warp(
        self,
        moving_image=None,
        spacing_xyz=None,
        return_transformation=False,
        eval_dvf=False,
        mode="bilinear",
    ):
        """
        Warp *moving_image* into the fixed image space.

        Parameters
        ----------
        moving_image         : torch.Tensor  3-D volume (defaults to self.moving_image)
        spacing_xyz          : tuple  voxel spacing (defaults to self.voxel_size_xyz)
        return_transformation: bool  also return the DVF and Jacobian tensors
        eval_dvf             : bool  compute deformation gradient (needed for strain)
        mode                 : str   'bilinear' | 'nearest'

        Returns
        -------
        warped_img : np.ndarray
        (optionally) dvf, dvf_jacobian, dvf_jacobian_det, dvf_jacobian_phys
        """
        self.optimizer.zero_grad()

        if spacing_xyz is None:
            spacing_xyz = self.voxel_size_xyz
        if moving_image is None:
            moving_image = self.moving_image

        coordinates = self.possible_coordinate_tensor[KEY_SAX_VIEW]["fixed"]
        dvf_jacobian = dvf_jacobian_det = dvf_jacobian_phys = None

        if eval_dvf:
            transform_rel, dvf_jacobian, dvf_jacobian_phys = self._predict_displacement(
                coordinates,
                chunk_size=10_000,
                eval_dvf=True,
                spacing_xyz=spacing_xyz,
                img_shape=moving_image.shape,
                compute_physical_dvf=True,
            )
            dvf_jacobian_det = (
                torch.det(dvf_jacobian).reshape(moving_image.shape).numpy()
            )
            dvf_jacobian = dvf_jacobian.reshape(moving_image.shape + (3, 3)).numpy()
            dvf_jacobian_phys = dvf_jacobian_phys.reshape(
                moving_image.shape + (3, 3)
            ).numpy()
            dvf_jacobian_phys = dvf_jacobian_phys + np.identity(3)
        else:
            transform_rel, _ = self._predict_displacement(
                coordinates, chunk_size=10_000
            )

        if transform_rel.device != self.device:
            transform_rel = transform_rel.to(self.device)

        forward_estimate = coordinates + transform_rel
        x, y, z = self._model_to_image_voxel_coords(
            forward_estimate, "offset", spacing_xyz
        )
        forward_estimate = torch.stack([x, y, z], dim=-1)[None]

        warped_img = self._torch_grid_sampling(
            moving_image, forward_estimate, moving_image.shape, mode=mode
        )
        warped_img = warped_img.detach().cpu().numpy()
        if mode == "nearest":
            warped_img = warped_img.astype(np.int32)

        if return_transformation:
            x0, y0, z0 = self._model_to_image_voxel_coords(
                coordinates, "offset", spacing_xyz
            )
            orig_coords = torch.stack([x0, y0, z0], dim=-1)[None]
            dvf = forward_estimate - orig_coords
            del orig_coords
            return (
                warped_img,
                dvf.reshape(moving_image.shape + (3,)).detach().cpu().numpy(),
                dvf_jacobian,
                dvf_jacobian_det,
                dvf_jacobian_phys,
            )

        del coordinates
        return warped_img

    def warp_4ch_view(self, view_key, eval_dvf=False):
        """
        Warp a 2-D long-axis view (4-chamber) using the learned SAX deformation.

        Parameters
        ----------
        view_key : str  e.g. KEY_4CH_VIEW or KEY_4CH_SEG_VIEW
        eval_dvf : bool  compute and return the 2-D Jacobian

        Returns
        -------
        resampled_view : np.ndarray
        (optionally) transform_rel, dvf_jacobian, dvf_jacobian_det
        """
        img_type = "seg" if "seg" in view_key else "img"
        mode = "nearest" if img_type == "seg" else "bilinear"

        torch_moving = (
            torch.from_numpy(self.cimage_moving.views[view_key]["np_img"])
            .float()
            .to(self.device)
        )
        tgt_shape_zyx = torch_moving.shape

        coords_4ch = (
            self.possible_coordinate_tensor[KEY_4CH_VIEW]["fixed"]
            .clone()
            .to(self.device)
        )

        if eval_dvf:
            coords_4ch = coords_4ch.requires_grad_(True)
            transform_rel, dvf_jacobian = self._predict_displacement(
                coords_4ch, eval_dvf=True
            )
            dvf_jacobian_det = torch.det(dvf_jacobian).reshape(tgt_shape_zyx).numpy()
            dvf_jacobian = dvf_jacobian.reshape(tgt_shape_zyx + (3, 3)).numpy()
        else:
            transform_rel, _ = self._predict_displacement(coords_4ch)

        if transform_rel.device != self.device:
            transform_rel = transform_rel.to(self.device)

        forward_estimate = coords_4ch + transform_rel
        x, y, z = self._model_to_image_voxel_coords(
            forward_estimate, "backward_2dview", key_of_view=KEY_4CH_VIEW
        )
        forward_estimate = torch.stack([x, y, z], dim=-1)

        if tgt_shape_zyx[0] == 1:
            shape_div = torch.tensor(
                tgt_shape_zyx[1:][::-1] + (2,), dtype=torch.float32, device=self.device
            )
        else:
            shape_div = torch.tensor(
                tgt_shape_zyx[::-1], dtype=torch.float32, device=self.device
            )

        t_coords_normed = (forward_estimate / ((shape_div[None, None] - 1) / 2)) - 1
        t_coords_normed = t_coords_normed.reshape(tuple(tgt_shape_zyx) + (3,))[..., :3]
        resampled_view = execute_resampling(torch_moving, t_coords_normed, mode=mode)
        resampled_view = resampled_view.detach().cpu().numpy()
        if img_type == "seg":
            resampled_view = resampled_view.astype(np.int32)

        if eval_dvf:
            return (
                resampled_view,
                transform_rel.reshape(torch_moving.shape + (3,)).detach().cpu().numpy(),
                dvf_jacobian,
                dvf_jacobian_det,
            )
        return resampled_view

    # ── Temporal warping ──────────────────────────────────────────────────────

    def seq_warp(self, t_idx: int, mode: str = "bilinear", eval_dvf: bool = True):
        """
        Warp the full image at timepoint *t_idx* to the reference frame.

        Parameters
        ----------
        t_idx    : int   timepoint to warp
        mode     : str   'bilinear' | 'nearest'
        eval_dvf : bool  compute spatial deformation gradient

        Returns
        -------
        warped_np       : np.ndarray  (Z, Y, X)         warped intensities
        dvf_np          : np.ndarray  (Z, Y, X, 3)      displacement in voxels
        dvf_jacobian    : np.ndarray  (Z, Y, X, 3, 3)   deformation gradient F
        dvf_jacobian_det: np.ndarray  (Z, Y, X)         det(F)
        final_coords_np : np.ndarray  (1, Z, Y, X, 3)   warped coords for mask resampling
        """
        coords_xt = self.temporal_coordinate_tensor[KEY_SAX_VIEW][t_idx]

        disp_rel, dvf_jacobian = self._predict_displacement_temporal(
            coords_xt,
            chunk_size=10_000,
            eval_dvf=eval_dvf,
        )
        disp_rel = disp_rel.to(self.device)

        # Map displaced coordinates to voxel space.
        f_est = coords_xt[:, :3] + disp_rel
        x_idx, y_idx, z_idx = self._model_to_image_voxel_coords(
            f_est, input_scaling="offset", spacing_xyz=self.spacing_xyz
        )
        final_coords = torch.stack([x_idx, y_idx, z_idx], dim=-1)[None]

        # Warp intensities.
        moving_img = self.sequence[t_idx].get_sax_image(device=self.device)
        warped_np = (
            self._torch_grid_sampling(
                moving_img, final_coords, moving_img.shape, mode=mode
            )
            .detach()
            .cpu()
            .numpy()
        )

        # DVF in voxel-index space.
        x0, y0, z0 = self._model_to_image_voxel_coords(
            coords_xt[:, :3], "offset", self.spacing_xyz
        )
        Z, Y, X = moving_img.shape
        dvf_np = np.stack(
            [
                (x_idx - x0).detach().cpu().numpy(),
                (y_idx - y0).detach().cpu().numpy(),
                (z_idx - z0).detach().cpu().numpy(),
            ],
            axis=-1,
        ).reshape(Z, Y, X, 3)

        dvf_jacobian_det = torch.det(dvf_jacobian).reshape(Z, Y, X).numpy()
        dvf_jacobian = (
            dvf_jacobian.detach()
            .cpu()
            .numpy()
            .reshape(Z, Y, X, 3, 3)
            .astype(np.float32)
        )
        final_coords_np = final_coords.detach().cpu().numpy()

        return warped_np, dvf_np, dvf_jacobian, dvf_jacobian_det, final_coords_np
