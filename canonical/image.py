"""
Canonical-space image representation for cardiac MRI.

Provides the CanonicalImage class and MRILabel enum (previously in
kwatsch/canonical_space.py).
"""

import numpy as np
import torch
from scipy import ndimage
import SimpleITK as sitk
import scipy
from enum import Enum
from torch import inverse as tr_inv
from collections import defaultdict

from canonical.transforms import (
    identity_grid,
    execute_resampling,
    get_voxel_to_world_transforms,
    make_homogeneous_identity_grid,
    make_homegeneous_identity_grid,  # legacy alias
)
from utils.coords import (
    KEY_SAX_VIEW,
    KEY_2CH_VIEW,
    KEY_2CH_SEG_VIEW,
    KEY_4CH_SEG_VIEW,
    KEY_4CH_VIEW,
)
from utils.cardiac import (
    normalize_image,
    get_center,
    check_apex_base_orientation,
    rotation_matrix,
)
from postprocessing.contours import convert_mask_to_contour, create_mask_epi_heart


class MRILabel(Enum):
    """Label IDs used inside CanonicalImage (SAX view segmentations)."""

    BG = 0
    LVBP = 1
    LV = 2
    RVBP = 3


class CanonicalImage(object):
    """
    Wraps a 3-D cardiac MRI image together with its segmentation mask and
    provides methods to align it to a canonical (canonical orientation) space.
    """

    def __init__(
        self,
        sitk_image: sitk.Image,
        sitk_seg: sitk.Image,
        label=MRILabel,
        key=KEY_SAX_VIEW,
        device="cuda",
        normalize=False,
        xyz_sequence=True,
        dtype=torch.float32,
        source_obj=None,
        z_flip=None,
    ):
        self.views = defaultdict(dict)
        self.meshes = defaultdict(dict)
        self.main_key = key
        self.dtype = dtype
        self.device = device
        self.z_flip = z_flip
        self.views[key]["sitk_img"] = sitk_image
        (
            self.views[self.main_key]["rotate"],
            self.views[self.main_key]["scale"],
            self.views[self.main_key]["translate"],
        ) = get_voxel_to_world_transforms(sitk_image, device=device)
        self.views[self.main_key]["origin_xyz"] = (
            torch.from_numpy(np.asarray(sitk_image.GetOrigin()).astype(np.float32))
            .float()
            .to(device)
        )
        self.views[key]["sitk_seg"] = sitk_seg
        self.views[key]["spacing_xyz"] = (
            torch.from_numpy(
                np.asarray(self.views[key]["sitk_img"].GetSpacing()).astype(np.float32)
            )
            .float()
            .to(self.device)
        )
        self.xyz_sequence = xyz_sequence
        self.label = label
        self.shape_zyx = sitk_image.GetSize()[::-1]
        self.t_possible_coords = defaultdict(dict)
        self.t_coords_aligned_xyz = None
        self.views[key]["np_img"] = sitk.GetArrayFromImage(sitk_image).astype(
            np.float32
        )
        self.views[key]["np_seg"] = sitk.GetArrayFromImage(sitk_seg).astype(np.int32)
        self.source_obj = source_obj
        if normalize:
            self.views[key]["np_img"] = normalize_image(
                self.views[key]["np_img"], percentile=(1, 99)
            )
        if source_obj is None:
            self._prepare_transformation()
        else:
            self._copyInformation(source_obj)
            lv_com_zyx = get_center(
                self.views[self.main_key]["np_seg"], self.label.LVBP.value
            )
            self.views[self.main_key]["original_lv_com_xyz"] = torch.multiply(
                torch.from_numpy(lv_com_zyx[::-1].copy()).float().to(self.device),
                self.views[self.main_key]["spacing_xyz"],
            )

    def _generate_canonical_direction(self):
        self.views[self.main_key]["canon_rotate"] = self.views[self.main_key]["rotate"]

    def _copyInformation(self, source):
        """Copy alignment info from a source (fixed) CanonicalImage."""
        self.z_flip = source.z_flip
        self.flip_y_m = source.flip_y_m.clone()
        self.flip_z_m = source.flip_z_m.clone()
        self.rot_rv_lv_m = source.rot_rv_lv_m.clone()
        self.views[self.main_key]["lv_com_xyz"] = source.views[self.main_key][
            "lv_com_xyz"
        ].clone()
        self.views[self.main_key]["coords_scaled"] = source.views[self.main_key][
            "coords_scaled"
        ].clone()
        self.views[self.main_key]["canon_rotate"] = source.views[self.main_key][
            "canon_rotate"
        ].clone()
        self.views[self.main_key]["lv_com_aligned_xyz"] = source.views[self.main_key][
            "lv_com_aligned_xyz"
        ].clone()
        self.views[self.main_key]["coords_scaled_centered"] = source.views[
            self.main_key
        ]["coords_scaled_centered"].clone()
        self.views[self.main_key]["rotate"] = source.views[self.main_key][
            "rotate"
        ].clone()
        self.views[self.main_key]["scale"] = source.views[self.main_key][
            "scale"
        ].clone()
        self.views[self.main_key]["translate"] = source.views[self.main_key][
            "translate"
        ].clone()

    def _prepare_transformation(self):
        self.flip_y_m = torch.eye(4, device=self.device)
        self.flip_z_m = torch.eye(4, device=self.device)
        self.rotmat = torch.eye(4, device=self.device)
        if self.z_flip is None:
            self._check_apex_base_orientation(self.views[self.main_key]["np_seg"])
        lv_com_zyx = get_center(
            self.views[self.main_key]["np_seg"], self.label.LVBP.value
        )
        rv_com_zyx = get_center(
            self.views[self.main_key]["np_seg"], self.label.RVBP.value
        )
        self.rot_rv_lv_m = torch.eye(4).to(self.device)
        self.vec_lv_rv = rv_com_zyx - lv_com_zyx
        self.rot_rv_lv_m[:3, :3] = torch.from_numpy(
            CanonicalImage.get_orientation(y=self.vec_lv_rv[1], x=self.vec_lv_rv[2])
        ).float()
        self.prepare_y_z_flip()
        self.views[self.main_key]["lv_com_xyz"] = torch.multiply(
            torch.from_numpy(lv_com_zyx[::-1].copy()).float().to(self.device),
            self.views[self.main_key]["spacing_xyz"],
        )
        coords = self._init_scale_coords(
            self.shape_zyx, self.views[self.main_key]["scale"]
        )
        self.views[self.main_key]["coords_scaled"] = coords
        self._generate_canonical_direction()

    def _init_scale_coords(
        self,
        shape_zyx: tuple,
        m_scale: torch.FloatTensor,
        filter_on_indices=None,
        return_homogenous=False,
    ):
        """Scale voxel-index grid to world coordinates and shift to voxel centres."""
        coords = identity_grid(
            shape_zyx, device=self.device, do_flip_sequence=self.xyz_sequence
        )
        coords = torch.cat(
            [coords, torch.ones(coords.shape[:-1] + (1,), device=self.device)], dim=-1
        )
        coords = coords @ m_scale
        voxel_origin = -torch.diag(m_scale) / 2
        coords = coords + voxel_origin
        if filter_on_indices is not None:
            coords = coords[filter_on_indices].squeeze()
        if return_homogenous:
            return coords
        return coords[..., :3]

    def align_images(self, rv_lv_rot_matrix=None, include_contours=False):
        """Align both image and segmentation to the canonical orientation."""
        if rv_lv_rot_matrix is not None:
            self.rot_rv_lv_m = rv_lv_rot_matrix
        t_img = torch.from_numpy(self.views[self.main_key]["np_img"]).float()
        self.views[self.main_key]["np_img_aligned"] = (
            self.align(
                t_img,
                mode="bilinear",
                src_shape_zyx=t_img.shape,
                tgt_shape_zyx=t_img.shape,
            )
            .detach()
            .cpu()
            .numpy()
        )
        direction = (
            self.views[self.main_key]["rotate"][:3, :3].detach().cpu().numpy().flatten()
        )
        spacing = (
            torch.diag(self.views[self.main_key]["scale"])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        origin = self.views[self.main_key]["translate"][:3, 3].detach().cpu().numpy()
        self.views[self.main_key]["sitk_img_aligned"] = self.create_sitk_image(
            self.views[self.main_key]["np_img_aligned"],
            spacing_xyz=spacing,
            origin_xyz=origin,
            direction=direction,
            dtype=np.float32,
        )
        t_seg = torch.from_numpy(self.views[self.main_key]["np_seg"]).float()
        self.views[self.main_key]["np_seg_aligned"] = (
            self.align(
                t_seg,
                mode="nearest",
                src_shape_zyx=t_seg.shape,
                tgt_shape_zyx=t_seg.shape,
            )
            .detach()
            .cpu()
            .numpy()
        )

        if include_contours:
            epi_mask = create_mask_epi_heart(
                self.views[self.main_key]["np_seg_aligned"],
                num_dilations=2,
                kernel=(2, 2),
            )
            contours, contour_as_mask, normals, normal_as_mask = (
                convert_mask_to_contour(
                    self.views[self.main_key]["np_seg_aligned"],
                    epi_mask=epi_mask,
                    upfactor=8,
                    compute_derivatives=True,
                )
            )
            self.views[self.main_key]["np_contour_as_mask"] = contour_as_mask
            self.views[self.main_key]["np_normal_as_mask"] = normal_as_mask
            self.views[self.main_key]["np_contour"] = contours
            self.views[self.main_key]["np_normal"] = normals

        self.views[self.main_key]["origin_shift"] = np.zeros(3).astype(np.float32)
        zmask = (self.views[self.main_key]["np_seg_aligned"] == 1).any((1, 2))
        if ~zmask[0]:
            last_empty_slice = int(np.min(np.where(zmask)[0]))
            new_origin = self.views[self.main_key]["sitk_img"][
                :, :, last_empty_slice:
            ].GetOrigin()
            self.views[self.main_key]["origin_shift"] = np.asarray(new_origin) - origin

        self.views[self.main_key]["sitk_seg_aligned"] = self.create_sitk_image(
            self.views[self.main_key]["np_seg_aligned"],
            spacing_xyz=spacing,
            origin_xyz=origin,
            direction=direction,
            dtype=np.int32,
        )

        if self.source_obj is None:
            lv_com_zyx = get_center(
                self.views[self.main_key]["np_seg_aligned"], label=self.label.LVBP.value
            )
            self.views[self.main_key]["lv_com_aligned_xyz"] = torch.multiply(
                torch.from_numpy(lv_com_zyx[::-1].copy()).float().to(self.device),
                self.views[self.main_key]["spacing_xyz"],
            )
            coords = self._init_scale_coords(
                self.shape_zyx, self.views[self.main_key]["scale"]
            )
            self.views[self.main_key]["coords_scaled"] = coords
            self.views[self.main_key]["coords_scaled_centered"] = (
                coords - self.views[self.main_key]["lv_com_aligned_xyz"]
            )

    def align_coords(self, coords):
        """Map raw voxel coordinates to aligned, centred world coordinates."""
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        if coords.device != self.device:
            coords = coords.to(self.device)
        coords = torch.cat(
            [coords, torch.ones(coords.shape[:-1] + (1,), device=self.device)], dim=-1
        )
        coords = coords @ self.views[self.main_key]["scale"]
        voxel_origin = -torch.diag(self.views[self.main_key]["scale"]) / 2
        coords = coords + voxel_origin
        coords = coords[..., :3]
        coords = coords - self.views[self.main_key]["lv_com_aligned_xyz"]
        return coords

    def _remove_rv_mask(self, np_array_seg: np.ndarray) -> np.ndarray:
        np_array_seg[np_array_seg == self.label.RVBP.value] = 0
        return np_array_seg

    def _get_sitk_slice(self, key: str, sitk_img) -> sitk.Image:
        mid_slice_id = sitk_img.GetSize()[-1] // 2
        sitk_new = sitk_img[:, :, mid_slice_id : mid_slice_id + 1]
        self.views[key]["slice_id"] = mid_slice_id
        return sitk_new

    def add_mesh(self, key, mesh, latent=None):
        """Register a mesh (e.g. shape model) with optional latent vector."""
        if np.any(self.views[self.main_key]["origin_shift"] != 0):
            pass
        com_mesh_xyz = mesh.points.mean(0)
        mesh.points = mesh.points - com_mesh_xyz
        self.meshes[key]["mesh"] = mesh
        self.meshes[key]["latent"] = None
        if latent is not None:
            self.meshes[key]["latent"] = latent

    def add_view(
        self,
        sitk_img: sitk.Image,
        key,
        dtype=np.float32,
        normalize=False,
        do_align=True,
        keep_3d=False,
        origin_offset=None,
    ):
        """Add a secondary cardiac view (e.g. 4CH) and compute its coordinates in SAX space."""
        mid_slice_id = None
        if (
            not keep_3d
            and key in [KEY_4CH_VIEW, KEY_4CH_SEG_VIEW, KEY_2CH_VIEW, KEY_2CH_SEG_VIEW]
            and (len(sitk_img.GetSize()) == 3 and sitk_img.GetSize()[-1] > 1)
        ):
            sitk_img = self._get_sitk_slice(key, sitk_img)
        self.views[key]["sitk_img"] = sitk_img
        self.views[key]["np_img"] = sitk.GetArrayFromImage(sitk_img).astype(dtype)
        self.views[key]["shape_zyx"] = self.views[key]["np_img"].shape
        self.views[key]["spacing_xyz"] = (
            torch.from_numpy(
                np.asarray(self.views[key]["sitk_img"].GetSpacing()).astype(np.float32)
            )
            .float()
            .to(self.device)
        )
        if normalize:
            self.views[key]["np_img"] = normalize_image(
                self.views[key]["np_img"], percentile=(1, 99)
            )
        if key == KEY_2CH_SEG_VIEW:
            self.views[key]["np_img"] = self._remove_rv_mask(self.views[key]["np_img"])
        R_tgt, S_tgt, T_tgt = get_voxel_to_world_transforms(
            sitk_img, device=self.device
        )
        if origin_offset is not None:
            T_tgt[:3, 3] = T_tgt[:3, 3] + torch.from_numpy(origin_offset).float().to(
                self.device
            )
        if mid_slice_id is not None:
            S_tgt[2, 2] = 1.0
        (
            self.views[key]["rotate"],
            self.views[key]["scale"],
            self.views[key]["translate"],
        ) = (R_tgt, S_tgt, T_tgt)
        mode = "nearest" if dtype == np.int32 else "bilinear"
        if do_align:
            self._align_added_view(key, mode, keep_3d=keep_3d)
        self.views[key]["coords_in_main_scaled_centered"] = (
            self._get_coords_2dview_in_sax(key, keep_3d=keep_3d)
        )

    def _get_coords_2dview_in_sax(self, key, keep_3d=False):
        view_coords_in_main_coords, _ = self._coords_to_view(key, self.main_key)
        view_coords_in_main_coords = self._align_voxel_coords(
            view_coords_in_main_coords, self.shape_zyx, z_dim=1
        )
        self.views[key]["bogus"] = view_coords_in_main_coords
        view_coords_in_main_coords = (
            view_coords_in_main_coords @ self.views[self.main_key]["scale"]
        )
        view_coords_in_main_coords[..., :3] = (
            view_coords_in_main_coords[..., :3]
            - self.views[self.main_key]["lv_com_aligned_xyz"]
        )
        return view_coords_in_main_coords

    def _align_added_view(self, key, mode, keep_3d=False):
        aligned_view, filtered_coords_scaled, filter_indices = (
            self.resample_to_canonical_view(key, mode=mode, keep_3d=keep_3d)
        )
        self.views[key]["canon_spacing_xyz"] = torch.diag(
            self.views[key]["canon_scale"]
        )[:3]
        self.views[key]["np_img_aligned"] = aligned_view.numpy()
        filter_indices = filter_indices.detach().cpu().numpy()
        self.views[key]["np_img_aligned_filtered"] = np.squeeze(
            self.views[key]["np_img_aligned"].flatten()[filter_indices]
        )
        self.views[key]["filtered_coords_scaled"] = filtered_coords_scaled[..., :3]
        filtered_coords_scaled[..., :3] = (
            filtered_coords_scaled[..., :3]
            - self.views[self.main_key]["lv_com_aligned_xyz"]
        )
        self.views[key]["filtered_coords_centered"] = filtered_coords_scaled[
            ..., :3
        ].clone()
        direction = (
            self.views[self.main_key]["canon_rotate"][:3, :3]
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )
        spacing = (
            self.views[key]["canon_spacing_xyz"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        origin = self.views[key]["canon_translate"][:3, 3].detach().cpu().numpy()
        self.views[key]["sitk_img_aligned"] = self.create_sitk_image(
            aligned_view.detach().cpu().numpy(),
            spacing_xyz=spacing,
            origin_xyz=origin,
            direction=direction,
            dtype=np.float32,
        )

    @staticmethod
    def _blur(np_array, zoom, sigma=None):
        if sigma is None:
            sigma = 0.25 / zoom
        for z in range(np_array.shape[0]):
            np_array[z, :, :] = scipy.ndimage.gaussian_filter(np_array[z, :, :], sigma)
        return np_array

    def resample_to_canonical_view(self, key, mode="bilinear", keep_3d=False):
        """Resample a 2-D long-axis view into the 3-D SAX canonical volume."""
        np_array = self.views[key]["np_img"].copy()
        vox_to_canon, grid_indices, tgt_shape_zyx, zoom = (
            self._coords_to_canonical_view(key, keep_3d=keep_3d)
        )
        if zoom is not None:
            np_array = CanonicalImage._blur(np_array, zoom)
        torch_array = torch.from_numpy(np_array).float().to(self.device)
        src_shape_zyx = torch_array.shape
        filtered_coords_scaled = self._init_scale_coords(
            tgt_shape_zyx,
            self.views[key]["canon_scale"],
            grid_indices,
            return_homogenous=True,
        )
        resampled_array = self._resample(
            torch_array, vox_to_canon, src_shape_zyx, tgt_shape_zyx, mode=mode
        )
        np_resampled_array = resampled_array.detach().cpu().squeeze()
        return np_resampled_array, filtered_coords_scaled, grid_indices

    def _coords_to_canonical_view(self, key, keep_3d=False):
        zoom = int(
            torch.ceil(
                self.views[self.main_key]["scale"][2, 2]
                / self.views[key]["scale"][0, 0]
            ).item()
        )
        num_z_slices = int(self.shape_zyx[0] * zoom)
        tgt_shape_zyx = np.asarray((num_z_slices,) + self.shape_zyx[1:])
        new_scale = self.views[self.main_key]["scale"].clone()
        new_scale[2, 2] = self.views[self.main_key]["scale"][2, 2] / zoom
        self.views[key]["canon_scale"] = new_scale
        new_trans = self.views[self.main_key]["translate"].clone()
        self.views[key]["canon_translate"] = new_trans
        ident_grid = make_homogeneous_identity_grid(tgt_shape_zyx, device=self.device)
        voxel_grid_aligned = self._align_voxel_coords(ident_grid, tgt_shape_zyx)
        world_grid = (
            voxel_grid_aligned
            @ new_scale.T
            @ self.views[self.main_key]["rotate"].T
            @ new_trans.T
        )
        voxel_grid_in_tgt = (
            world_grid
            @ tr_inv(self.views[key]["translate"]).T
            @ tr_inv(self.views[key]["rotate"]).T
            @ tr_inv(self.views[key]["scale"]).T
        )
        voxel_grid_in_tgt = voxel_grid_in_tgt.reshape(-1, 4)
        voxel_grid_in_tgt[:, :3] = voxel_grid_in_tgt[:, :3] - 0.5
        if not keep_3d:
            z_bound_min, z_bound_max = -1.5, 1.5
            indices = torch.nonzero(
                (voxel_grid_in_tgt[..., 2] > z_bound_max)
                | (voxel_grid_in_tgt[..., 2] < z_bound_min)
            )
            coords_indices_2dview_only = torch.nonzero(
                (voxel_grid_in_tgt[..., 2] <= z_bound_max)
                & (voxel_grid_in_tgt[..., 2] >= z_bound_min)
            )
            voxel_grid_in_tgt[indices, :] = 0
        else:
            z_bound_min = torch.min(voxel_grid_in_tgt[..., 2])
            z_bound_max = torch.max(voxel_grid_in_tgt[..., 2])
            coords_indices_2dview_only = torch.nonzero(
                (voxel_grid_in_tgt[..., 2] <= z_bound_max)
                & (voxel_grid_in_tgt[..., 2] >= z_bound_min)
            )
        return voxel_grid_in_tgt, coords_indices_2dview_only, tuple(tgt_shape_zyx), zoom

    def resample_to_view(
        self,
        input_image=None,
        key_to=KEY_4CH_VIEW,
        img_type="img",
        key_from=None,
        de_align=False,
    ) -> torch.Tensor:
        """Resample image or segmentation into the coordinate space of a 2-D view."""
        assert img_type in ["img", "seg"]
        mode = "nearest" if img_type == "seg" else "bilinear"
        if key_from is not None:
            np_array = (
                self.views[key_from]["np_seg"]
                if img_type == "seg"
                else self.views[key_from]["np_img"]
            )
        else:
            np_array = input_image
        torch_array = torch.from_numpy(np_array).float().to(self.device)
        src_shape_zyx = np_array.shape
        vox_to_2dview, tgt_shape_zyx = self._coords_to_view(key_to, key_from)
        if de_align:
            vox_to_2dview = self._de_align_voxel_coords(
                vox_to_2dview, src_shape_zyx, z_dim=1
            )
        resampled_array = self._resample(
            torch_array, vox_to_2dview, src_shape_zyx, tgt_shape_zyx, mode=mode
        )
        return resampled_array.detach().cpu().squeeze()

    def _align_voxel_coords(self, voxel_grid, shape_zyx, z_dim=None):
        if isinstance(shape_zyx, tuple):
            shape_zyx = np.asarray(shape_zyx).astype(np.int32)
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (
            torch.from_numpy(shape_zyx[::-1].copy()).float() / 2
        ) - 0.5
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.flip_z_m.T
            @ self.flip_y_m.T
            @ tr_inv(self.rot_rv_lv_m)
            @ trans_origin_xyz.T
        )
        if z_dim is None:
            z_dim = shape_zyx[0]
        return voxel_grid.reshape(tuple((z_dim, -1, 4))) @ trans_m

    def _de_align_voxel_coords(self, voxel_grid, shape_zyx=None, z_dim=None):
        if shape_zyx is None:
            shape_zyx = self.shape_zyx
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (torch.tensor(shape_zyx[::-1]) / 2) - 0.5
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.rot_rv_lv_m
            @ self.flip_y_m.T
            @ self.flip_z_m.T
            @ trans_origin_xyz.T
        )
        if z_dim is None:
            z_dim = shape_zyx[0]
        return voxel_grid.reshape(z_dim, -1, 4) @ trans_m

    def de_scale_aligned_voxel_coords(self, voxel_grid, scale_m=None):
        """Map aligned, scaled, centred SAX coords back to original voxel coordinates."""
        voxel_grid[..., :3] = (
            voxel_grid[..., :3] + self.views[self.main_key]["lv_com_aligned_xyz"]
        )
        if scale_m is None:
            scale_m = self.views[self.main_key]["scale"]
        if voxel_grid.shape[-1] == 3:
            if scale_m.dim() == 1:
                voxel_grid = torch.divide(voxel_grid, scale_m)
            else:
                voxel_grid = voxel_grid @ tr_inv(scale_m[:3, :3])
        else:
            voxel_grid = voxel_grid @ tr_inv(scale_m)
        return voxel_grid

    def _de_align_view(self, torch_array, mode="bilinear"):
        shape_zyx = torch_array.shape
        ident_grid = make_homogeneous_identity_grid(shape_zyx, device=self.device)
        ident_grid = ident_grid.reshape(shape_zyx[0], -1, 4)
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (torch.tensor(shape_zyx[::-1]) / 2) - 0.5
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.rot_rv_lv_m
            @ self.flip_y_m.T
            @ self.flip_z_m.T
            @ trans_origin_xyz.T
        )
        transformed_grid = ident_grid @ trans_m
        return self._resample(
            torch_array, transformed_grid, shape_zyx, shape_zyx, mode=mode
        )

    def _coords_to_view(self, key_to, key_from):
        tgt_shape_zyx = self.views[key_to]["np_img"].shape
        ident_grid = make_homogeneous_identity_grid(tgt_shape_zyx, device=self.device)
        ident_grid = ident_grid.reshape(tgt_shape_zyx[0], -1, 4)
        world_grid = (
            ident_grid
            @ self.views[key_to]["scale"].T
            @ self.views[key_to]["rotate"].T
            @ self.views[key_to]["translate"].T
        )
        target_grid = (
            world_grid
            @ tr_inv(self.views[key_from]["translate"]).T
            @ tr_inv(self.views[key_from]["rotate"]).T
            @ tr_inv(self.views[key_from]["scale"]).T
        )
        return target_grid, tgt_shape_zyx

    def from_canon_to_original_view(
        self, key_to: str, coords: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Convert canonical (aligned SAX) coordinates back to original 2-D view voxels."""
        assert key_to in [
            KEY_4CH_VIEW,
            KEY_4CH_SEG_VIEW,
            KEY_2CH_VIEW,
            KEY_2CH_SEG_VIEW,
        ]
        if len(coords.shape) == 3:
            coords = coords.squeeze()
        if coords.shape[-1] == 3:
            coords = torch.cat(
                [coords, torch.ones((len(coords), 1), device=coords.device)], dim=-1
            )
        coords = coords[None]
        src_shape_zyx = self.views[key_to]["np_img_aligned"].shape
        coords = self.de_scale_aligned_voxel_coords(
            coords, scale_m=self.views[key_to]["canon_scale"]
        )
        coords = self._de_align_voxel_coords(coords, src_shape_zyx, z_dim=1)
        trans_m = (
            self.views[key_to]["canon_scale"].T
            @ self.views[self.main_key]["rotate"].T
            @ self.views[self.main_key]["translate"].T
        )
        world_coords = coords @ trans_m
        trans_m = (
            tr_inv(self.views[key_to]["translate"]).T
            @ tr_inv(self.views[key_to]["rotate"]).T
            @ tr_inv(self.views[key_to]["scale"]).T
        )
        return world_coords @ trans_m

    def _resample(
        self, torch_array, new_grid, src_shape_zyx, tgt_shape_zyx, mode="bilinear"
    ):
        if src_shape_zyx[0] == 1:
            shape_div = torch.tensor(
                src_shape_zyx[1:][::-1]
                + (
                    2,
                    2,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            shape_div = torch.tensor(
                src_shape_zyx[::-1] + (2,), dtype=torch.float32, device=self.device
            )
        t_coords_normed = (new_grid / ((shape_div[None, None, None] - 1) / 2)) - 1
        t_coords_normed = t_coords_normed.reshape(tuple(tgt_shape_zyx) + (4,))
        t_coords_normed = t_coords_normed[..., :3]
        return execute_resampling(torch_array, t_coords_normed, mode=mode)

    def create_sitk_image(
        self,
        np_array: np.ndarray,
        spacing_xyz=None,
        origin_xyz=None,
        direction=None,
        dtype=np.float32,
    ):
        """Create a SimpleITK image from a numpy array with given metadata."""
        sitk_img = sitk.GetImageFromArray(np_array.astype(dtype))
        sitk_img.SetOrigin(origin_xyz.astype(np.float64))
        sitk_img.SetSpacing(spacing_xyz.astype(np.float64))
        sitk_img.SetDirection(direction.astype(np.float64))
        return sitk_img

    @property
    def shape(self):
        return self.shape_zyx

    def get_sax_image(self, device=None, image_type="image"):
        """Return SAX image or mask array (optionally on device)."""
        assert image_type in [
            "image",
            "mask",
            "contour",
            "normal",
            "contour_as_mask",
            "normal_as_mask",
        ]
        key_map = {
            "image": "np_img_aligned",
            "mask": "np_seg_aligned",
            "contour_as_mask": "np_contour_as_mask",
            "normal_as_mask": "np_normal_as_mask",
            "contour": "np_contour",
            "normal": "np_normal",
        }
        obj = self.views[self.main_key][key_map[image_type]]
        if device is not None:
            if type(obj) is not np.ndarray:
                obj = obj.numpy()
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_4ch_image(self, device=None, mask=False):
        obj = (
            self.views[KEY_4CH_SEG_VIEW]["np_img"]
            if mask
            else self.views[KEY_4CH_VIEW]["np_img"]
        )
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_4ch_image_in_sax(self, device=None, mask=False):
        obj = (
            self.views[KEY_4CH_SEG_VIEW]["np_img_aligned"]
            if mask
            else self.views[KEY_4CH_VIEW]["np_img_aligned"]
        )
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_4ch_slice_in_sax(self, device=None, mask=False):
        obj = (
            self.views[KEY_4CH_SEG_VIEW]["np_img_aligned_filtered"]
            if mask
            else self.views[KEY_4CH_VIEW]["np_img_aligned_filtered"]
        )
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_2ch_image(self, device=None, mask=False):
        obj = (
            self.views[KEY_2CH_SEG_VIEW]["np_img"]
            if mask
            else self.views[KEY_2CH_VIEW]["np_img"]
        )
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_2ch_slice_in_sax(self, device=None, mask=False):
        obj = (
            self.views[KEY_2CH_SEG_VIEW]["np_img_aligned_filtered"]
            if mask
            else self.views[KEY_2CH_VIEW]["np_img_aligned_filtered"]
        )
        if device is not None:
            obj = torch.from_numpy(obj).float().to(device)
        return obj

    def get_sax_coords(self, device=None) -> torch.FloatTensor:
        obj = self.views[self.main_key]["coords_scaled_centered"]
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_4ch_coords(self, device=None) -> torch.FloatTensor:
        obj = self.views[KEY_4CH_VIEW]["filtered_coords_centered"]
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_4ch_coords_in_canon(self, device=None) -> torch.FloatTensor:
        obj = self.views["lax4ch"]["coords_in_main_scaled_centered"].clone()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_2ch_coords(self, device=None) -> torch.FloatTensor:
        obj = self.views[KEY_2CH_VIEW]["filtered_coords_centered"]
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_lax_2ch_coords_in_canon(self, device=None) -> torch.FloatTensor:
        obj = self.views["lax2ch"]["coords_in_main_scaled_centered"].clone()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_mesh(self, key):
        return self.meshes[key]["mesh"]

    def get_mesh_coords(self, key, device=None, filter_base=True):
        """Return mesh point coordinates, optionally filtering the LV base."""
        points = np.array(self.meshes[key]["mesh"].points).astype(np.float32)
        if filter_base:
            coords_max_z = np.max(points, axis=0)[-1]
            coords_max_z = coords_max_z - 2.5 * float(
                self.views[self.main_key]["spacing_xyz"][-1]
            )
            points = points[points[:, 2] <= coords_max_z]
        if device is None:
            return points
        return torch.from_numpy(points).float().to(device)

    def get_mesh_latent(self, key, device=None):
        if device is None:
            return self.meshes[key]["latent"]
        return torch.from_numpy(self.meshes[key]["latent"]).float().to(device)

    def get_spacing(self, key, device=None):
        obj = self.views[key]["spacing_xyz"]
        if isinstance(obj, tuple):
            obj = torch.as_tensor(obj).float()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_4ch_spacing(self, device=None):
        obj = self.views[KEY_4CH_VIEW]["canon_spacing_xyz"]
        if isinstance(obj, tuple):
            obj = torch.as_tensor(obj).float()
        if device is not None and device != obj.device:
            obj = obj.to(device)
        return obj

    def get_sax_meta_data(self, info_type):
        """Return direction, origin, or spacing from the SAX SimpleITK image."""
        if info_type == "direction":
            return self.views[KEY_SAX_VIEW]["sitk_img"].GetDirection()
        elif info_type == "origin":
            return self.views[KEY_SAX_VIEW]["sitk_img"].GetOrigin()
        elif info_type == "spacing":
            return self.views[KEY_SAX_VIEW]["sitk_img"].GetSpacing()
        raise ValueError(f"Unknown info_type '{info_type}'")

    @staticmethod
    def get_orientation(y, x):
        """Compute in-plane rotation matrix aligning the RV-LV vector to the x-axis."""
        angle = np.arctan2(y, x)
        rotmat = rotation_matrix(rot_z=angle)
        return rotmat

    def _check_apex_base_orientation(self, np_seg):
        self.z_flip = check_apex_base_orientation(np_seg, self.label.LVBP.value)
        print("INFO - CanonicalImage flip z-axis orientation? {}".format(self.z_flip))

    def align(
        self, torch_array, tgt_shape_zyx=None, src_shape_zyx=None, mode="bilinear"
    ):
        """Apply the full alignment transformation to an image tensor."""
        t_coords_aligned_xyz = identity_grid(
            tgt_shape_zyx, device=self.device, do_flip_sequence=self.xyz_sequence
        )
        t_coords_aligned_xyz = torch.cat(
            [
                t_coords_aligned_xyz,
                torch.ones(t_coords_aligned_xyz.shape[:-1] + (1,), device=self.device),
            ],
            dim=-1,
        )
        trans_origin_xyz = torch.eye(4).to(self.device)
        trans_origin_xyz[:3, 3] = (
            torch.tensor(tgt_shape_zyx[::-1], device=self.device) / 2
        ) - 0.5
        trans_m = (
            tr_inv(trans_origin_xyz).T
            @ self.flip_z_m.T
            @ self.flip_y_m.T
            @ tr_inv(self.rot_rv_lv_m)
            @ trans_origin_xyz.T
        )
        t_coords_aligned_xyz = (
            t_coords_aligned_xyz.reshape(tgt_shape_zyx[0], -1, 4) @ trans_m
        )
        warped_arr = self._resample(
            torch_array, t_coords_aligned_xyz, src_shape_zyx, tgt_shape_zyx, mode=mode
        )
        return warped_arr.squeeze()

    def prepare_y_z_flip(self):
        """Initialise the flip matrices for y and z axes."""
        self.flip_y_m = torch.eye(4).to(self.device)
        self.flip_y_m[0, 0] = 1
        self.flip_y_m[1, 1] = -1
        self.flip_z_m = torch.eye(4).to(self.device)
        if self.z_flip:
            self.flip_z_m[2, 2] = -1
