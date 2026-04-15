"""
Spatial transformation utilities for canonical-space coordinate grids.

Provides identity-grid constructors, homogeneous coordinate helpers, and
voxel-to-world transform generators (previously in kwatsch/common.py).
"""

import torch
import numpy as np
import SimpleITK as sitk


def identity_grid(dims, device="cuda", dtype=torch.float32, do_flip_sequence=False):
    """
    Build a flat coordinate tensor spanning the given dimensions.

    Parameters
    ----------
    dims : tuple of int  (z, y, x)
    device : str
    dtype : torch dtype
    do_flip_sequence : bool  if True, reverse the xyz ordering of the last dim

    Returns
    -------
    coordinate_tensor : torch.Tensor  shape (#coords, 3)
    """
    coordinate_tensor = [torch.arange(s, dtype=dtype, device=device) for s in dims]
    coordinate_tensor = torch.meshgrid(coordinate_tensor, indexing="ij")
    if do_flip_sequence:
        coordinate_tensor = torch.stack(coordinate_tensor[::-1], dim=len(dims))
    else:
        coordinate_tensor = torch.stack(coordinate_tensor, dim=len(dims))
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    return coordinate_tensor.to(device=device)


def make_identity_grid(shape: tuple, stackdim="last", device="cpu"):
    """
    Create a 3-D identity coordinate grid with reversed (x, y, z) ordering.

    Parameters
    ----------
    shape : tuple  (z, y, x)
    stackdim : 'last', 'first', or int
    device : str

    Returns
    -------
    torch.Tensor  shape (*shape, 3) with (x, y, z) coords in last dim
    """
    if isinstance(stackdim, int):
        dim = stackdim
    elif stackdim == "last":
        dim = len(shape)
    elif stackdim == "first":
        dim = 0
    else:
        raise ValueError("Incorrect stackdim given.")

    coords = [torch.arange(0, s, dtype=torch.float32, device=device) for s in shape]
    grids = torch.meshgrid(coords, indexing="ij")
    # Reverse ordering: shape is (z, y, x) but we want (x, y, z) coords
    return torch.stack(grids[::-1], dim=dim)


def make_homogeneous_identity_grid(target_shape: tuple, device="cpu"):
    """
    Create a 3-D identity grid with an appended homogeneous (ones) coordinate.

    Parameters
    ----------
    target_shape : tuple  (z, y, x)
    device : str

    Returns
    -------
    torch.Tensor  shape (*target_shape, 4)
    """
    grid = make_identity_grid(target_shape, device=device)
    h_coords = torch.ones(grid.shape[:-1] + (1,), device=device)
    return torch.cat([grid, h_coords], dim=-1)


# Keep legacy spelling as alias
make_homegeneous_identity_grid = make_homogeneous_identity_grid


def get_voxel_to_world_transforms(tgt_image: sitk.Image, device="cpu"):
    """
    Extract homogeneous rotation, scale and translation matrices from a SimpleITK image.

    Returns
    -------
    tr_R_tgt : torch.Tensor  (4, 4) rotation
    tr_S_tgt : torch.Tensor  (4, 4) voxel scaling
    tr_T_tgt : torch.Tensor  (4, 4) origin translation
    """
    spacing_tgt = tgt_image.GetSpacing()
    dim = len(tgt_image.GetSize())

    R_tgt = np.zeros((4, 4))
    R_tgt[:dim, :dim] = np.reshape(
        np.asarray(tgt_image.GetDirection()), (dim, dim)
    ).astype(np.float32)
    R_tgt[dim:, dim:] = 1

    T_tgt = np.eye(4)
    T_tgt[:dim, dim] = np.asarray(tgt_image.GetOrigin()).astype(np.float32)

    S_tgt = np.eye(4)
    S_tgt[:dim, :dim] = np.diag(np.asarray(spacing_tgt)).astype(np.float32)

    tr_R_tgt = torch.from_numpy(R_tgt.astype(np.float32)).to(device)
    tr_S_tgt = torch.from_numpy(S_tgt.astype(np.float32)).to(device)
    tr_T_tgt = torch.from_numpy(T_tgt.astype(np.float32)).to(device)
    return tr_R_tgt, tr_S_tgt, tr_T_tgt


def execute_resampling(
    src_3d_tensor: torch.Tensor,
    transformed_ident_grid: torch.Tensor,
    mode="bilinear",
    do_detach=True,
) -> torch.Tensor:
    """
    Resample *src_3d_tensor* at locations defined by *transformed_ident_grid*.

    Parameters
    ----------
    src_3d_tensor : torch.Tensor  (Z, Y, X) or (1, 1, Z, Y, X)
    transformed_ident_grid : torch.Tensor  normalised grid for grid_sample
    mode : str  'bilinear' or 'nearest'
    do_detach : bool  detach result and move to CPU when True

    Returns
    -------
    torch.Tensor  resampled volume
    """
    if src_3d_tensor.dim() == 3:
        src_3d_tensor = src_3d_tensor[None, None]
    elif src_3d_tensor.dim() == 2:
        src_3d_tensor = src_3d_tensor[None, None, None]
    if transformed_ident_grid.dim() == 4:
        transformed_ident_grid = transformed_ident_grid[None]
    elif transformed_ident_grid.dim() == 3:
        transformed_ident_grid = transformed_ident_grid[None, None]
    if src_3d_tensor.device != transformed_ident_grid.device:
        src_3d_tensor = src_3d_tensor.to(transformed_ident_grid.device)

    resampled_img = torch.nn.functional.grid_sample(
        src_3d_tensor,
        transformed_ident_grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    if do_detach:
        return resampled_img.detach().cpu().squeeze()
    return resampled_img
