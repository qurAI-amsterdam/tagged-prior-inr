"""
Cardiac MRI label definitions and image-processing helpers.

Full content of the former CMRI/general.py, re-exported here so that the
rest of the codebase can import from utils.cardiac instead.
"""

import numpy as np
import SimpleITK as sitk
import cv2 as cv
from math import sin, cos
from scipy import ndimage
from enum import Enum


class MMS2MRILabel(Enum):
    """Label IDs used in the MMS2 cardiac MRI segmentation dataset."""

    BG = 0
    LVBP = 1
    LV = 2
    RVBP = 3
    SEP = 4
    EPI = 5


def get_center(arr: np.ndarray, label: int = 1, spacing=None, do_flip_sequence=False):
    """
    Return the centre of mass of voxels with the given label.

    Parameters
    ----------
    arr : np.ndarray
    label : int  label value to select
    spacing : optional array-like  multiply result by spacing (mm)
    do_flip_sequence : bool  reverse xyz -> zyx if True

    Returns
    -------
    center : np.ndarray  float32
    """
    lbl_arr = arr == int(label)
    arr_masked = arr * lbl_arr
    center = np.asarray(ndimage.center_of_mass(arr_masked)).astype(np.float32)
    if do_flip_sequence:
        center = center[::-1]
    if spacing is not None:
        return np.multiply(center, spacing)
    return center


def rotation_matrix(rot_x=0, rot_y=0, rot_z=0):
    """Return a 3×3 rotation matrix for the given Euler angles (radians)."""
    rotmat = np.eye(3)
    if rot_x:
        rotmat = rotmat @ [
            (1, 0, 0),
            (0, cos(rot_x), -sin(rot_x)),
            (0, sin(rot_x), cos(rot_x)),
        ]
    if rot_y:
        rotmat = rotmat @ [
            (cos(rot_y), 0, sin(rot_y)),
            (0, 1, 0),
            (-sin(rot_y), 0, cos(rot_y)),
        ]
    if rot_z:
        rotmat = rotmat @ [
            (cos(rot_z), -sin(rot_z), 0),
            (sin(rot_z), cos(rot_z), 0),
            (0, 0, 1),
        ]
    return rotmat


def check_apex_base_orientation(np_seg, label_value):
    """
    Determine whether the z-axis needs to be flipped so that apex=low index.

    Returns True if a flip is required.
    """
    zmask = (np_seg == label_value).any((1, 2))
    num_seg_slices = np.count_nonzero(zmask)
    segmask = (np_seg[zmask] == label_value).astype(np.int32)
    z_cmas = round(get_center(segmask)[0])
    return z_cmas < num_seg_slices / 2


def blur_mask(mask, kernel_shape=(31, 31), num_dilations=2, apply_blur=True):
    """
    Morphologically dilate (and optionally Gaussian-blur) a binary mask.

    Operates slice-by-slice if the mask is 3-D.
    """
    mask = mask.astype(np.float32)
    if mask.ndim == 3:
        for idx in range(len(mask)):
            mask[idx] = blur_mask(
                mask[idx],
                kernel_shape,
                num_dilations=num_dilations,
                apply_blur=apply_blur,
            )
    else:
        assert mask.ndim == 2
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_shape)
        mask = cv.dilate(mask, se, iterations=num_dilations)
        if apply_blur:
            mask = cv.GaussianBlur(mask, kernel_shape, sigmaX=0)
    return mask


def normalize_image(img3d: np.ndarray, percentile=(0, 100)) -> np.ndarray:
    """Percentile-clip and min-max normalize a 3-D image to [0, 1]."""
    i_min, i_max = np.percentile(img3d, percentile)
    div_term = i_max - i_min
    norm_img3d = np.divide(
        (img3d - i_min),
        div_term,
        out=np.ones_like(img3d, dtype=np.float64),
        where=div_term != 0,
    )
    return norm_img3d.clip(0, 1)


def determine_three_slices(np_mask_zyx: np.ndarray):
    """
    Select apical, mid and basal slice indices from a 3-D segmentation mask.

    Returns
    -------
    z_pos : dict  keys 'apical', 'mid', 'basal'
    z_idx : np.ndarray  shape (3,)
    """
    assert np_mask_zyx.ndim == 3
    z_mask = (np_mask_zyx != 0).any((1, 2)).astype(np.int32)
    mask_idcs = np.where(z_mask > 0)
    z_min, z_max = min(mask_idcs[0]), max(mask_idcs[0])
    num_slices = z_max - z_min + 1
    z_pos = {}
    z_pos["apical"] = int(z_min + round((num_slices - 1) * 0.25))
    z_pos["mid"] = int(z_min + round((num_slices - 1) * 0.5))
    z_pos["basal"] = int(z_min + round((num_slices - 1) * 0.75))
    z_idx = np.array([z_pos["apical"], z_pos["mid"], z_pos["basal"]]).astype(np.int32)
    return z_pos, z_idx


def sitk_save(
    fname: str,
    arr: np.ndarray,
    spacing_zyx=None,
    dtype=np.float32,
    direction=None,
    origin=None,
    source_image=None,
):
    """Save a NumPy array as a SimpleITK image file.

    Supports 3-D and 4-D arrays. When *source_image* is supplied its metadata
    (spacing, direction, origin) is copied to the output image.
    """
    if arr.ndim == 4:
        volumes = [
            sitk.GetImageFromArray(arr[v].astype(dtype), False)
            for v in range(arr.shape[0])
        ]
        img = sitk.JoinSeries(volumes)
    else:
        img = sitk.GetImageFromArray(arr.astype(dtype))
    if source_image is not None:
        img.CopyInformation(source_image)
    else:
        if spacing_zyx is not None:
            img.SetSpacing(spacing_zyx[::-1])
        if direction is not None:
            img.SetDirection(direction)
        if origin is not None:
            img.SetOrigin(origin)
    sitk.WriteImage(img, fname, True)


def compute_slice_range_cstructure(np_mask, label, percentile=(0, 100)):
    """
    Return the range of z-slices that contain the given cardiac structure label.

    Parameters
    ----------
    np_mask : np.ndarray  shape (Z, Y, X)
    label   : enum member with a .value attribute
    percentile : tuple  clip the range to these percentiles

    Returns
    -------
    np.ndarray  of integer slice indices
    """
    z_mask = (np_mask == label.value).any((1, 2)).astype(np.int32)
    mask_idcs = np.where(z_mask > 0)
    z_min, z_max = min(mask_idcs[0]), max(mask_idcs[0])
    slice_range = np.arange(z_min, z_max + 1)
    z_min, z_max = np.percentile(slice_range, percentile).astype(np.int32)
    return np.arange(z_min, z_max + 1)
