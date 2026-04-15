"""
Canonical-space alignment helpers for cardiac MRI volumes.

Provides functions to load, crop, normalise and align 3D/4D SAX and LAX
cardiac MRI images into a canonical orientation prior to registration.
"""

import numpy as np
import torch
import SimpleITK as sitk
import SimpleITK

from utils.cardiac import (
    MMS2MRILabel,
    get_center,
    check_apex_base_orientation,
    rotation_matrix,
)
from utils.coords import KEY_SAX_VIEW, KEY_SAX_SEG_VIEW
from canonical.image import CanonicalImage
from canonical.sequence import CanonicalSequence

# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------


def convert_to_binary_and_get_bbox(seg):
    """
    Compute the bounding box of the non-zero region of a SimpleITK segmentation.

    Parameters
    ----------
    seg : sitk.Image  3-D segmentation image

    Returns
    -------
    tuple  (start_x, start_y, start_z, size_x, size_y, size_z) in voxel indices
    """
    np_seg = sitk.GetArrayFromImage(seg)
    seg_binary = np_seg > 0
    seg_binary_sitk = sitk.Cast(
        sitk.GetImageFromArray(seg_binary.astype(int)), sitk.sitkUInt8
    )
    seg_binary_sitk.CopyInformation(seg)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(seg_binary_sitk)
    return label_shape_filter.GetBoundingBox(1)


def convert_to_binary_and_get_bbox_sequence(np_seg):
    """
    Compute the bounding box of the non-zero region of a NumPy segmentation array.

    Parameters
    ----------
    np_seg : np.ndarray  segmentation volume (any shape)

    Returns
    -------
    tuple  (start_x, start_y, start_z, size_x, size_y, size_z) in voxel indices
    """
    seg_binary = (np_seg > 0).astype(np.uint8)
    seg_binary_sitk = sitk.GetImageFromArray(seg_binary)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(seg_binary_sitk)
    return label_shape_filter.GetBoundingBox(1)


# ---------------------------------------------------------------------------
# RV/LV rotation
# ---------------------------------------------------------------------------


def get_rv_lv_rot_matrix(
    np_seg, label, device="cpu", angle=None, do_flip=None, load_dict=None
):
    """
    Compute the rotation matrix that aligns the RV–LV axis to a canonical
    horizontal orientation and determine whether a y-flip is required.

    Parameters
    ----------
    np_seg : np.ndarray  3-D segmentation array (ZYX)
    label             : enum with LVBP and RVBP attributes
    device : str        torch device string
    angle : float|None  override rotation angle (radians); computed if None
    do_flip : bool      initial flip flag passed to CanonicalImage
    load_dict : dict    must contain the ``sitk_fixed3d_*`` keys consumed by
                        ``get_image_objects``

    Returns
    -------
    tuple  (rot_rv_lv_m, need_flip, rotated_seg)
    """
    lv_com_zyx = get_center(np_seg, label.LVBP.value)
    rv_com_zyx = get_center(np_seg, label.RVBP.value)

    rot_rv_lv_m = torch.eye(4)
    vec_lv_rv = rv_com_zyx - lv_com_zyx
    angle = angle if angle is not None else np.arctan2(vec_lv_rv[1], vec_lv_rv[2])
    rotmat = rotation_matrix(rot_z=angle)
    rot_rv_lv_m[:3, :3] = torch.from_numpy(rotmat).float()

    load_dict["rv_lv_rot_matrix"] = rot_rv_lv_m.to(device)
    load_dict["do_flip"] = do_flip

    data_dict = get_image_objects(
        load_dict,
        do_normalize=True,
        device="cuda",
        include_contours=True,
    )

    img_fixed = data_dict["fixed_img"]
    rotated_seg = img_fixed.get_sax_image(image_type="mask")

    y_index = 1
    need_flip = False

    lv_com_zyx = get_center(rotated_seg, label.LVBP.value)

    porc_above_rv = np.sum(
        rotated_seg[:, : int(lv_com_zyx[y_index]), :] == label.RVBP.value
    )
    porc_below_rv = np.sum(
        rotated_seg[:, int(lv_com_zyx[y_index]) :, :] == label.RVBP.value
    )

    if porc_above_rv > porc_below_rv:
        need_flip = True
        rot_rv_lv_m = torch.matmul(
            rot_rv_lv_m.to(device),
            torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            .float()
            .to(device),
        )

    return rot_rv_lv_m.to(device), need_flip, rotated_seg


# ---------------------------------------------------------------------------
# Image-object constructors
# ---------------------------------------------------------------------------


def get_image_objects(
    data_dict_volume,
    device="cpu",
    do_normalize=False,
    include_contours=False,
):
    """
    Wrap a pair of fixed/moving 3-D volumes into aligned ``CanonicalImage``
    objects.

    Parameters
    ----------
    data_dict_volume : dict   must contain ``sitk_fixed3d_img_sax``,
                              ``sitk_mov3d_img_sax``, ``sitk_fixed3d_mask_sax``,
                              ``sitk_mov3d_mask_sax``, ``do_flip``,
                              ``rv_lv_rot_matrix``.
    device : str
    do_normalize : bool
    include_contours : bool passed to ``align_images``

    Returns
    -------
    dict  ``{"fixed_img": CanonicalImage, "moving_img": CanonicalImage, "spacing": tuple}``
    """
    sitk_fixed3d_img_sax = data_dict_volume["sitk_fixed3d_img_sax"]
    sitk_mov3d_img_sax = data_dict_volume["sitk_mov3d_img_sax"]
    sitk_fixed3d_mask_sax = data_dict_volume["sitk_fixed3d_mask_sax"]
    sitk_mov3d_mask_sax = data_dict_volume["sitk_mov3d_mask_sax"]

    do_flip = data_dict_volume["do_flip"]
    rv_lv_rot_matrix = data_dict_volume["rv_lv_rot_matrix"]

    fixed_cimage = CanonicalImage(
        sitk_fixed3d_img_sax,
        sitk_fixed3d_mask_sax,
        label=MMS2MRILabel,
        device=device,
        normalize=True,
        z_flip=do_flip,
    )
    fixed_cimage.align_images(
        rv_lv_rot_matrix=rv_lv_rot_matrix, include_contours=include_contours
    )

    moving_cimage = CanonicalImage(
        sitk_mov3d_img_sax,
        sitk_mov3d_mask_sax,
        label=MMS2MRILabel,
        device=device,
        normalize=do_normalize,
        source_obj=fixed_cimage,
    )
    moving_cimage.align_images(
        rv_lv_rot_matrix=rv_lv_rot_matrix, include_contours=include_contours
    )

    spacing = sitk_mov3d_img_sax.GetSpacing()

    return {
        "fixed_img": fixed_cimage,
        "moving_img": moving_cimage,
        "spacing": spacing,
    }


def get_sequence_objects(data_dict_volume, device="cpu"):
    """
    Wrap a 4-D cine volume dict into a ``CanonicalSequence`` object.

    Parameters
    ----------
    data_dict_volume : dict   must contain ``img4d_sax``, ``seg4d_sax``,
                              ``do_flip``, ``rv_lv_rot_matrix``.
    device : str

    Returns
    -------
    dict  ``{"sequence": CanonicalSequence, "spacing": tuple}``
    """
    if "img4d_sax" in data_dict_volume:
        print("INFO - Detected 4D cine. Wrapping into CanonicalSequence")

        cseq = CanonicalSequence(
            image_seq=data_dict_volume["img4d_sax"],
            mask_seq=data_dict_volume["seg4d_sax"],
            label=MMS2MRILabel,
            device=device,
            normalize=True,
            do_flip=data_dict_volume["do_flip"],
            rv_lv_rot_matrix=data_dict_volume["rv_lv_rot_matrix"],
        )

        spacing = cseq[0].get_spacing("sax")

        return {
            "sequence": cseq,
            "spacing": spacing,
        }


# ---------------------------------------------------------------------------
# 3-D rotation info extraction
# ---------------------------------------------------------------------------


def get_3d_rotation_info(
    img,
    mask,
    lax_img=None,
    lax_seg=None,
    tp_fixed=None,
    tp_moving=None,
    swap_labels=False,
    crop_ROI=False,
    device="cuda",
):
    """
    Extract the canonical-orientation flip flag and RV/LV rotation matrix from
    a pair of 3-D time-point slices of a 4-D sequence.

    Parameters
    ----------
    img : sitk.Image    4-D SAX cine (X, Y, Z, T)
    mask : sitk.Image   4-D SAX segmentation
    lax_img : sitk.Image|None
    lax_seg : sitk.Image|None
    tp_fixed : int      time-point index for the fixed frame
    tp_moving : int     time-point index for the moving frame
    swap_labels : bool  swap LVBP and RVBP labels (for ACDC compatibility)
    crop_ROI : bool     crop to the bounding box before computing the rotation
    device : str

    Returns
    -------
    tuple  (do_flip, rv_lv_rot_matrix)
    """
    sitk_fixed_3d_img_sax = img[:, :, :, tp_fixed]
    sitk_mov_3d_img_sax = img[:, :, :, tp_moving]
    sitk_fixed_3d_mask_sax = mask[:, :, :, tp_fixed]
    sitk_mov_3d_mask_sax = mask[:, :, :, tp_moving]

    if swap_labels:
        np_sax_seg = sitk.GetArrayFromImage(sitk_fixed_3d_mask_sax)
        np_sax_seg_temporal = np.where(
            np_sax_seg == MMS2MRILabel.LVBP.value, MMS2MRILabel.RVBP.value, np_sax_seg
        )
        sitk_fixed3d_mask_sax_new = np.where(
            np_sax_seg == MMS2MRILabel.RVBP.value,
            MMS2MRILabel.LVBP.value,
            np_sax_seg_temporal,
        )
        sitk_fixed3d_mask_sax_new = sitk.GetImageFromArray(sitk_fixed3d_mask_sax_new)
        sitk_fixed3d_mask_sax_new.CopyInformation(sitk_fixed_3d_mask_sax)
        sitk_fixed_3d_mask_sax = sitk_fixed3d_mask_sax_new

    if crop_ROI:
        padding = [15, 10, 5]
        bounding_box = convert_to_binary_and_get_bbox(sitk_fixed_3d_mask_sax)
        start_x, start_y, start_z, size_x, size_y, size_z = bounding_box
        start_x = max(0, start_x - padding[0])
        start_y = max(0, start_y - padding[1])
        start_z = max(0, start_z - padding[2])
        size_x = min(
            sitk_fixed_3d_img_sax.GetSize()[0] - start_x, size_x + 2 * padding[0]
        )
        size_y = min(
            sitk_fixed_3d_img_sax.GetSize()[1] - start_y, size_y + 2 * padding[1]
        )
        size_z = min(
            sitk_fixed_3d_img_sax.GetSize()[2] - start_z, size_z + 2 * padding[2]
        )
        sitk_fixed_3d_img_sax = sitk_fixed_3d_img_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_fixed_3d_mask_sax = sitk_fixed_3d_mask_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_mov_3d_img_sax = sitk_mov_3d_img_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_mov_3d_mask_sax = sitk_mov_3d_mask_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]

    if (
        max(sitk.GetArrayFromImage(sitk_fixed_3d_img_sax).flatten()) > 1.0
        or min(sitk.GetArrayFromImage(sitk_fixed_3d_img_sax).flatten()) < 0.0
    ):
        print("Normalizing images to be between 0 and 1")
        sitk_fixed_3d_img_sax = sitk.RescaleIntensity(sitk_fixed_3d_img_sax, 0, 1)
        sitk_mov_3d_img_sax = sitk.RescaleIntensity(sitk_mov_3d_img_sax, 0, 1)

    print(
        "Images in sitk shape (flipped when converted to arr):",
        sitk_fixed_3d_img_sax.GetSize(),
    )
    load_dict = {
        "sitk_fixed3d_img_sax": sitk_fixed_3d_img_sax,
        "sitk_mov3d_img_sax": sitk_mov_3d_img_sax,
        "sitk_fixed3d_mask_sax": sitk_fixed_3d_mask_sax,
        "sitk_mov3d_mask_sax": sitk_mov_3d_mask_sax,
    }
    np_seg_sax = SimpleITK.GetArrayFromImage(sitk_fixed_3d_mask_sax).astype(np.int32)
    print("Images in array shape:", np_seg_sax.shape)
    do_flip = check_apex_base_orientation(np_seg_sax, MMS2MRILabel.LVBP.value)
    rv_lv_rot_matrix, _y_flip, _ = get_rv_lv_rot_matrix(
        np_seg_sax,
        label=MMS2MRILabel,
        device=device,
        do_flip=do_flip,
        load_dict=load_dict,
    )

    load_dict["do_flip"] = do_flip
    load_dict["rv_lv_rot_matrix"] = rv_lv_rot_matrix

    return do_flip, rv_lv_rot_matrix


# Legacy spelling preserved as alias.
get_3d_roation_info = get_3d_rotation_info


# ---------------------------------------------------------------------------
# High-level alignment entry points
# ---------------------------------------------------------------------------


def get_canonical_sequence_aligned(
    img,
    mask,
    swap_labels=False,
    crop_ROI=False,
    device="cuda",
):
    """
    Load, normalise, optionally crop, and canonically orient a 4-D SAX cine
    sequence.

    Parameters
    ----------
    img : sitk.Image    4-D SAX cine (X, Y, Z, T)
    mask : sitk.Image   4-D SAX segmentation
    swap_labels : bool  swap LVBP/RVBP (for ACDC datasets)
    crop_ROI : bool     crop volumes to the heart bounding box before alignment
    device : str

    Returns
    -------
    dict  as returned by ``get_sequence_objects``
    """

    def _minmax_norm_4d(img4d):
        arr = sitk.GetArrayFromImage(img4d).astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax <= vmin:
            return img4d
        scale = (vmax - vmin) + 1e-8
        frames = []
        for t in range(arr.shape[0]):
            frame3d = sitk.GetImageFromArray((arr[t] - vmin) / scale)
            frames.append(frame3d)
        out4d = sitk.JoinSeries(frames)
        out4d.CopyInformation(img4d)
        return out4d

    X, Y, Z, T = img.GetSize()
    print(f"INFO - Detected 4D cine with T={T}")

    do_flip, rv_lv_rot_matrix = get_3d_rotation_info(
        img,
        mask,
        tp_fixed=T - 1,
        tp_moving=T // 2,
        crop_ROI=True,
    )

    if crop_ROI:
        mask3d_t0 = mask[:, :, :, 0]
        padding = [15, 10, 5]
        sx, sy, sz, dx, dy, dz = convert_to_binary_and_get_bbox(mask3d_t0)
        sx = max(0, sx - padding[0])
        sy = max(0, sy - padding[1])
        sz = max(0, sz - padding[2])
        dx = min(X - sx, dx + 2 * padding[0])
        dy = min(Y - sy, dy + 2 * padding[1])
        dz = min(Z - sz, dz + 2 * padding[2])
        img = img[sx : sx + dx, sy : sy + dy, sz : sz + dz, 0:T]
        mask = mask[sx : sx + dx, sy : sy + dy, sz : sz + dz, 0:T]

    img = _minmax_norm_4d(img)

    load_dict = {
        "img4d_sax": img,
        "seg4d_sax": mask,
        "do_flip": do_flip,
        "rv_lv_rot_matrix": rv_lv_rot_matrix,
    }

    return get_sequence_objects(load_dict, device=device)


def get_canonical_image_aligned(
    img,
    mask,
    tp_fixed=None,
    tp_moving=None,
    swap_labels=False,
    crop_ROI=False,
    device="cuda",
):
    """
    Align a pair of 3-D volumes (extracted from time-points of a 4-D sequence)
    into canonical orientation and return wrapped ``CanonicalImage`` objects.

    Parameters
    ----------
    img : sitk.Image    4-D SAX cine (X, Y, Z, T)
    mask : sitk.Image   4-D SAX segmentation
    tp_fixed : int      fixed time-point index
    tp_moving : int     moving time-point index
    swap_labels : bool
    crop_ROI : bool
    device : str

    Returns
    -------
    dict  as returned by ``get_image_objects``
    """
    sitk_fixed_3d_img_sax = img[:, :, :, tp_fixed]
    sitk_mov_3d_img_sax = img[:, :, :, tp_moving]
    sitk_fixed_3d_mask_sax = mask[:, :, :, tp_fixed]
    sitk_mov_3d_mask_sax = mask[:, :, :, tp_moving]

    if swap_labels:
        np_sax_seg = sitk.GetArrayFromImage(sitk_fixed_3d_mask_sax)
        np_sax_seg_temporal = np.where(
            np_sax_seg == MMS2MRILabel.LVBP.value, MMS2MRILabel.RVBP.value, np_sax_seg
        )
        sitk_fixed3d_mask_sax_new = np.where(
            np_sax_seg == MMS2MRILabel.RVBP.value,
            MMS2MRILabel.LVBP.value,
            np_sax_seg_temporal,
        )
        sitk_fixed3d_mask_sax_new = sitk.GetImageFromArray(sitk_fixed3d_mask_sax_new)
        sitk_fixed3d_mask_sax_new.CopyInformation(sitk_fixed_3d_mask_sax)
        sitk_fixed_3d_mask_sax = sitk_fixed3d_mask_sax_new

    if crop_ROI:
        padding = [15, 10, 5]
        bounding_box = convert_to_binary_and_get_bbox(sitk_fixed_3d_mask_sax)
        start_x, start_y, start_z, size_x, size_y, size_z = bounding_box
        start_x = max(0, start_x - padding[0])
        start_y = max(0, start_y - padding[1])
        start_z = max(0, start_z - padding[2])
        size_x = min(
            sitk_fixed_3d_img_sax.GetSize()[0] - start_x, size_x + 2 * padding[0]
        )
        size_y = min(
            sitk_fixed_3d_img_sax.GetSize()[1] - start_y, size_y + 2 * padding[1]
        )
        size_z = min(
            sitk_fixed_3d_img_sax.GetSize()[2] - start_z, size_z + 2 * padding[2]
        )
        sitk_fixed_3d_img_sax = sitk_fixed_3d_img_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_fixed_3d_mask_sax = sitk_fixed_3d_mask_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_mov_3d_img_sax = sitk_mov_3d_img_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]
        sitk_mov_3d_mask_sax = sitk_mov_3d_mask_sax[
            start_x : start_x + size_x,
            start_y : start_y + size_y,
            start_z : start_z + size_z,
        ]

    if (
        max(sitk.GetArrayFromImage(sitk_fixed_3d_img_sax).flatten()) > 1.0
        or min(sitk.GetArrayFromImage(sitk_fixed_3d_img_sax).flatten()) < 0.0
    ):
        print("Normalizing images to be between 0 and 1")
        sitk_fixed_3d_img_sax = sitk.RescaleIntensity(sitk_fixed_3d_img_sax, 0, 1)
        sitk_mov_3d_img_sax = sitk.RescaleIntensity(sitk_mov_3d_img_sax, 0, 1)
    print(
        "Images in sitk shape (flipped when converted to arr):",
        sitk_fixed_3d_img_sax.GetSize(),
    )
    load_dict = {
        "sitk_fixed3d_img_sax": sitk_fixed_3d_img_sax,
        "sitk_mov3d_img_sax": sitk_mov_3d_img_sax,
        "sitk_fixed3d_mask_sax": sitk_fixed_3d_mask_sax,
        "sitk_mov3d_mask_sax": sitk_mov_3d_mask_sax,
    }
    np_seg_sax = SimpleITK.GetArrayFromImage(sitk_fixed_3d_mask_sax).astype(np.int32)
    print("Images in array shape:", np_seg_sax.shape)
    do_flip = check_apex_base_orientation(np_seg_sax, MMS2MRILabel.LVBP.value)
    rv_lv_rot_matrix, _y_flip, _ = get_rv_lv_rot_matrix(
        np_seg_sax,
        label=MMS2MRILabel,
        device=device,
        do_flip=do_flip,
        load_dict=load_dict,
    )

    load_dict["do_flip"] = do_flip
    load_dict["rv_lv_rot_matrix"] = rv_lv_rot_matrix

    return get_image_objects(
        load_dict,
        do_normalize=True,
        device="cuda",
        include_contours=True,
    )
