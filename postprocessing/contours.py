"""
Contour extraction and manipulation for cardiac segmentation masks.

Merges:
  - CMRI/contours/common.py  (contour extraction from masks)
  - postprocessing_utils/new_contours.py  (contour manipulation and visualization)
"""

import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import skimage
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from scipy.ndimage import (
    distance_transform_edt as dt,
    binary_dilation,
    generate_binary_structure,
)
from copy import deepcopy

from utils.cardiac import MMS2MRILabel

# =============================================================================
# Section 1: Contour extraction from masks  (from CMRI/contours/common.py)
# =============================================================================


def splinify(contour, s=5, datapoints=512, compute_derivatives=False):
    """
    Fit a periodic B-spline to a contour.

    Parameters
    ----------
    contour : np.ndarray  shape (N, 2)  input contour in (x, y)
    s : float  smoothing factor (5 for mask-derived contours, 0 for Medis contours)
    datapoints : int  number of output points
    compute_derivatives : bool  return tangent vectors if True

    Returns
    -------
    xy : np.ndarray  shape (datapoints, 2)
    derivs : np.ndarray or None  shape (datapoints, 2) tangent vectors
    """
    x, y = contour.T
    tck, u = interpolate.splprep([x, y], s=s, k=3, per=0)
    spline = interpolate.BSpline(tck[0], tck[1], tck[2], extrapolate=True, axis=1)
    xy = spline(np.linspace(0, 1, datapoints, endpoint=True))
    if compute_derivatives:
        dxdt, dydt = interpolate.splev(
            np.linspace(0, 1, datapoints, endpoint=True), tck, der=1
        )
        derivs = np.stack((dxdt.astype(np.float32), dydt.astype(np.float32)))
        return xy.T, derivs.T
    else:
        return xy.T, None


def approximate_contour(
    contour, factor=4, smooth=0.05, periodic=False, compute_derivatives=False
):
    """
    Approximate a contour by upsampling and smoothing with splines.

    Courtesy of UK Biobank toolkit (Wenjia Bai).

    Parameters
    ----------
    contour : np.ndarray  input contour
    factor : int  upsampling factor
    smooth : float  smoothing factor controlling number of spline knots
    periodic : bool  True for closed contours
    compute_derivatives : bool  return first-derivative vectors if True

    Returns
    -------
    contour2 : np.ndarray  upsampled/smoothed contour
    derivatives : np.ndarray or None  (only if compute_derivatives=True)
    """
    N = len(contour)
    dt_val = 1.0 / N
    t = np.arange(N) * dt_val
    x = contour[:, 0]
    y = contour[:, 1]

    r = int(0.5 * N)
    t_pad = np.concatenate((np.arange(-r, 0) * dt_val, t, 1 + np.arange(0, r) * dt_val))
    if periodic:
        x_pad = np.concatenate((x[-r:], x, x[:r]))
        y_pad = np.concatenate((y[-r:], y, y[:r]))
    else:
        x_pad = np.concatenate(
            (np.repeat(x[0], repeats=r), x, np.repeat(x[-1], repeats=r))
        )
        y_pad = np.concatenate(
            (np.repeat(y[0], repeats=r), y, np.repeat(y[-1], repeats=r))
        )

    fx = interpolate.UnivariateSpline(t_pad, x_pad, s=smooth * len(t_pad))
    fy = interpolate.UnivariateSpline(t_pad, y_pad, s=smooth * len(t_pad))

    N2 = N * factor
    dt2 = 1.0 / N2
    t2 = np.arange(N2) * dt2
    x2, y2 = fx(t2), fy(t2)
    contour2 = np.stack((x2, y2), axis=1)
    if compute_derivatives:
        dfx = fx.derivative(1)
        dfy = fy.derivative(1)
        dx2, dy2 = dfx(t2), dfy(t2)
        return contour2, np.stack((dx2, dy2), axis=1)
    return contour2


def contours_from_mask(mask3d, label=MMS2MRILabel):
    """
    Extract and smooth endo/epicardial contours from a 3-D segmentation mask.

    Parameters
    ----------
    mask3d : np.ndarray  (Z, Y, X) integer segmentation
    label : enum  label class with LVBP and LV attributes

    Returns
    -------
    endo_contours, epi_contours : list of np.ndarray (one per slice)
    """
    endo = (mask3d == label.LVBP.value).astype(np.int32)
    epi = np.logical_or(mask3d == label.LVBP.value, mask3d == label.LV.value).astype(
        np.int32
    )
    endo_contours, epi_contours = [], []
    for z in range(mask3d.shape[0]):
        contours, hierarchy = cv2.findContours(
            cv2.inRange(endo[z], 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        if contours is None:
            endo_contours.append(None)
            epi_contours.append(None)
            continue
        endo_contour = contours[0][:, 0, :]
        contours, hierarchy = cv2.findContours(
            cv2.inRange(epi[z], 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        epi_contour = contours[0][:, 0, :]
        endo_contours.append(approximate_contour(endo_contour, periodic=True))
        epi_contours.append(approximate_contour(epi_contour, periodic=True))
    return endo_contours, epi_contours


def get_septum_contour(
    lvm_mask2d, rv_contour, num_dilations=3, kernel=(3, 3), return_rv=False
):
    """
    Find the septum contour as the overlap between dilated LV myocardium and RV contour.

    Parameters
    ----------
    lvm_mask2d : np.ndarray  (Y, X) binary LV myocardium mask
    rv_contour : np.ndarray  (#points, 2) RV contour in (x, y)
    num_dilations : int
    kernel : tuple  morphological kernel size
    return_rv : bool  if True also return the trimmed RV contour

    Returns
    -------
    septum_con : np.ndarray  septum contour points
    septum_indices : np.ndarray  indices into rv_contour
    (rv_contour, rv_indices) : only if return_rv=True
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    lvm_mask2d = cv2.dilate(lvm_mask2d.astype(np.uint8), se, iterations=num_dilations)
    septum_con = []
    septum_indics = []
    new_rv_contour = []
    new_rv_indices = []
    for idx, (x, y) in enumerate(rv_contour):
        x, y = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
        if lvm_mask2d[y, x] == 1:
            septum_con += [[x, y]]
            septum_indics.append(int(idx))
        else:
            new_rv_contour.append([x, y])
            new_rv_indices.append(int(idx))
    septum_con = np.array(septum_con).astype(float)
    septum_indics = np.array(septum_indics).astype(np.int32)
    if return_rv:
        rv_contour = np.array(new_rv_contour).astype(float)
        return (
            septum_con,
            septum_indics,
            rv_contour,
            np.asarray(new_rv_indices).astype(np.int32),
        )
    return septum_con, septum_indics


def create_mask_epi_heart(mask_3d, label=MMS2MRILabel, num_dilations=2, kernel=(2, 2)):
    """
    Create a whole-heart epicardial mask by dilating the RV blood-pool and combining
    with all non-background voxels.

    Parameters
    ----------
    mask_3d : np.ndarray  (Z, Y, X) segmentation
    label : enum  label class with RVBP attribute
    num_dilations : int
    kernel : tuple

    Returns
    -------
    epi_mask : np.ndarray  (Z, Y, X) int32
    """
    epi_mask = np.zeros_like(mask_3d).astype(np.int32)
    for s in range(len(mask_3d)):
        rv_mask2d = mask_3d[s] == MMS2MRILabel.RVBP
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
        rv_mask2d = cv2.dilate(rv_mask2d.astype(np.uint8), se, iterations=num_dilations)
        epi_mask[s, rv_mask2d != 0] = 1
        epi_mask[s, mask_3d[s] != 0] = 1
    return epi_mask


def compute_normals(derivative):
    """
    Compute inward-pointing normals from tangent vectors.

    Parameters
    ----------
    derivative : np.ndarray  (#coords, 2) tangent vectors (dx, dy)

    Returns
    -------
    normals : np.ndarray  (#coords, 2)
    """
    derivative = np.divide(
        derivative, np.linalg.norm(derivative, axis=-1, keepdims=True)
    )
    normals = np.flip(derivative, axis=-1)
    normals[:, 0] = -1 * normals[:, 0]
    return normals


def merge_contour_normal_points(
    con_dict, norm_dict, label=MMS2MRILabel.RVBP, slice_range=None, z_part_dict=None
):
    """
    Flatten per-slice contour and normal dicts into combined 3-D arrays.

    Parameters
    ----------
    con_dict : dict  slice_id -> {label_name -> contour array}
    norm_dict : dict  slice_id -> {label_name -> normals array}
    label : enum member
    slice_range : iterable or None  restrict to these slice indices
    z_part_dict : dict or None  slice_id -> 'apical'/'mid'/'basal'/'apex'

    Returns
    -------
    con_3d : np.ndarray  (#points, 3) (x, y, z)
    norm : np.ndarray  (#points, 2)
    z_part_indices : dict  (only if z_part_dict is not None)
    """
    con_3d, norm = None, []
    z_part_indices = {"apical": [], "mid": [], "basal": []}
    for slice_id, slice_dict in con_dict.items():
        if slice_range is not None and slice_id not in slice_range:
            continue
        if label.name in slice_dict.keys():
            if z_part_dict is not None:
                cpart = z_part_dict[slice_id].replace("apex", "apical")
                start_idx = len(con_3d) if con_3d is not None else 0
                z_part_indices[cpart].extend(
                    list(
                        np.arange(
                            start_idx, start_idx + len(slice_dict[label.name])
                        ).astype(np.int32)
                    )
                )
            con = np.concatenate(
                [
                    slice_dict[label.name],
                    np.full((len(slice_dict[label.name]), 1), slice_id),
                ],
                axis=-1,
            )
            con_3d = (
                np.concatenate([con_3d, con], axis=0) if con_3d is not None else con
            )
            norm.append(norm_dict[slice_id][label.name])

    if z_part_dict is not None:
        for k, v in z_part_indices.items():
            z_part_indices[k] = np.array(z_part_indices[k]).astype(np.int32)
        return con_3d, np.concatenate(norm), z_part_indices
    return con_3d, np.concatenate(norm)


def convert_mask_to_contour(
    mask3d,
    label=MMS2MRILabel,
    epi_mask=None,
    tp=0,
    upfactor=None,
    compute_derivatives=False,
):
    """
    Convert a 3-D segmentation mask to per-slice contours (and optionally normals).

    Parameters
    ----------
    mask3d : np.ndarray  (Z, Y, X) integer segmentation
    label : enum  label class with LVBP, LV, RVBP, EPI, SEP attributes
    epi_mask : np.ndarray or None  pre-computed whole-heart epi mask
    tp : int  timepoint (currently only 0 supported)
    upfactor : int or None  spline upsampling factor
    compute_derivatives : bool  compute and return normal vectors

    Returns
    -------
    contours : dict  slice_id -> {label_name -> contour array}
    contours_as_masks : np.ndarray  (Z, #classes+1, Y, X)
    normals : dict  (only if compute_derivatives=True)
    normals_as_masks : np.ndarray  (only if compute_derivatives=True)
    """
    assert tp == 0
    con_list_out = list()
    contours = defaultdict(dict)
    normals = defaultdict(dict)
    cardiac_structures = [label.LVBP, label.LV, label.RVBP, label.EPI, label.SEP]
    contours_as_masks = np.zeros(
        (mask3d.shape[0], len(cardiac_structures) + 1, mask3d.shape[1], mask3d.shape[2])
    ).astype(np.int32)
    normals_as_masks = np.zeros(
        (
            mask3d.shape[0],
            len(cardiac_structures) + 1,
            mask3d.shape[1],
            mask3d.shape[2],
            2,
        )
    ).astype(float)

    for idx, m_slice in enumerate(mask3d):
        con_slice, norm_slice = {}, {}
        for cardiac_struc in [label.LVBP, label.LV, label.RVBP, label.EPI, label.SEP]:
            cls_lbl, cls_idx = cardiac_struc.name, cardiac_struc.value
            if np.count_nonzero(m_slice == cls_idx) >= 10:
                c = Contour()
                c.fromMask(
                    (m_slice == cls_idx).astype(np.int32),
                    num_dilations=2 if cls_lbl == "RVBP" else None,
                )
                try:
                    if upfactor is not None:
                        c.increase_resolution(
                            upfactor=upfactor, compute_derivatives=compute_derivatives
                        )
                        c.contour = c.contour.round(decimals=3)
                except ValueError:
                    print("Warning: something went wrong with splinify...retrying!")
                    c = Contour()
                    c.fromMask((m_slice == cls_idx).astype(np.int32), num_dilations=2)
                    if upfactor is not None:
                        c.increase_resolution(
                            upfactor=upfactor, compute_derivatives=compute_derivatives
                        )
                        c.contour = c.contour.round(decimals=3)
                con_slice[cls_lbl] = c.contour
                if compute_derivatives:
                    norm_slice[cls_lbl] = c.normal
                    normals_as_masks[idx, cls_idx] = contour_to_mask(
                        c.contour, m_slice.shape, value=c.normal
                    )
                contours_as_masks[idx, cls_idx] = c.contour_to_mask()
                cntr = np.concatenate(
                    (c.contour, np.full((len(c.contour), 1), idx)), axis=-1
                )
                con_list_out.append(
                    [tp, mask3d.shape, cls_idx, cntr.flatten().tolist()]
                )

                if (
                    label.EPI.name not in con_slice.keys()
                    and epi_mask is not None
                    and np.any(epi_mask[idx] != 0)
                ):
                    c = Contour()
                    c.fromMask(epi_mask[idx], num_dilations=None)
                    if upfactor is not None:
                        c.increase_resolution(
                            upfactor=upfactor, compute_derivatives=compute_derivatives
                        )
                        c.contour = c.contour.round(decimals=3)
                        if compute_derivatives:
                            norm_slice[label.EPI.name] = c.normal
                            normals_as_masks[idx, label.EPI.value] = contour_to_mask(
                                c.contour, m_slice.shape, value=c.normal
                            )
                    contours_as_masks[idx, label.EPI.value] = c.contour_to_mask()
                    cntr = np.concatenate(
                        (c.contour, np.full((len(c.contour), 1), idx)), axis=-1
                    )
                    con_list_out.append(
                        [tp, mask3d.shape, label.EPI.value, cntr.flatten().tolist()]
                    )
                    con_slice[label.EPI.name] = cntr

                if (
                    cls_lbl == "RVBP"
                    and cls_lbl in con_slice
                    and np.any(m_slice == label.LV.value)
                ):
                    sep_contour, sep_indices, new_rv_contour, new_rv_indices = (
                        get_septum_contour(
                            (m_slice == label.LV.value).astype(np.int32),
                            con_slice["RVBP"],
                            num_dilations=2,
                            return_rv=True,
                        )
                    )
                    norm_x = gaussian_filter(norm_slice[label.RVBP.name][:, 0], sigma=3)
                    norm_y = gaussian_filter(norm_slice[label.RVBP.name][:, 1], sigma=3)
                    norm_slice[label.RVBP.name] = np.concatenate(
                        [norm_x[:, None], norm_y[:, None]], axis=-1
                    )
                    sep_con_mask = contour_to_mask(sep_contour, m_slice.shape)
                    sep_normals = None
                    if np.count_nonzero(sep_con_mask) > 0 and len(new_rv_indices) > 0:
                        sep_normals = norm_slice[label.RVBP.name][sep_indices].copy()
                        new_rv_contour = new_rv_contour[2:-2]
                        new_rv_indices = new_rv_indices[2:-2]
                        norm_slice[label.RVBP.name] = norm_slice[label.RVBP.name][
                            new_rv_indices
                        ]
                        con_slice[label.RVBP.name] = new_rv_contour
                        contours_as_masks[idx, label.RVBP.value] = contour_to_mask(
                            con_slice[label.RVBP.name], m_slice.shape
                        )
                        normals_as_masks[idx, label.RVBP.value] = contour_to_mask(
                            con_slice[label.RVBP.name],
                            m_slice.shape,
                            value=norm_slice[label.RVBP.name],
                        )

                    if (
                        sep_contour is not None
                        and len(new_rv_indices) > 0
                        and np.count_nonzero(sep_con_mask) > 20
                    ):
                        con_slice[label.SEP.name] = sep_contour
                        if compute_derivatives:
                            sep_normals = -1 * sep_normals
                            norm_slice[label.SEP.name] = sep_normals[4:-4]
                            sep_contour = sep_contour[4:-4]
                            con_slice[label.SEP.name] = con_slice[label.SEP.name][4:-4]
                            normals_as_masks[idx, label.SEP.value] = contour_to_mask(
                                con_slice[label.SEP.name],
                                m_slice.shape,
                                value=norm_slice[label.SEP.name],
                            )
                        contours_as_masks[idx, label.SEP.value] = contour_to_mask(
                            sep_contour, m_slice.shape
                        )
                        sep_contour = np.concatenate(
                            (sep_contour, np.full((len(sep_contour), 1), idx)), axis=-1
                        )
                        con_list_out.append(
                            [
                                tp,
                                mask3d.shape,
                                label.SEP.value,
                                sep_contour.flatten().tolist(),
                            ]
                        )

        contours[idx] = con_slice
        if compute_derivatives:
            normals[idx] = norm_slice

    if compute_derivatives:
        return contours, contours_as_masks, normals, normals_as_masks
    return contours, contours_as_masks


def contour_to_mask(contour, shape_yx, value=None):
    """
    Rasterise a contour onto a 2-D mask.

    Parameters
    ----------
    contour : np.ndarray  (#points, 2+)  (x, y) in first two columns
    shape_yx : tuple  output mask shape (Y, X)
    value : np.ndarray or None  values to assign; None → binary mask

    Returns
    -------
    mask : np.ndarray  shape_yx or shape_yx+(value.shape[-1],)
    """
    coord_xy = np.rint(contour[..., :2]).astype(np.int32)
    if value is None or (value is not None and value.ndim == 1):
        mask = np.zeros(shape_yx).astype(np.uint8 if value is None else float)
    else:
        mask = np.zeros(shape_yx + (value.shape[-1],), float)
    if len(coord_xy) > 0:
        if value is None:
            mask[(coord_xy[:, 1], coord_xy[:, 0])] = 1
        else:
            mask[(coord_xy[:, 1], coord_xy[:, 0])] = value
    return mask


class Contour(object):
    """Thin wrapper around a 2-D contour array with helpers for mask conversion and spline refinement."""

    def __init__(self):
        self.contour = None
        self.mask = None
        self.derivative = None
        self.normal = None
        self.shape = None
        self.origin = "contour"

    @staticmethod
    def _check_cntr_results(cntrs):
        max_idx = 0
        for idx, con in enumerate(cntrs):
            if len(con) > len(cntrs[max_idx]):
                max_idx = idx
        return max_idx

    def fromContour(self, contour, shape):
        self.contour = contour
        self.shape = shape
        self.origin = "contour"

    def fromMask(self, mask, num_dilations=None):
        if num_dilations is not None:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            mask = cv2.dilate(mask.astype(np.uint8), se, iterations=num_dilations)
        cntrs, hierarchy = cv2.findContours(
            cv2.inRange(mask.astype(np.uint8), 1, 1),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE,
        )
        idx = self._check_cntr_results(cntrs)
        self.contour = np.squeeze(cntrs[idx])
        self.origin = "mask"
        self.shape = mask.shape
        self.mask = mask

    def contour_to_mask(self, contour=None, shape_yx=None):
        if contour is None:
            contour = self.contour
        if shape_yx is None:
            shape_yx = self.shape
        return contour_to_mask(contour, shape_yx)

    def as_filled_mask(self, shape=None):
        """Return a filled polygon mask for this contour."""
        if not shape:
            shape = self.shape
        img = np.zeros(shape, float)
        pts = self.contour[:, None].round().astype(np.int32)
        return cv2.fillPoly(img, pts=[pts], color=1.0)

    @property
    def grid_indices(self):
        return self.contour.round().astype(np.int32)

    def showMask(self):
        plt.imshow(self.as_filled_mask(), cmap="gray")

    def showContour(self, c="b", plot_vertices=False):
        contour = self.contour
        plt.plot(contour[:, 0], contour[:, 1], c=c)
        if plot_vertices:
            plt.scatter(contour[:, 0], contour[:, 1], c=c)

    def increase_resolution(
        self, compute_derivatives=False, open_contour=True, upfactor=4
    ):
        """Upsample and smooth the contour using spline interpolation."""
        splinfiy_func = splinify
        samples = int(len(self.contour) * upfactor)
        if self.origin == "processed":
            pass
        elif self.origin == "mask":
            self.contour = splinfiy_func(
                self.contour, 20, samples, compute_derivatives=compute_derivatives
            )
            self.origin = "processed"
        elif self.origin == "contour":
            self.contour = splinfiy_func(
                self.contour, 0, samples, compute_derivatives=compute_derivatives
            )
            self.origin = "processed"
        if compute_derivatives:
            self.contour, self.derivative = self.contour[0], self.contour[1]
            self.derivative = np.divide(
                self.derivative, np.linalg.norm(self.derivative, axis=-1, keepdims=True)
            )
            self.normal = np.flip(self.derivative, axis=-1)
            self.normal[:, 0] = -1 * self.normal[:, 0]

    def equidistant_points(self, segments=8, segments_per_segment=16):
        """Return equidistant points and segment boundary points along the contour."""
        self.increase_resolution()
        cnt = self.contour
        diff = np.diff(np.vstack((cnt[0], cnt, cnt[0])), axis=0)
        cumsum = np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))
        segment_arclength = cumsum[-1] / (segments * segments_per_segment)
        div, mod = np.modf(cumsum / segment_arclength)
        unique_nums, segment_idcs = np.unique(mod, return_index=True)
        segment_idcs = segment_idcs[:-1]
        all_points = cnt[segment_idcs]
        segment_points = cnt[segment_idcs[::segments_per_segment]]
        return all_points, segment_points


def splinify_open_contour(
    contour, s=0, datapoints=512, compute_derivatives=False, order=2
):
    """
    Fit a polynomial to an open contour (e.g. septum centerline).

    Parameters
    ----------
    contour : np.ndarray or Contour  input open contour
    s : float  unused (kept for API compatibility with splinify)
    datapoints : int  output points
    compute_derivatives : bool
    order : int  polynomial degree

    Returns
    -------
    xy : np.ndarray  (datapoints, 2)
    derivs : np.ndarray or None  (datapoints, 2)
    """
    if isinstance(contour, Contour):
        cnt = contour.contour
    else:
        cnt = contour
    x, y = cnt[:, 0], cnt[:, 1]
    x_idx = np.argsort(x)
    y, x = y[x_idx], x[x_idx]
    z = np.polyfit(x, y, deg=order)
    p = np.poly1d(z)
    xnew = np.linspace(x[0], x[-1], datapoints)
    ynew = p(xnew)
    xy = np.stack((xnew.astype(np.float32), ynew.astype(np.float32)))
    ret = xy
    if compute_derivatives:
        dydt = np.gradient(ynew)
        dxdt = np.gradient(xnew)
        derivs = np.stack((dxdt.astype(np.float32), dydt.astype(np.float32)))
        if isinstance(contour, Contour):
            contour.contour = ret.T
            return contour, derivs.T
        else:
            return ret.T, derivs.T
    return ret.T


# =============================================================================
# Section 2: Contour manipulation and visualization  (from new_contours.py)
# =============================================================================


def plot_endo_epi_curves_per_slice(
    Err_T,
    Ecc_T,
    Ell_T,
    endo_T,
    epi_T,
    slices=None,
    time=None,
    sharey_by_component=True,
    suptitle=None,
    figsize_per_row=3.0,
):
    """
    Plot endocardial vs epicardial mean strain curves per slice for RR, CC, LL.

    Parameters
    ----------
    Err_T, Ecc_T, Ell_T : np.ndarray  (T, Z, Y, X)  strain components
    endo_T, epi_T : np.ndarray  (T, Z, Y, X) bool  ring masks
    slices : list or None  z-indices to plot; auto-selected if None
    time : np.ndarray or None  1-D time axis; uses frame indices if None
    sharey_by_component : bool
    suptitle : str or None
    figsize_per_row : float  height in inches per slice row

    Returns
    -------
    fig, axes, curves
    """
    assert Err_T.shape == Ecc_T.shape == Ell_T.shape, "strain arrays must match"
    assert (
        endo_T.shape == epi_T.shape == Err_T.shape
    ), "ring masks must match strain shape"
    T, Z, Y, X = Err_T.shape

    if slices is None:
        has_ring = [z for z in range(Z) if np.any(endo_T[:, z]) or np.any(epi_T[:, z])]
        slices = has_ring if has_ring else list(range(Z))
    n_slices = len(slices)

    if time is None:
        tvec = np.arange(T, dtype=float)
        xlab = "Frame"
    else:
        tvec = np.asarray(time, dtype=float)
        xlab = "Time"

    comps = [("RR", Err_T), ("CC", Ecc_T), ("LL", Ell_T)]
    curves = {z: {"RR": {}, "CC": {}, "LL": {}} for z in slices}
    ymins = {c: +np.inf for c, _ in comps}
    ymaxs = {c: -np.inf for c, _ in comps}

    for z in slices:
        for cname, A in comps:
            for ring_name, M in (("Endocardial", endo_T), ("Epicardial", epi_T)):
                series = np.full((T,), np.nan, dtype=float)
                for t in range(T):
                    m = M[t, z]
                    if np.any(m):
                        vals = A[t, z][m]
                        if np.any(~np.isnan(vals)):
                            series[t] = float(np.nanmean(vals))
                curves[z][cname][ring_name] = series
                if sharey_by_component:
                    v = series[~np.isnan(series)]
                    if v.size:
                        ymins[cname] = min(ymins[cname], np.nanmin(v))
                        ymaxs[cname] = max(ymaxs[cname], np.nanmax(v))

    fig, axes = plt.subplots(
        n_slices, 3, figsize=(12, max(2.0, figsize_per_row * n_slices)), squeeze=False
    )

    for i, z in enumerate(slices):
        for j, (cname, _) in enumerate(comps):
            ax = axes[i, j]
            s_endo = curves[z][cname]["Endocardial"]
            s_epi = curves[z][cname]["Epicardial"]
            ax.plot(tvec, s_endo, label="Endocardial")
            ax.plot(tvec, s_epi, label="Epicardial")
            if sharey_by_component:
                if np.isfinite(ymins[cname]) and np.isfinite(ymaxs[cname]):
                    pad = 0.05 * max(1e-6, ymaxs[cname] - ymins[cname])
                    ax.set_ylim(ymins[cname] - pad, ymaxs[cname] + pad)
            if i == 0:
                ax.set_title(cname)
            if j == 0:
                ax.set_ylabel(f"Slice {z}")
            ax.set_xlabel(xlab)
            if i == 0 and j == 2:
                ax.legend(loc="best", frameon=False)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes, curves


def endo_epi_contours(
    seg_ed_zyx,
    myo_id,
    lvbp_id,
    endo_thick=2,
    epi_thick=2,
    endo_offset=1,
    verbose=False,
):
    """
    Compute thick, disjoint subendocardial and subepicardial band masks within the myocardium.

    Parameters
    ----------
    seg_ed_zyx : np.ndarray  (Z, Y, X) ED segmentation
    myo_id, lvbp_id : int  label values
    endo_thick : int  subendocardial band thickness in voxels
    epi_thick : int  subepicardial band thickness in voxels
    endo_offset : int  step away from the endocardial boundary in voxels
    verbose : bool

    Returns
    -------
    endo_mask : (Z, Y, X) bool
    epi_mask  : (Z, Y, X) bool
    d_endo_all : (Z, Y, X) float  distance to endocardial boundary (voxels) inside MYO
    d_epi_all  : (Z, Y, X) float  distance to epicardial boundary (voxels) inside MYO
    """
    seg = seg_ed_zyx
    myo = seg == myo_id
    lvbp = seg == lvbp_id
    outside = ~(myo | lvbp)

    Z, Y, X = seg.shape
    endo_mask = np.zeros((Z, Y, X), dtype=bool)
    epi_mask = np.zeros((Z, Y, X), dtype=bool)
    d_endo_all = np.full((Z, Y, X), np.nan, dtype=np.float32)
    d_epi_all = np.full((Z, Y, X), np.nan, dtype=np.float32)

    st2 = generate_binary_structure(2, 1)

    for z in range(Z):
        my = myo[z]
        lv = lvbp[z]
        out = outside[z]
        if not my.any():
            continue

        endo_boundary = my & binary_dilation(lv, structure=st2, iterations=1)
        epi_boundary = my & binary_dilation(out, structure=st2, iterations=1)

        if not endo_boundary.any():
            endo_boundary = my & binary_dilation(lv, structure=st2, iterations=2)
        if not epi_boundary.any():
            epi_boundary = my & binary_dilation(out, structure=st2, iterations=2)

        feat_endo = ~endo_boundary
        feat_epi = ~epi_boundary
        d_endo = dt(feat_endo).astype(np.float32)
        d_epi = dt(feat_epi).astype(np.float32)
        d_endo[~my] = np.nan
        d_epi[~my] = np.nan
        d_endo_all[z] = d_endo
        d_epi_all[z] = d_epi

        endo_raw = my & (d_endo >= endo_offset) & (d_endo < endo_offset + endo_thick)
        epi_raw = my & (d_epi >= 0) & (d_epi < epi_thick)

        both = endo_raw & epi_raw
        if np.any(both):
            closer_to_endo = np.nan_to_num(d_endo[both], nan=np.inf) <= np.nan_to_num(
                d_epi[both], nan=np.inf
            )
            idx = np.where(both)
            take_endo = tuple(i[closer_to_endo] for i in idx)
            take_epi = tuple(i[~closer_to_endo] for i in idx)
            endo_raw[both] = False
            epi_raw[both] = False
            endo_raw[take_endo] = True
            epi_raw[take_epi] = True

        endo_mask[z] = endo_raw
        epi_mask[z] = epi_raw

        if verbose:
            print(
                f"[z={z}] myo={my.sum()} endo_boundary={endo_boundary.sum()} "
                f"epi_boundary={epi_boundary.sum()} endo_band={endo_raw.sum()} "
                f"epi_band={epi_raw.sum()} "
                f"d_endo(min,max)={(np.nanmin(d_endo), np.nanmax(d_endo))} "
                f"d_epi(min,max)={(np.nanmin(d_epi), np.nanmax(d_epi))}"
            )

    return endo_mask, epi_mask, d_endo_all, d_epi_all


def create_aha_segment_avg_with_band_rowlayout(
    results_tp,
    pid,
    time=None,
    show=False,
    myo_id=2,
    lvbp_id=1,
    endo_offset=1,
    endo_thick=2,
    epi_offset=0,
    epi_thick=2,
    components=("RR", "CC", "LL"),
    segment_ids=tuple(range(1, 17)),
    overlay_alpha=0.35,
    percentile=10,
):
    """
    Create a Panel figure: one row per slice, columns per strain component.

    For each AHA segment, plot mean(endo, epi) as a solid line with the endo-epi
    range as a shaded band.

    Parameters
    ----------
    results_tp : dict  patient_id -> result dict with keys:
        'array_seg', 'array_img', 'strain_over_time', 'aha_over_time'
    pid : key  patient identifier
    time : np.ndarray or None
    show : bool  call plt.show() if True
    myo_id, lvbp_id : int  label IDs
    endo_offset, endo_thick, epi_offset, epi_thick : int  band parameters
    components : tuple of str  strain components to plot
    segment_ids : tuple of int  AHA segment IDs (1-16)
    overlay_alpha : float  kept for API compatibility
    percentile : int  lower percentile threshold for mean computation within band

    Returns
    -------
    pn.Column  Panel layout
    segment_values : dict  {z: {component: {segment_id: {'mean', 'lower', 'upper'}}}}
    """
    import panel as pn

    R = results_tp[pid]
    seg_Tzyx = R["array_seg"]
    img_Tzyx = R["array_img"]
    strain = R["strain_over_time"]
    aha_Tzyx = R["aha_over_time"]
    T, Z, Y, X = seg_Tzyx.shape

    if strain.ndim == 5 and strain.shape[1] == 3:
        Err_T, Ecc_T, Ell_T = strain[:, 0], strain[:, 1], strain[:, 2]
    elif strain.ndim == 5 and strain.shape[-1] == 3:
        Err_T, Ecc_T, Ell_T = strain[..., 0], strain[..., 1], strain[..., 2]
    else:
        raise ValueError(f"Unexpected strain shape: {strain.shape}")

    comp_map = {"RR": Err_T, "CC": Ecc_T, "LL": Ell_T}
    components = [c for c in components if c in comp_map] or ["RR", "CC", "LL"]
    tvec = np.arange(T, dtype=float) if time is None else np.asarray(time, dtype=float)

    seg_ed_zyx = seg_Tzyx[0]
    aha_ed_zyx = aha_Tzyx[0] if aha_Tzyx.ndim == 4 else aha_Tzyx

    endo_band_zyx, epi_band_zyx, _, _ = endo_epi_contours(
        seg_ed_zyx,
        myo_id,
        lvbp_id,
        endo_thick=endo_thick,
        epi_thick=epi_thick,
        endo_offset=endo_offset,
        verbose=False,
    )

    present_per_slice = []
    for z in range(Z):
        present = np.intersect1d(np.unique(aha_ed_zyx[z]), segment_ids)
        present = present[present > 0]
        present_per_slice.append(present)

    all_sids = (
        np.unique(np.concatenate([p for p in present_per_slice if p.size]))
        if any(p.size for p in present_per_slice)
        else np.array([], int)
    )
    cmap = plt.cm.tab20
    sid_to_color = {sid: cmap((sid % 20) / 20.0) for sid in all_sids}

    def series_over_time(A_tzyx, mask_zyx, pct=10):
        if not np.any(mask_zyx):
            return np.full((T,), np.nan)
        out = np.full((T,), np.nan)
        for tt in range(T):
            vals = A_tzyx[tt][mask_zyx]
            if np.any(~np.isnan(vals)):
                thr = np.nanpercentile(vals, pct)
                vals_filt = vals[vals >= thr]
                if vals_filt.size:
                    out[tt] = float(np.nanmean(vals_filt))
        return out

    segment_values = {z: {c: {} for c in components} for z in range(Z)}
    nrows, ncols = Z, len(components)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 2.6 * max(1, nrows)), squeeze=False
    )

    for z in range(Z):
        seg_ids = present_per_slice[z]
        for j, cname in enumerate(components):
            A = comp_map[cname]
            ax = axes[z, j]
            ymins_local, ymaxs_local = [], []

            for sid in seg_ids:
                smask = aha_ed_zyx[z] == sid
                endo_mask_seg = endo_band_zyx[z] & smask
                epi_mask_seg = epi_band_zyx[z] & smask

                y_endo = series_over_time(A[:, z], endo_mask_seg, pct=percentile)
                y_epi = series_over_time(A[:, z], epi_mask_seg, pct=percentile)

                Y_stack = np.vstack([y_endo, y_epi])
                mean_curve = np.nanmean(Y_stack, axis=0)
                lower = np.nanmin(Y_stack, axis=0)
                upper = np.nanmax(Y_stack, axis=0)

                segment_values[z][cname][int(sid)] = {
                    "mean": mean_curve,
                    "lower": lower,
                    "upper": upper,
                }

                color = sid_to_color.get(int(sid), "k")
                ax.fill_between(
                    tvec, lower, upper, color=color, alpha=0.22, linewidth=0
                )
                ax.plot(tvec, mean_curve, color=color, lw=1.8, label=f"AHA {int(sid)}")
                ymins_local.append(np.nanmin(lower))
                ymaxs_local.append(np.nanmax(upper))

            ax.set_title(f"{cname} — Slice {z}", fontsize=10)
            if z == nrows - 1:
                ax.set_xlabel("Time")
            if j == 0:
                ax.set_ylabel("Strain")
            ax.tick_params(labelsize=8)

            if ymins_local and ymaxs_local:
                lo, hi = np.nanmin(ymins_local), np.nanmax(ymaxs_local)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    pad = 0.05 * (hi - lo)
                    ax.set_ylim(lo - pad, hi + pad)

            if j == ncols - 1 and seg_ids.size:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=sid_to_color.get(int(s), "k"),
                        lw=1.8,
                        label=f"AHA {int(s)}",
                    )
                    for s in seg_ids[:10]
                ]
                band_patch = plt.Line2D(
                    [0], [0], color="gray", lw=6, alpha=0.22, label="Endo-Epi range"
                )
                ax.legend(
                    handles=handles + [band_patch],
                    loc="best",
                    fontsize=8,
                    frameon=False,
                    ncol=1,
                )

    fig.tight_layout()
    layout = [pn.pane.Matplotlib(fig, tight=True)]
    if show:
        plt.show()
    plt.close(fig)
    return pn.Column(*layout), segment_values
