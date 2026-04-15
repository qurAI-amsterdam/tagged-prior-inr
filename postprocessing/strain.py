"""
Strain computation for cardiac motion registration results.

Merges:
  - CMRI/evaluation/dvf.py  (DVF/Jacobian-based Lagrangian strain)
  - postprocessing_utils/new_strain.py  (ED-normal projection and engineering conversion)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt as dt
from scipy.ndimage import binary_dilation, generate_binary_structure

from utils.cardiac import MMS2MRILabel, compute_slice_range_cstructure

# =============================================================================
# Section 1: DVF / Jacobian-based Lagrangian strain  (from CMRI/evaluation/dvf.py)
# =============================================================================


def lagrange_green_strain_tensor(F: np.ndarray, add_identity=False) -> np.ndarray:
    """
    Compute the Lagrangian Green strain tensor E = 0.5 (F^T F - I).

    Parameters
    ----------
    F : np.ndarray  shape (Z, Y, X, 3, 3)  or  (#points, 3, 3)
        Deformation gradient.  By default the identity is assumed to be
        already included (add_identity=False).
    add_identity : bool  add I before computing if True

    Returns
    -------
    E : np.ndarray  same shape as F
    """
    if add_identity:
        F = F + np.identity(3)
    if F.ndim == 5:
        C = F.transpose((0, 1, 2, 4, 3)) @ F
    elif F.ndim == 3:
        C = F.transpose((0, 2, 1)) @ F
    else:
        raise ValueError(
            "lagrange_green_strain_tensor: unsupported rank (rank={})".format(F.ndim)
        )
    return 0.5 * (C - np.identity(3))


def polar_grid(nx=128, ny=128):
    """
    Generate a polar-coordinate grid of shape (nx, ny).

    Returns
    -------
    phi : np.ndarray  angle in [0, 2π]
    r   : np.ndarray  radius
    """
    x, y = np.meshgrid(
        np.linspace(-nx // 2, nx // 2, nx), np.linspace(-ny // 2, ny // 2, ny)
    )
    phi = (np.arctan2(y, x) + np.pi).T
    r = np.sqrt(x**2 + y**2 + 1e-8)
    return phi, r


def convert_strain_to_polar(E, in_xyz_shape=False):
    """
    Project the Lagrangian strain tensor into radial (Err), circumferential (Ecc)
    and longitudinal (Ell) components.

    Parameters
    ----------
    E : np.ndarray  shape (Z, Y, X, 3, 3) when in_xyz_shape=False
    in_xyz_shape : bool  set True if E has shape (X, Y, Z, 3, 3)

    Returns
    -------
    Err, Ecc, Ell : np.ndarray  each shape matching the spatial dims of E
    """
    if not in_xyz_shape:
        E = E.transpose((2, 1, 0, 3, 4))
    shape_xyz = E.shape[:3]
    phi, _ = polar_grid(*shape_xyz[:2])
    cos = np.cos(phi)
    sin = np.sin(phi)
    Q = np.zeros((cos.shape + (2, 2))).astype(np.float32)
    Q[..., 0, 0], Q[..., 0, 1], Q[..., 1, 0], Q[..., 1, 1] = cos, sin, -sin, cos
    Q = Q[:, :, None]
    Q_T = np.moveaxis(Q, 4, 3)
    E_transformed = Q @ E[..., :2, :2] @ Q_T
    Err, Ecc, Ell = E_transformed[..., 0, 0], E_transformed[..., 1, 1], E[..., 2, 2]
    if not in_xyz_shape:
        Err = Err.transpose((2, 1, 0))
        Ecc = Ecc.transpose((2, 1, 0))
        Ell = Ell.transpose((2, 1, 0))
    return Err, Ecc, Ell


# =============================================================================
# Section 2: ED-normal projection and engineering conversion  (new_strain.py)
# =============================================================================


def ed_basis_and_bands(
    seg_ed_zyx,
    myo_id,
    lvbp_id,
    endo_offset=1,
    endo_thick=2,
    epi_offset=0,
    epi_thick=2,
    spacing_xy=None,
):
    """
    Compute ED-fixed radial/circumferential/longitudinal unit vectors and
    endo/epi thick band masks within the myocardium.

    Parameters
    ----------
    seg_ed_zyx : np.ndarray  (Z, Y, X) ED segmentation
    myo_id, lvbp_id : int  label values
    endo_offset, endo_thick : float  voxel or mm offsets/thicknesses for endo band
    epi_offset, epi_thick  : float  voxel or mm offsets/thicknesses for epi band
    spacing_xy : tuple (sy, sx) in mm; if given, distances are in mm

    Returns
    -------
    rhat, that, lhat : (Z, Y, X, 3) unit vectors (ED-fixed)
    endo_band, epi_band : (Z, Y, X) bool masks (disjoint, within MYO)
    """
    Z, Y, X = seg_ed_zyx.shape
    myo = seg_ed_zyx == myo_id
    lvbp = seg_ed_zyx == lvbp_id
    outside = ~(myo | lvbp)

    if spacing_xy is None:
        sy = sx = 1.0
        u_end_off, u_end_thk = endo_offset, endo_thick
        u_epi_off, u_epi_thk = epi_offset, epi_thick
    else:
        sy, sx = map(float, spacing_xy)
        u_end_off, u_end_thk = float(endo_offset), float(endo_thick)
        u_epi_off, u_epi_thk = float(epi_offset), float(epi_thick)

    rhat = np.zeros((Z, Y, X, 3), dtype=np.float32)
    that = np.zeros_like(rhat)
    lhat = np.zeros_like(rhat)
    endo_band = np.zeros((Z, Y, X), dtype=bool)
    epi_band = np.zeros_like(endo_band)

    st2 = generate_binary_structure(2, 1)

    for z in range(Z):
        my, lv, out = myo[z], lvbp[z], outside[z]
        if not my.any():
            continue

        sdf_endo = dt(~lv, sampling=(sy, sx)) - dt(lv, sampling=(sy, sx))
        gy, gx = np.gradient(sdf_endo, sy, sx)
        n = np.sqrt(gx * gx + gy * gy) + 1e-8
        rx, ry = gx / n, gy / n

        rhat[z, ..., 0] = rx
        rhat[z, ..., 1] = ry
        rhat[z, ..., 2] = 0.0

        that[z, ..., 0] = -ry
        that[z, ..., 1] = rx
        l = np.cross(rhat[z], that[z])
        l /= np.linalg.norm(l, axis=-1, keepdims=True) + 1e-8
        lhat[z] = l

        endo_line = my & binary_dilation(lv, st2, 1)
        epi_line = my & binary_dilation(out, st2, 1)

        d_end = dt(~endo_line, sampling=(sy, sx)).astype(np.float32)
        d_epi = dt(~epi_line, sampling=(sy, sx)).astype(np.float32)
        d_end[~my] = np.nan
        d_epi[~my] = np.nan

        eb = my & (d_end >= u_end_off) & (d_end < u_end_off + u_end_thk)
        pb = my & (d_epi >= u_epi_off) & (d_epi < u_epi_off + u_epi_thk)

        both = eb & pb
        if np.any(both):
            take_e = np.nan_to_num(d_end[both], np.inf) <= np.nan_to_num(
                d_epi[both], np.inf
            )
            idx = np.where(both)
            eb[both] = False
            pb[both] = False
            eb[tuple(i[take_e] for i in idx)] = True
            pb[tuple(i[~take_e] for i in idx)] = True

        endo_band[z] = eb
        epi_band[z] = pb

    return rhat, that, lhat, endo_band, epi_band


def project_strain_with_ed_normals(
    F_Tzyx33,
    seg_ed_zyx,
    myo_id,
    lvbp_id,
    use_inverse_F=False,
    endo_offset=1,
    endo_thick=2,
    epi_offset=0,
    epi_thick=2,
    spacing_xy=None,
    filter_myo=True,
):
    """
    Project the deformation gradient onto ED-fixed radial/circumferential/longitudinal
    directions to obtain Err, Ecc and Ell over the full sequence.

    Parameters
    ----------
    F_Tzyx33 : np.ndarray  (T, Z, Y, X, 3, 3)  deformation gradient (I + du/dX)
    seg_ed_zyx : np.ndarray  (Z, Y, X) ED segmentation
    myo_id, lvbp_id : int
    use_inverse_F : bool  invert F before computing strain
    spacing_xy : tuple (sy, sx) in mm

    Returns
    -------
    Err_T, Ecc_T, Ell_T : (T, Z, Y, X)  with NaN outside myocardium
    endo_band, epi_band : (Z, Y, X) thick masks (ED-fixed, disjoint)
    """
    myo = seg_ed_zyx == myo_id
    rhat, that, lhat, endo_band, epi_band = ed_basis_and_bands(
        seg_ed_zyx,
        myo_id,
        lvbp_id,
        endo_offset=endo_offset,
        endo_thick=endo_thick,
        epi_offset=epi_offset,
        epi_thick=epi_thick,
        spacing_xy=spacing_xy,
    )
    T, Z, Y, X = F_Tzyx33.shape[:4]
    I3 = np.eye(3, dtype=F_Tzyx33.dtype)
    Err_T = np.full((T, Z, Y, X), np.nan, dtype=np.float32)
    Ecc_T = np.full_like(Err_T, np.nan)
    Ell_T = np.full_like(Err_T, np.nan)

    for t in range(T):
        F = F_Tzyx33[t]
        if use_inverse_F:
            F = np.linalg.inv(F)
        E = 0.5 * (np.transpose(F, (0, 1, 2, 4, 3)) @ F - I3)
        Err = np.einsum("...i,...ij,...j->...", rhat, E, rhat)
        Ecc = np.einsum("...i,...ij,...j->...", that, E, that)
        Ell = np.einsum("...i,...ij,...j->...", lhat, E, lhat)
        if filter_myo:
            Err_T[t] = np.where(myo, Err, 0)
            Ecc_T[t] = np.where(myo, Ecc, 0)
            Ell_T[t] = np.where(myo, Ell, 0)
        else:
            Err_T[t] = Err
            Ecc_T[t] = Ecc
            Ell_T[t] = Ell

    return Err_T, Ecc_T, Ell_T, endo_band, epi_band


def strain_to_engineering_percent(E_comp_Tzyx, clip=None):
    """
    Convert a Lagrangian strain component to engineering strain in percent.

    Parameters
    ----------
    E_comp_Tzyx : np.ndarray  (T, Z, Y, X)
    clip : tuple (min, max) or None

    Returns
    -------
    np.ndarray  same shape, values in percent
    """
    out = np.sqrt(1.0 + 2.0 * E_comp_Tzyx) - 1.0
    out *= 100.0
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out
