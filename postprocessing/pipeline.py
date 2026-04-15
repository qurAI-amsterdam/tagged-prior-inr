"""
Temporal post-processing pipeline for implicit cardiac motion registration.

Provides the standalone post_process_sequence_completed function (previously a
method of ImplicitRegistratorSequence in models/models_temporal_new.py).
"""

import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter

from postprocessing.strain import (
    project_strain_with_ed_normals,
    strain_to_engineering_percent,
)
from postprocessing.heart_model import get_heart_model_and_metrics
from utils.cardiac import MMS2MRILabel


def post_process_sequence_completed(
    model,
    spacing_xyz,
    save_dir,
    save_npz=True,
    compute_masks=True,
    compute_physical_dvf=False,
    compute_dice=False,
    kwargs=None,
):
    """
    Temporal post-processing for a fitted ImplicitRegistratorSequence.

    Produces a result dict with per-timepoint warped images, DVFs, deformation
    gradients, strain maps, AHA labels and optional Dice scores.

    Parameters
    ----------
    model : ImplicitRegistratorSequence  fitted temporal registration model
    spacing_xyz : torch.Tensor or np.ndarray  voxel spacing (x, y, z) in mm
    save_dir : str or Path  directory to write result_dict_temporal.npz
    save_npz : bool  write compressed numpy archive if True
    compute_masks : bool  warp segmentation masks (nearest-neighbour) to reference
    compute_physical_dvf : bool  not supported; reserved for future use
    compute_dice : bool  compute Dice overlap vs reference segmentation
    kwargs : dict or None  unused; kept for API compatibility

    Returns
    -------
    result_dict_T : dict
        Keys: 'dvf_over_time', 'array_img', 'array_seg', 'spacing',
              'warped_sax_over_time', 'warped_mask_over_time',
              'jaccobian_over_time', 'jaccobian_det_over_time',
              'strain_over_time', 'strain_over_time_smooth',
              'aha_over_time', 'heart_model_metrics',
              'heart_model_metrics_filtered', 'dice'
    """
    device = getattr(model, "device", "cuda")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    T = model.T
    ref_img_np = model.reference_image.get_sax_image(device=None)
    ref_mask_np = (
        model.reference_image.get_sax_image(image_type="mask", device=None)
        if compute_dice or compute_masks
        else None
    )
    Z, Y, X = ref_img_np.shape

    warped_T = np.zeros((T, Z, Y, X), dtype=np.float32)
    dvf_T = np.zeros((T, Z, Y, X, 3), dtype=np.float32)
    warped_mask_T = np.zeros((T, Z, Y, X), dtype=np.int16) if compute_masks else None
    F_T = np.zeros((T, Z, Y, X, 3, 3), dtype=np.float32)
    Fdet_T = np.zeros((T, Z, Y, X), dtype=np.float32)
    dice_T = None
    array_img = np.zeros((T, Z, Y, X), dtype=np.float32)
    array_seg = np.zeros((T, Z, Y, X), dtype=np.int16)

    if compute_dice:
        from objectives.dice import compute_overlap

        dice_T = np.zeros((T, 3), dtype=np.float32)

    for t in range(T):
        array_img[t] = model.sequence[t].get_sax_image(device=None)
        moving_mask = model.sequence[t].get_sax_image(image_type="mask", device=None)
        array_seg[t] = moving_mask

        warped_t, dvf_t, F, F_det, final_coords_np = model.seq_warp(
            t, mode="bilinear", eval_dvf=True
        )
        warped_T[t] = warped_t
        dvf_T[t] = dvf_t
        F_T[t] = F
        Fdet_T[t] = F_det

        if compute_masks:
            moving_m = torch.from_numpy(moving_mask).to(device)
            # final_coords_np carries a systematic -0.5 voxel shift from the
            # voxel-centre convention in _init_scale_coords.  For bilinear
            # (image) warping this is invisible, but nearest-neighbour
            # (mask) warping rounds to the wrong voxel at boundaries.
            # Adding 0.5 restores integer voxel indices and matches the
            # behaviour of the external scipy.ndimage.map_coordinates path.
            
            # before 
            # torch.from_numpy(final_coords_np).to(device),

            coords_mask = torch.from_numpy(final_coords_np).to(device) + 0.5
            warped_m = model._torch_grid_sampling(
                moving_m,
                coords_mask,
                moving_m.shape,
                mode="nearest",
            )
            warped_mask_T[t] = warped_m.detach().cpu().numpy()

        if compute_dice and warped_mask_T is not None and ref_mask_np is not None:
            dice_T[t] = compute_overlap(
                warped_mask_T[t], ref_mask_np, classes=[1, 2, 3]
            )

    Err_T, Ecc_T, Ell_T, endo_band, epi_band = project_strain_with_ed_normals(
        F_T,
        ref_mask_np,
        MMS2MRILabel.LV.value,
        MMS2MRILabel.LVBP.value,
    )

    if model.convert_to_engineering:
        Err_T = strain_to_engineering_percent(Err_T)
        Ecc_T = strain_to_engineering_percent(Ecc_T)
        Ell_T = strain_to_engineering_percent(Ell_T)

    strain_T = np.stack((Err_T, Ecc_T, Ell_T), axis=-1)

    print("Applying Gaussian smoothing to strain maps...")
    strain_T_smooth = gaussian_filter(
        strain_T,
        sigma=(0, 0, model.strain_sigma, model.strain_sigma, 0),
        mode="nearest",
    )

    try:
        heart_model_aha, heart_model_metrics = get_heart_model_and_metrics(
            T, array_seg, dvf_T, strain_T
        )
        # Reuse precomputed AHA geometry — avoids repeating the expensive
        # per-timepoint HeartModel.heart_model() call for the smoothed variant
        _, heart_model_metrics_smooth = get_heart_model_and_metrics(
            T, array_seg, dvf_T, strain_T_smooth, aha_precomputed=heart_model_aha
        )
    except Exception as e:
        print(f"Error computing heart model metrics: {e}")
        heart_model_aha = None
        heart_model_metrics = {}
        heart_model_metrics_smooth = {}

    if isinstance(spacing_xyz, torch.Tensor):
        spacing_np = spacing_xyz.detach().cpu().numpy()
    else:
        spacing_np = np.asarray(spacing_xyz)

    result_dict_T = {
        "dvf_over_time": dvf_T,
        "array_img": array_img,
        "array_seg": array_seg,
        "spacing": spacing_np,
        "warped_sax_over_time": warped_T,
        "warped_mask_over_time": warped_mask_T,
        "jaccobian_over_time": F_T,
        "jaccobian_det_over_time": Fdet_T,
        "strain_over_time": strain_T,
        "strain_over_time_smooth": strain_T_smooth,
        "aha_over_time": heart_model_aha,
        "heart_model_metrics": heart_model_metrics,
        "heart_model_metrics_filtered": heart_model_metrics_smooth,
        "dice": dice_T,
    }

    if save_npz:
        np.savez(save_dir / "result_dict_temporal.npz", **result_dict_T)

    return result_dict_T
