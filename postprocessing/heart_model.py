"""
HeartModel and AHA-segment metric computation.

Merges:
  - utils/heart_model.py  (HeartModel class)
  - get_heart_model_and_metrics extracted from models/models_temporal_new.py
"""

import numpy as np
from scipy import ndimage


class HeartModel:
    """
    Build an AHA 17-segment model from a 3-D cardiac segmentation volume.

    Parameters
    ----------
    segmentation : np.ndarray  (Z, Y, X) integer label array
    """

    def __init__(self, segmentation):
        self.segmentation = segmentation
        self.bloodpool_rv_label = 3
        self.myocardium_lv_label = 2
        self.bloodpool_lv_label = 1

    def radial_slices(
        self, imwidth, imheight, center_x, center_y, labels, offset_radians
    ):
        """
        Assign each in-plane pixel to an AHA segment label via radial sectors.

        Parameters
        ----------
        imwidth, imheight : int  image dimensions
        center_x, center_y : float  LV blood-pool centroid in pixel coordinates
        labels : list of int  AHA segment IDs for the sectors
        offset_radians : float  angular offset for the first sector

        Returns
        -------
        label_map : np.ndarray  (imheight, imwidth) int
        """
        assert isinstance(labels, list)
        phi_offset = offset_radians
        numof_segments = len(labels)

        grid_x, grid_y = np.meshgrid(np.arange(imwidth), np.arange(imheight))
        phi = np.arctan2(grid_x - center_x, grid_y - center_y)
        phi += np.pi + phi_offset
        phi = phi % (2 * np.pi)

        segments = np.linspace(0, 2 * np.pi, numof_segments + 1, True)

        label_map = np.zeros_like(phi, dtype=int)
        for idx in range(numof_segments):
            label = labels[idx]
            l_bound, r_bound = min(segments[idx : idx + 2]), max(
                segments[idx : idx + 2]
            )
            mask = np.logical_and(phi > l_bound, phi <= r_bound)
            label_map[mask] = label

        return label_map

    def get_basal_mid_apical_regions(self, mask, apex=True):
        """
        Divide the LV/RV longitudinal extent into basal, mid, apical (and apex) thirds.

        Parameters
        ----------
        mask : np.ndarray  (Z, Y, X) binary
        apex : bool  if True also return a separate apex region

        Returns
        -------
        tuple of (start, end) slice index pairs: apical, mid, basal[, apex]
        """
        counts = mask.sum((1, 2))
        counts = np.ma.masked_equal(counts, 0)

        valid_indices = np.where(counts.mask == False)[0]
        idx_min, idx_max = valid_indices[0], valid_indices[-1]

        third = (idx_max - idx_min) // 3
        fifteen_per_cent_of_third = third // 6

        basal_range = (idx_max - third, idx_max)
        mid_range = (idx_min + third - 1, idx_max - third)

        if apex:
            apical_range = (
                idx_min + fifteen_per_cent_of_third - 1,
                idx_min + third - 1,
            )
            apex_range = (idx_min, idx_min + fifteen_per_cent_of_third - 1)
            return apical_range, mid_range, basal_range, apex_range
        else:
            apical_range = (idx_min, idx_min + third - 1)
            return apical_range, mid_range, basal_range

    def lv_slice_regions(self, apex=True):
        """Return (apical, mid, basal[, apex]) slice-index tuples for the LV blood pool."""
        mask = self.segmentation == self.bloodpool_lv_label
        return self.get_basal_mid_apical_regions(mask, apex=apex)

    def rv_slice_regions(self):
        """Return (apical, mid, basal) slice-index tuples for the RV blood pool."""
        mask = self.segmentation == self.bloodpool_rv_label
        return self.get_basal_mid_apical_regions(mask)

    def orientation(self):
        """
        Compute the LV-RV orientation angle in radians.

        Returns
        -------
        float  angle of the LV centroid relative to the RV centroid
        """
        lv_regions = self.lv_slice_regions()
        rv_regions = self.rv_slice_regions()

        lv_avg = None
        for region in lv_regions:
            for zpos in range(region[0], region[1] + 1):
                segm_slice = self.segmentation[zpos]
                contrib = (segm_slice == self.bloodpool_lv_label).astype(np.uint8)
                lv_avg = contrib if lv_avg is None else lv_avg + contrib

        rv_avg = None
        for region in rv_regions:
            for zpos in range(region[0], region[1] + 1):
                segm_slice = self.segmentation[zpos]
                contrib = (segm_slice == self.bloodpool_rv_label).astype(np.uint8)
                rv_avg = contrib if rv_avg is None else rv_avg + contrib

        rvy, rvx = ndimage.center_of_mass(rv_avg)
        lvy, lvx = ndimage.center_of_mass(lv_avg)
        return np.arctan2(lvy - rvy, lvx - rvx)

    def heart_model(self, apex=False):
        """
        Build the AHA 17-segment label volume.

        Parameters
        ----------
        apex : bool  include segment 17 (true apex)

        Returns
        -------
        heart_mask : np.ndarray  (Z, Y, X) int  AHA segment labels within LV myocardium
        slice_maps : np.ndarray  (Z, Y, X) int  full radial sector maps
        """
        heart_mask = np.zeros_like(self.segmentation)
        slice_maps = np.zeros_like(self.segmentation)
        lv_regions = self.lv_slice_regions(apex=apex)

        for idx in range(len(self.segmentation)):
            segm_slice = self.segmentation[idx]

            if len(lv_regions) == 4:
                if lv_regions[2][0] <= idx <= lv_regions[2][1]:
                    segment_labels = list(range(1, 7))
                    phi_offset = np.pi / 6
                elif lv_regions[1][0] <= idx < lv_regions[1][1]:
                    segment_labels = list(range(7, 13))
                    phi_offset = np.pi / 6
                elif lv_regions[0][0] <= idx < lv_regions[0][1]:
                    segment_labels = list(range(13, 17))
                    phi_offset = np.pi / 4
                elif lv_regions[3][0] <= idx < lv_regions[3][1]:
                    segment_labels = [17]
                    phi_offset = 0
                else:
                    continue
            else:
                if lv_regions[2][0] <= idx <= lv_regions[2][1]:
                    segment_labels = list(range(1, 7))
                    phi_offset = np.pi / 6
                elif lv_regions[1][0] <= idx < lv_regions[1][1]:
                    segment_labels = list(range(7, 13))
                    phi_offset = np.pi / 6
                elif lv_regions[0][0] <= idx < lv_regions[0][1]:
                    segment_labels = list(range(13, 17))
                    phi_offset = np.pi / 4
                else:
                    continue

            y, x = ndimage.center_of_mass(segm_slice == self.bloodpool_lv_label)
            imheight, imwidth = segm_slice.shape
            orientation = self.orientation()

            slice_map = self.radial_slices(
                imwidth, imheight, x, y, segment_labels, orientation + phi_offset
            )
            slice_maps[idx] = slice_map

            if len(lv_regions) == 4:
                if lv_regions[3][0] <= idx <= lv_regions[2][1]:
                    heart_mask[idx] = slice_map * (
                        segm_slice == self.myocardium_lv_label
                    )
            else:
                if lv_regions[0][0] <= idx <= lv_regions[2][1]:
                    heart_mask[idx] = slice_map * (
                        segm_slice == self.myocardium_lv_label
                    )

        return heart_mask, slice_maps

    def separate_slice(self, imslice, center, radial_spokes):
        """Placeholder for slice separation logic (returns slice unchanged)."""
        assert imslice.dtype == bool
        return imslice

    def calculate_AHA_avgs(self, AHA_seg_array, descriptor, segments=17):
        """
        Compute per-segment statistics of a descriptor array.

        Parameters
        ----------
        AHA_seg_array : np.ndarray  (Z, Y, X) int  AHA segment labels
        descriptor : np.ndarray  same shape  values to aggregate
        segments : int  number of AHA segments

        Returns
        -------
        avg, min, max, std : dict  str(segment_id) -> float
        """
        descriptor_aha_avg = {}
        descriptor_aha_min = {}
        descriptor_aha_max = {}
        descriptor_aha_std = {}

        for seg_id in range(1, segments + 1):
            mask = AHA_seg_array == seg_id
            if descriptor[mask].size == 0:
                continue
            descriptor_aha_avg[str(seg_id)] = descriptor[mask].mean()
            descriptor_aha_min[str(seg_id)] = descriptor[mask].min()
            descriptor_aha_max[str(seg_id)] = descriptor[mask].max()
            descriptor_aha_std[str(seg_id)] = descriptor[mask].std()

        return (
            descriptor_aha_avg,
            descriptor_aha_min,
            descriptor_aha_max,
            descriptor_aha_std,
        )


def get_heart_model_and_metrics(
    T, array_seg, dvf_over_time, strain_over_time, aha_precomputed=None
):
    """
    Compute AHA 17-segment labels and per-segment DVF/strain metrics over time.

    Parameters
    ----------
    T : int  number of timepoints
    array_seg : np.ndarray  (T, Z, Y, X) segmentation over time
    dvf_over_time : np.ndarray  (T, Z, Y, X, 3) displacement vector field
    strain_over_time : np.ndarray  (T, Z, Y, X, 3) strain components (RR, CC, LL)
    aha_precomputed : np.ndarray or None  (T, Z, Y, X) precomputed AHA labels.
        When provided the expensive per-timepoint HeartModel.heart_model() geometry
        step is skipped and these labels are used directly.

    Returns
    -------
    aha_17_seg : np.ndarray  (T, Z, Y, X) AHA labels (consistent across time)
    metrics : dict  keys 'dvf', 'rr', 'cc', 'll'; each has sub-keys 'avg', 'min', 'max', 'std'
              each sub-key is a list of per-timepoint dicts
    """
    aha_17_seg = []
    metrics = {
        "dvf": {"avg": [], "min": [], "max": [], "std": []},
        "rr": {"avg": [], "min": [], "max": [], "std": []},
        "cc": {"avg": [], "min": [], "max": [], "std": []},
        "ll": {"avg": [], "min": [], "max": [], "std": []},
    }

    for t in range(T):
        hm = HeartModel(array_seg[t])
        if aha_precomputed is not None:
            AHA_seg = aha_precomputed[t]
        else:
            AHA_seg, _ = hm.heart_model(apex=True)
            aha_17_seg.append(AHA_seg)

        dvf_avg, dvf_min, dvf_max, dvf_std = hm.calculate_AHA_avgs(
            AHA_seg, dvf_over_time[t], segments=17
        )
        metrics["dvf"]["avg"].append(dvf_avg)
        metrics["dvf"]["min"].append(dvf_min)
        metrics["dvf"]["max"].append(dvf_max)
        metrics["dvf"]["std"].append(dvf_std)

        rr_avg, rr_min, rr_max, rr_std = hm.calculate_AHA_avgs(
            AHA_seg, strain_over_time[t][..., 0], segments=17
        )
        cc_avg, cc_min, cc_max, cc_std = hm.calculate_AHA_avgs(
            AHA_seg, strain_over_time[t][..., 1], segments=17
        )
        ll_avg, ll_min, ll_max, ll_std = hm.calculate_AHA_avgs(
            AHA_seg, strain_over_time[t][..., 2], segments=17
        )

        for key, vals in zip(
            ["rr", "cc", "ll"],
            [
                (rr_avg, rr_min, rr_max, rr_std),
                (cc_avg, cc_min, cc_max, cc_std),
                (ll_avg, ll_min, ll_max, ll_std),
            ],
        ):
            metrics[key]["avg"].append(vals[0])
            metrics[key]["min"].append(vals[1])
            metrics[key]["max"].append(vals[2])
            metrics[key]["std"].append(vals[3])

    for key in ["dvf", "rr", "cc", "ll"]:
        for stat in ["avg", "min", "max", "std"]:
            metrics[key][stat] = np.array(metrics[key][stat])

    if aha_precomputed is not None:
        return aha_precomputed, metrics
    return np.array(aha_17_seg), metrics
