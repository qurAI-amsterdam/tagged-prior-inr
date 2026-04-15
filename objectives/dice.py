"""
Dice overlap / segmentation-similarity objective.

Provides compute_overlap (previously in kwatsch/dice_loss.py).
"""

import numpy as np


def compute_overlap(mask1, mask2, classes=(1, 2, 3)):
    """
    Compute per-class Dice coefficient between two segmentation arrays.

    Parameters
    ----------
    mask1, mask2 : np.ndarray  integer label arrays of the same shape
    classes : iterable of int  label values to evaluate

    Returns
    -------
    np.ndarray  shape (len(classes),) with Dice scores in [0, 1]
    """
    losses = []
    for c in classes:
        bin_m1 = (mask1 == c).astype(np.int32)
        bin_m2 = (mask2 == c).astype(np.int32)
        nominator = np.sum(bin_m1 * bin_m2)
        denominator = np.sum(bin_m1) + np.sum(bin_m2)
        if denominator > 0:
            losses.append(2 * nominator / denominator)
        else:
            losses.append(0)
    return np.asarray(losses)
