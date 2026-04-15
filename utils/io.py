"""
Data I/O utilities for loading cardiac MRI images and segmentations.

Provides helpers for reading 4D cine SAX/LAX images from disk and for
constructing experiment output folder paths.
"""

import os
import time
import SimpleITK


def get_experiments_folder(path_to_data, folder_name=None, addition=None):
    """
    Construct a path for storing registration experiment outputs.

    Parameters
    ----------
    path_to_data : str or Path  root data directory
    folder_name : str or None   explicit subfolder name; auto-generated from
                                current date/time when None
    addition : str or None      optional suffix appended after the timestamp

    Returns
    -------
    str  path to the experiments folder (not created by this function)
    """
    if folder_name is None:
        date_time = time.strftime("%Y%m%d_%H%M")
        folder_name = "Experiment_" + date_time
        if addition is not None:
            folder_name = folder_name + "_" + addition

    path_to_experiments = os.path.join(
        path_to_data, "registration_output_experiments", folder_name
    )

    return path_to_experiments


def get_images_with_segmentations(path_to_sax, path_to_seg_sax, **kwargs):
    """
    Load SAX 4D cine image and segmentation from disk.

    Parameters
    ----------
    path_to_sax : str       full path to the SAX 4D cine image file
    path_to_seg_sax : str   full path to the SAX segmentation file
    **kwargs                unused; accepted for call-site compatibility

    Returns
    -------
    tuple  (img4d_sax, seg4d_sax)
    """
    img4d_sax = SimpleITK.ReadImage(path_to_sax)
    seg4d_sax = SimpleITK.ReadImage(path_to_seg_sax)
    return img4d_sax, seg4d_sax
