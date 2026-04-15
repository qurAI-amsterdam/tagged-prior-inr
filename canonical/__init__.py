"""
Canonical-space representations and alignment for cardiac MRI.

Module layout
-------------
image.py      – CanonicalImage     (single 3-D frame in canonical orientation)
sequence.py   – CanonicalSequence  (4-D cine sequence of CanonicalImages)
transforms.py – grid and resampling utilities used by the above
alignment.py  – pipeline functions that produce canonical objects from raw
                SimpleITK images (get_canonical_sequence_aligned, etc.)
"""

from canonical.image import CanonicalImage, MRILabel
from canonical.sequence import CanonicalSequence
from canonical.alignment import (
    get_canonical_sequence_aligned,
    get_canonical_image_aligned,
    get_sequence_objects,
    get_image_objects,
    get_rv_lv_rot_matrix,
    get_3d_rotation_info,
    get_3d_roation_info,
    convert_to_binary_and_get_bbox,
    convert_to_binary_and_get_bbox_sequence,
)
