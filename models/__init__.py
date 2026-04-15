"""
Implicit neural registration models for cardiac MRI motion estimation.

Module layout
-------------
temporal.py       – ImplicitRegistratorSequence  (training loop only)
setup.py          – RegistrationSetup    (__init__, default args, network/optimizer)
coords.py         – Coordinates          (base coord ops)
                   TemporalCoordinates   (temporal overrides + batch sampling)
forward.py        – ForwardPass          (_predict_displacement, _predict_displacement_temporal, Jacobian)
warp.py           – Warp                 (image/coord warping, seq_warp)

Regularization lives in objectives/regularizers.py alongside the standalone
penalty functions (compute_jacobian_matrix, compute_balanced_jacobian_loss, etc.).
"""

from models.temporal import ImplicitRegistratorSequence
