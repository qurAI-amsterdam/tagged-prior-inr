# Implicit Neural Registration for Cardiac Motion Estimation with a Temporal Prior

SIREN-based implicit neural registration for 4D cardiac cine MRI sequences. The network takes (x, y, z, t) coordinates as input and predicts the displacement field that maps each timepoint to a fixed reference frame. Optionally initialised from a pre-trained temporal prior.

---

## Method overview



---

## Repository structure

```
run_registration.py          # Entry point — loads config, loops over patients
registration_config.cfg      # All hyperparameters and paths

models/
  temporal.py                # Training loop (fit_sequence)
  setup.py                   # Network, optimiser, regularisation init
  coords.py                  # Coordinate handling and temporal encoding
  forward.py                 # Chunked inference and Jacobian computation
  warp.py                    # Image and coordinate warping

networks/
  siren.py                   # SIREN architecture

objectives/
  ncc.py                     # Normalised cross-correlation loss
  regularizers.py            # Jacobian and bending energy penalties

canonical/
  alignment.py               # Canonical orientation alignment entry points
  image.py                   # CanonicalImage — single-frame wrapper
  sequence.py                # CanonicalSequence — 4D cine wrapper
  transforms.py              # Resampling utilities

postprocessing/
  pipeline.py                # Post-processing entry point
  strain.py                  # Strain computation from DVF
  heart_model.py             # AHA-17 segment assignment and metrics
  contours.py                # Contour extraction

utils/
  cardiac.py                 # Cardiac label definitions and mask utilities
  coords.py                  # View-key constants
  io.py                      # Image loading and experiment folder helpers
  config.py                  # Settings serialisation

create_prior_temporal.ipynb  # Notebook for training/exporting the temporal prior
```

---

## Setup

```bash
pip install torch SimpleITK numpy scipy tqdm pyyaml
```

---

## Running registration

```bash
python run_registration.py --config registration_config.cfg --gpu 0
```

**Multi-GPU / multi-worker** (one worker per GPU):

```bash
python run_registration.py --config registration_config.cfg --gpu 0 --worker-id 0 --n-workers 2
python run_registration.py --config registration_config.cfg --gpu 1 --worker-id 1 --n-workers 2
```

Outputs are written to:
```
{root}/registration_output_experiments/{experiment_folder_name}/{patient_id}/
  models/final_model.pth     # saved network weights
  *.npz                      # DVF, strain, segmentation arrays
```

---

## Configuration

All settings live in `registration_config.cfg`. Key parameters:

| Section | Key | Description |
|---|---|---|
| `[training]` | `lr` | Learning rate |
| `[training]` | `batch_size` | Coordinates sampled per step |
| `[training]` | `temporal_delta` | Step size for temporal sampling |
| `[network]` | `layers` | Network layer widths, e.g. `[4, 256, 256, 256, 3]` |
| `[network]` | `omega` | SIREN frequency parameter |
| `[network]` | `use_prior` | Load weights from a pre-trained prior |
| `[network]` | `path_to_prior` | Path to prior `.pth` file |
| `[regularization]` | `jacobian_regularization` | Enable Jacobian fold penalty |
| `[regularization]` | `alpha_jacobian` | Jacobian loss weight |
| `[regularization]` | `alpha_bending` | Bending energy weight (0 = off) |
| `[regularization]` | `early_stopping` | Stop when loss plateaus |
| `[experiment]` | `epochs` | Maximum training epochs |
| `[experiment]` | `seed` | Random seed |
| `[general settings]` | `save_net` | Save network weights after training |
| `[general settings]` | `debug` | Run on a single patient only |

---

## Prior initialisation

A temporal prior (network weights pre-trained on a population) can be used to initialise registration before fine-tuning on a new patient:

```ini
[network]
use_prior = True
path_to_prior = /path/to/prior_weights.pth
```

See `create_prior_temporal.ipynb` for how to train and export a prior from a set of patients.
