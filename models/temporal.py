"""
ImplicitRegistratorSequence — SIREN-based temporal cardiac motion registrator.

Responsibilities are distributed across dedicated modules:

    setup.py          → RegistrationSetup     (__init__, default args, network/optimizer)
    coords.py         → TemporalCoordinates   (coordinate collection, scaling, batch sampling)
    forward.py        → ForwardPass           (chunked inference, Jacobian, temporal forward)
    warp.py           → Warp                  (image and coordinate warping, seq_warp)
    objectives/regularizers.py → Regularization  (Jacobian, bending energy)

This file contains only the training loop (fit_sequence).
"""

import torch
import tqdm

from models.setup import RegistrationSetup
from models.coords import TemporalCoordinates
from models.forward import ForwardPass
from models.warp import Warp
from objectives.regularizers import Regularization


class ImplicitRegistratorSequence(
    RegistrationSetup,
    TemporalCoordinates,
    ForwardPass,
    Warp,
    Regularization,
):
    """
    Temporal implicit neural registrator for 4-D cardiac cine sequences.

    Construction, coordinate handling, network inference, warping, and
    regularization are all handled by the inherited classes above.
    This class only contains the training loop.
    """

    def fit_sequence(self, epochs: int = None):
        """
        Train the network on the full cine sequence.

        At each step a random timepoint is drawn and the network is optimised
        to map that frame to the fixed reference using the configured similarity
        loss. Optional regularisation terms are applied when their weights are
        non-zero.

        Per-component loss histories are stored in self.loss_components after
        training and displayed live in the progress bar via EMA-smoothed values.
        """
        if epochs is None:
            epochs = self.epochs

        torch.manual_seed(self.seed)
        self.loss_list = [0.0] * epochs

        # Per-component history (raw values, one entry per epoch)
        self.loss_components = {
            "sim":  [0.0] * epochs,
            "jac":  [0.0] * epochs,
            "bend": [0.0] * epochs,
        }

        # EMA state for the progress bar display (alpha=0.05 → slow/smooth)
        _ema_alpha = 0.05
        _ema = {k: 0.0 for k in self.loss_components}

        pbar = tqdm.tqdm(range(epochs), desc="Temporal fit", total=epochs)
        saved_loss = 0.0
        loss_trigger = 0
        loss_patience = max(1, epochs // 10)

        for i in pbar:
            # ── Sample timepoint and coordinate batch ────────────────────────
            t_idx = torch.randint(0, self.T, (1,)).item()
            coords_xt, batch_idx = self._sample_batch(t_idx, self.batch_size)

            # Keep spatial coords as a leaf tensor so autograd can differentiate.
            # tcol may be 1 column (scalar t) or 2K columns (Fourier encoding).
            xyz = coords_xt[:, :3].detach().clone().requires_grad_(True)
            tcol = coords_xt[:, 3:].detach()
            coords_xt = torch.cat([xyz, tcol], dim=-1)

            # ── Similarity loss ──────────────────────────────────────────────
            target_img = self.reference_image.get_sax_image(device=self.device)
            source_img = self.sequence[t_idx].get_sax_image(device=self.device)
            target_vals = self._interpolate(
                target_img, coords_xt[:, :3], self.spacing_xyz
            )
            output_rel = self.network(coords_xt)
            warped_vals = self._interpolate(
                source_img, coords_xt[:, :3] + output_rel, self.spacing_xyz
            )
            l_sim = self.criterion(warped_vals, target_vals)
            loss = l_sim

            # ── Regularisation terms ─────────────────────────────────────────
            l_jac = l_bend = loss.new_zeros(1).squeeze()

            if (
                getattr(self, "jacobian_regularization", False)
                and self.alpha_jacobian > 0
            ):
                l_jac = self._jacobian_reg(xyz, output_rel, batch_idx)
                loss = loss + l_jac

            if self.alpha_bending > 0:
                l_bend = self.alpha_bending * self._bending_reg(xyz, output_rel)
                loss = loss + l_bend

            # ── Optimiser step ───────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ── Record raw component values ──────────────────────────────────
            raw = {
                "sim":  l_sim.item(),
                "jac":  l_jac.item(),
                "bend": l_bend.item(),
            }
            for k, v in raw.items():
                self.loss_components[k][i] = v

            self.loss_list[i] = loss.item()

            # ── Update EMA and progress bar ──────────────────────────────────
            if i == 0:
                _ema = dict(raw)  # initialise EMA to first values
            else:
                for k, v in raw.items():
                    _ema[k] = (1 - _ema_alpha) * _ema[k] + _ema_alpha * v

            # Only show non-zero components to keep the bar readable
            postfix = {"sim": f"{_ema['sim']:.4f}"}
            postfix.update({
                k: f"{_ema[k]:.4f}"
                for k in ("jac", "bend")
                if raw[k] != 0.0
            })
            pbar.set_postfix(postfix)

            # ── Early stopping ───────────────────────────────────────────────
            if self.early_stopping:
                cur_loss = float(loss.detach().cpu())
                if abs(cur_loss - saved_loss) < 1e-3 or cur_loss > saved_loss:
                    loss_trigger += 1
                    if loss_trigger > loss_patience:
                        print(f"Early stopping at epoch {i}")
                        self.stopped_at_epoch = i + 1
                        return
                else:
                    loss_trigger = 0
                saved_loss = cur_loss

        self.stopped_at_epoch = epochs
