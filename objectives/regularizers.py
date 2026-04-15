"""
Regularization losses and penalties for implicit neural registration.

Standalone functions
--------------------
compute_jacobian_matrix        – autograd-based Jacobian / deformation gradient
compute_balanced_jacobian_loss – fold-prevention via balanced det(J) penalty
compute_bending_energy         – second-order smoothness (thin-plate-spline-like)

Regularization class
--------------------
Mixin that exposes _jacobian_reg and _bending_reg for use inside fit_sequence.
"""

import torch


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad


def compute_bending_energy(input_coords, output, batch_size=None, paperversion=False):
    """Compute the bending energy."""

    jacobian_matrix = compute_jacobian_matrix(input_coords, output, add_identity=False)

    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3, device="cuda")
    dy_xyz = torch.zeros(input_coords.shape[0], 3, 3, device="cuda")
    dz_xyz = torch.zeros(input_coords.shape[0], 3, 3, device="cuda")
    for i in range(3):
        dx_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 0])
        dy_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 1])
        dz_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 2])

    dx_xyz = torch.square(dx_xyz)
    dy_xyz = torch.square(dy_xyz)
    dz_xyz = torch.square(dz_xyz)

    if paperversion:
        yzfac = 2
    else:
        yzfac = 1

    loss = (
        torch.mean(dx_xyz[:, :, 0])
        + torch.mean(dy_xyz[:, :, 1])
        + torch.mean(dz_xyz[:, :, 2])
    )
    loss += (
        2 * torch.mean(dx_xyz[:, :, 1])
        + 2 * torch.mean(dx_xyz[:, :, 2])
        + yzfac * torch.mean(dy_xyz[:, :, 2])
    )

    return loss / batch_size


def compute_balanced_jacobian_loss(input_coords, output, loss_mask=None):
    """Compute the jacobian regularization loss."""
    jac = compute_jacobian_matrix(input_coords, output)
    if loss_mask is not None:
        jac = jac[loss_mask]

    resdet = torch.det(jac)
    resdet = torch.clip(resdet, min=0.1, max=10)
    loss = resdet - 1 + torch.ones(resdet.shape, device="cuda") / resdet - 1
    return torch.mean(loss)


def compute_jacobian_matrix(input_coords, output, add_identity=True):
    """
    Compute the Jacobian matrix of the output wrt the input.

    When add_identity=True, returns the deformation gradient F = I + du/dx.
    When add_identity=False, returns only the displacement gradient du/dx.
    """
    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3, device="cuda")
    for i in range(3):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix


# ---------------------------------------------------------------------------
# Regularization mixin — wraps the above with model-context awareness
# ---------------------------------------------------------------------------


class Regularization:
    """
    Regularization penalties for temporal cardiac registration.

    Designed as a mixin for ImplicitRegistratorSequence. Each method
    encapsulates one penalty term and returns a scalar tensor ready to be
    added to the main loss in fit_sequence.

    Accesses via MRO: self.alpha_jacobian, self.bendreg_paperversion,
    self.batch_size, self._batch_lv_mask.
    """

    def _jacobian_reg(self, xyz, output_rel, batch_idx):
        """
        Balanced Jacobian regularisation: penalise folds inside and outside the LV.

        The foreground (myocardium) weight is alpha_jacobian; the background weight
        is background_weight (much smaller, to allow free motion outside the heart).
        Both are set in registration_config.cfg under [regularization].
        """
        mask_in = self._batch_lv_mask(batch_idx)
        mask_bg = ~mask_in
        reg_in = compute_balanced_jacobian_loss(xyz, output_rel, loss_mask=mask_in)
        reg_bg = compute_balanced_jacobian_loss(xyz, output_rel, loss_mask=mask_bg)
        return self.alpha_jacobian * reg_in + self.background_weight * reg_bg

    def _bending_reg(self, xyz, output_rel):
        """
        Bending-energy regularisation: penalise high second-order spatial gradients.

        Promotes globally smooth deformation fields (thin-plate-spline-like).
        """
        return compute_bending_energy(
            xyz,
            output_rel,
            batch_size=self.batch_size,
            paperversion=self.bendreg_paperversion,
        )
