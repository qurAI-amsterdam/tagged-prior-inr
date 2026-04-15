"""
ForwardPass: SIREN network displacement prediction for full set of coordinates
and Jacobian utilities.

Provides _predict_displacement (memory-efficient coordinate-to-displacement
inference over 3-D coordinate sets), _predict_displacement_temporal (same
for 4-D (x,y,z,t) coordinates), scale_jacobian (converts normalised
Jacobians to physical mm units), and _select_indices (random batch sampling).
"""

import torch
import numpy as np

from objectives.regularizers import compute_jacobian_matrix


class ForwardPass:
    """Predict coordinate displacements via the SIREN network, with Jacobian scaling."""

    def _predict_displacement(
        self,
        coordinates,
        chunk_size=10_000,
        eval_dvf=False,
        spacing_xyz=None,
        img_shape=None,
        compute_physical_dvf=False,
    ):
        """
        Predict displacement vectors for 3-D *coordinates* in memory-safe chunks.

        Passes coordinate points (not images) through the SIREN network and
        accumulates the predicted displacement vectors.

        Parameters
        ----------
        coordinates          : torch.Tensor  (N, 3)  normalised spatial coords
        chunk_size           : int  max coordinates per network call
        eval_dvf             : bool  compute Jacobian via autograd if True
        spacing_xyz          : tuple  voxel spacing (needed if compute_physical_dvf)
        img_shape            : tuple  image shape   (needed if compute_physical_dvf)
        compute_physical_dvf : bool  also return physical-space Jacobian

        Returns
        -------
        transform_rel    : torch.Tensor  (N, 3)  predicted displacements
        dvf_jacobian     : torch.Tensor  (N, 3, 3)  or None
        dvf_jacobian_phys: torch.Tensor  (N, 3, 3)  only when compute_physical_dvf
        """
        n_chunks = (coordinates.shape[0] + chunk_size - 1) // chunk_size
        transform_rel = dvf_jacobian = dvf_jac_no_identity = None

        for chunk in torch.chunk(coordinates, chunks=n_chunks, dim=0):
            if eval_dvf:
                chunk = chunk.requires_grad_(True)
                out = self.network(chunk)
                jac = compute_jacobian_matrix(chunk, out, add_identity=True)
                # jac_0 (no identity) = F - I; avoids a second autograd backward pass
                jac_0 = jac - torch.eye(3, device=jac.device).unsqueeze(0)
                dvf_jacobian = (
                    jac.detach().cpu()
                    if dvf_jacobian is None
                    else torch.cat([dvf_jacobian, jac.detach().cpu()], dim=0)
                )
                dvf_jac_no_identity = (
                    jac_0.detach().cpu()
                    if dvf_jac_no_identity is None
                    else torch.cat([dvf_jac_no_identity, jac_0.detach().cpu()], dim=0)
                )
            else:
                self.network.eval()
                with torch.no_grad():
                    out = self.network(chunk)

            out = out.detach().cpu()
            transform_rel = (
                out if transform_rel is None else torch.cat([transform_rel, out], dim=0)
            )
            del chunk

        torch.cuda.empty_cache()

        if compute_physical_dvf:
            dvf_jac_phys = self.scale_jacobian(
                dvf_jac_no_identity,
                image_shape=np.flip(img_shape),
                spacing_xyz=spacing_xyz,
            )
            return transform_rel, dvf_jacobian, dvf_jac_phys

        return transform_rel, dvf_jacobian

    def scale_jacobian(
        self, J_norm: torch.Tensor, image_shape: tuple, spacing_xyz: tuple
    ) -> torch.Tensor:
        """
        Convert a Jacobian from normalised model coordinates to physical (mm) space.

        J_phys = S_out @ J_norm @ S_in  where S encodes the voxel-to-mm scaling.

        Parameters
        ----------
        J_norm       : torch.Tensor  (..., 3, 3)
        image_shape  : tuple  (nx, ny, nz) voxel counts
        spacing_xyz  : tuple  (sx, sy, sz) mm per voxel

        Returns
        -------
        J_phys : torch.Tensor  (..., 3, 3)
        """
        nx, ny, nz = image_shape
        sx, sy, sz = spacing_xyz
        a = torch.tensor([(nx * sx) / 2.0, (ny * sy) / 2.0, (nz * sz) / 2.0])
        S_in = torch.diag(1.0 / a).to(J_norm.device)
        S_out = torch.diag(a).to(J_norm.device)
        return S_out.unsqueeze(0) @ J_norm @ S_in.unsqueeze(0)

    def _select_indices(self, possible_coordinate_tensor, batch_size=10_000):
        """Sample two independent random batches of row indices (for cycle-loss pairs)."""
        n = possible_coordinate_tensor.shape[0]
        idx = torch.randperm(n, device="cuda")[:batch_size]
        idx_rev = torch.randperm(n, device="cuda")[:batch_size]
        coords = possible_coordinate_tensor[idx, :].requires_grad_(True)
        coords_rev = possible_coordinate_tensor[idx_rev, :].requires_grad_(True)
        return idx, idx_rev, coords, coords_rev

    # ── Temporal displacement prediction ──────────────────────────────────────

    def _predict_displacement_temporal(
        self,
        coordinates_xt: torch.Tensor,
        chunk_size: int = 10_000,
        eval_dvf: bool = False,
        img_shape: tuple = None,
    ):
        """
        Predict displacement vectors for 4-D (x, y, z, t) coordinate tensors.

        Identical to _predict_displacement but accepts a time column appended
        to the spatial coordinates. Gradients are taken w.r.t. the xyz part
        only, so the Jacobian remains a 3×3 spatial deformation gradient.

        Parameters
        ----------
        coordinates_xt : torch.Tensor  (N, 4)  [x, y, z, t_norm]
        chunk_size     : int
        eval_dvf       : bool  compute spatial Jacobian via autograd
        img_shape      : tuple (Z, Y, X)  unused here, reserved for physical DVF

        Returns
        -------
        transform_rel  : torch.Tensor  (N, 3)   predicted displacements
        dvf_jacobian   : torch.Tensor  (N, 3, 3) or None
        """
        n_chunks = (coordinates_xt.shape[0] + chunk_size - 1) // chunk_size
        transform_rel = None
        dvf_jacobian = None

        for chunk in torch.chunk(coordinates_xt, chunks=n_chunks, dim=0):
            xyz = chunk[..., :3].to(self.device)
            tcol = chunk[..., 3:].to(self.device).detach()  # 1 col or 2K Fourier cols

            if eval_dvf:
                xyz = xyz.requires_grad_(True)
                net_in = torch.cat([xyz, tcol], dim=-1)
                out_rel = self.network(net_in)
                jac = compute_jacobian_matrix(xyz, out_rel, add_identity=True)
                jac_cpu = jac.detach().cpu()
                dvf_jacobian = (
                    jac_cpu
                    if dvf_jacobian is None
                    else torch.cat([dvf_jacobian, jac_cpu], dim=0)
                )
            else:
                self.network.eval()
                with torch.no_grad():
                    out_rel = self.network(torch.cat([xyz, tcol], dim=-1))

            out_cpu = out_rel.detach().cpu()
            transform_rel = (
                out_cpu
                if transform_rel is None
                else torch.cat([transform_rel, out_cpu], dim=0)
            )

            del chunk, xyz, tcol, out_rel

        torch.cuda.empty_cache()
        return transform_rel, dvf_jacobian
