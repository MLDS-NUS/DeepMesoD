"""Full-covariance window likelihood for vector finite-range factors."""

import torch
import torch.nn.functional as F


def vector_patch_grid(context, patch_size, rows, components):
    patches = F.unfold(context, kernel_size=patch_size).transpose(1, 2)
    return patches.reshape(context.shape[0], rows, components, patch_size, patch_size)


def vector_window_gaussian_nll(factor, context, current, following, dt, mean_drift,
                               beta, coarse_volume, jitter=1e-8, patches=None):
    components = factor.n_components
    if patches is None:
        patches = vector_patch_grid(
            context, factor.patch_size, factor.n_likelihood_rows, components
        )
    mobility = factor.local_covariance(factor.blocks_on_patch_grid(patches))
    residual = following - current - dt[:, None] * mean_drift
    scale = (2 * dt / (beta * coarse_volume))[:, None, None]
    covariance = scale * mobility
    dimension = factor.n_likelihood_rows * components
    identity = torch.eye(dimension, device=context.device, dtype=context.dtype)
    cholesky = torch.linalg.cholesky(covariance + jitter * identity)
    solved = torch.cholesky_solve(residual[:, :, None], cholesky)[:, :, 0]
    quadratic = (residual * solved).sum(-1)
    logdet = 2 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
    loss = 0.5 * (quadratic + logdet).mean() / dimension
    with torch.no_grad():
        diagonal = torch.diagonal(mobility, dim1=-2, dim2=-1)
        off = mobility - torch.diag_embed(diagonal)
        energy = factor.blocks_on_patch_grid(patches).square().sum((2, 3, 4)).clamp_min(1e-12)
        center = factor.center_factor_index
        central = factor.blocks_on_patch_grid(patches)[:, :, center].square().sum((2, 3))
        metrics = {
            "loss": float(loss),
            "quadratic_per_dimension": float((quadratic / dimension).mean()),
            "mean_mobility": float(diagonal.mean()),
            "mean_absolute_off_diagonal": float(off.abs().mean()),
            "neighbor_energy_fraction": float(((energy - central) / energy).mean()),
            "residual_rms": float(residual.square().mean().sqrt()),
        }
    return loss, metrics
