import torch
import torch.nn.functional as F


def patch_grid(context, patch_size, number_of_rows):
    patches = F.unfold(context, kernel_size=patch_size)
    return patches.transpose(1, 2).reshape(
        context.shape[0],
        number_of_rows,
        1,
        patch_size,
        patch_size,
    )


def window_gaussian_nll(
    factor,
    context,
    current,
    following,
    dt,
    mean_drift,
    beta,
    coarse_volume,
    jitter=1e-8,
    patches=None,
):
    if patches is None:
        patches = patch_grid(
            context,
            factor.patch_size,
            factor.n_likelihood_rows,
        )
    coefficients = factor.coefficients_on_patch_grid(patches)
    dissipation = factor.local_covariance(coefficients)
    residual = following - current - dt[:, None] * mean_drift
    covariance = (2.0 * dt / (beta * coarse_volume))[:, None, None] * dissipation
    identity = torch.eye(
        factor.n_likelihood_rows,
        device=covariance.device,
        dtype=covariance.dtype,
    )
    cholesky = torch.linalg.cholesky(covariance + jitter * identity)
    solved = torch.cholesky_solve(
        residual.unsqueeze(-1),
        cholesky,
    ).squeeze(-1)
    quadratic = (residual * solved).sum(dim=-1)
    log_determinant = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(
        dim=-1
    )
    loss = 0.5 * (log_determinant + quadratic).mean()
    loss = loss / factor.n_likelihood_rows

    with torch.no_grad():
        diagonal = torch.diagonal(dissipation, dim1=-2, dim2=-1)
        off_diagonal = dissipation - torch.diag_embed(diagonal)
        center = factor.center_factor_index
        energy = coefficients.square().sum(dim=-1).clamp_min(1e-12)
        neighbor_fraction = (energy - coefficients[:, :, center].square()) / energy
        metrics = {
            "loss": float(loss.detach()),
            "quadratic_per_dimension": float((quadratic / factor.n_likelihood_rows).mean()),
            "mean_dissipation": float(diagonal.mean()),
            "mean_absolute_off_diagonal": float(off_diagonal.abs().mean()),
            "neighbor_energy_fraction": float(neighbor_fraction.mean()),
            "residual_rms": float(residual.square().mean().sqrt()),
        }
    return loss, metrics
