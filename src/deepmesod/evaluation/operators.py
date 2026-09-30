import torch


@torch.no_grad()
def conditional_complete_response(factor, fields, bins=50, batch_size=8, device="cuda"):
    device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    sums = torch.zeros(bins, dtype=torch.float64)
    squares = torch.zeros(bins, dtype=torch.float64)
    counts = torch.zeros(bins, dtype=torch.float64)
    for start in range(0, fields.shape[0], batch_size):
        field = fields[start : start + batch_size, None].to(device)
        coefficients = factor.coefficients_on_field(field)
        response = factor.apply_dissipation(coefficients, torch.ones_like(field[:, 0]))
        code = ((field[:, 0] + 1) * 0.5 * bins).floor().clamp(0, bins - 1).long()
        flat_code = code.reshape(-1).cpu()
        flat_response = response.reshape(-1).double().cpu()
        sums.scatter_add_(0, flat_code, flat_response)
        squares.scatter_add_(0, flat_code, flat_response.square())
        counts.scatter_add_(0, flat_code, torch.ones_like(flat_response))
    mean = sums / counts.clamp_min(1)
    variance = squares / counts.clamp_min(1) - mean.square()
    centers = torch.linspace(-1 + 1 / bins, 1 - 1 / bins, bins)
    return {
        "state": centers,
        "mean": mean,
        "standard_deviation": variance.clamp_min(0).sqrt(),
        "count": counts,
    }


@torch.no_grad()
def interaction_diagnostics(thermodynamics):
    kernel = thermodynamics.interaction.weight[0, 0].double()
    transpose = torch.flip(kernel, dims=(-2, -1))
    constraint = thermodynamics.interaction.constraint
    return {
        "constraint": constraint,
        "kernel_sum": float(kernel.sum()),
        "unit_sum_declared": constraint in {"sum_one", "centrosymmetric_sum_one"},
        "centrosymmetry_declared": constraint
        in {"centrosymmetric", "centrosymmetric_sum_one"},
        "sum_residual": float((kernel.sum() - 1).abs()),
        "centrosymmetry_relative_error": float(
            (kernel - transpose).norm() / kernel.norm().clamp_min(1e-20)
        ),
        "negative_mass": float((-kernel.clamp_max(0)).sum()),
    }
