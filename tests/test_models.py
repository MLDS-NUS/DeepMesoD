import torch

from deepmesod.models import InteractionKernel, UnitSumKernel
from deepmesod.operators import FiniteRangeFactor, divergence_on_field_local_exact


def test_interaction_kernel_has_exact_unit_sum():
    kernel = UnitSumKernel(5, centrosymmetric=True, initialization="normal")
    weight = kernel.weight
    assert torch.allclose(weight.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(weight, torch.flip(weight, dims=(-2, -1)), atol=1e-7)


def test_interaction_constraints_are_configurable():
    raw = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) / 10
    expected = {
        "none": (False, False),
        "sum_one": (False, True),
        "centrosymmetric": (True, False),
        "centrosymmetric_sum_one": (True, True),
    }
    for constraint, (symmetric, unit_sum) in expected.items():
        kernel = InteractionKernel(5, constraint=constraint)
        with torch.no_grad():
            kernel.raw.copy_(raw)
        weight = kernel.weight
        assert (
            torch.allclose(weight, torch.flip(weight, dims=(-2, -1)), atol=1e-7)
            is symmetric
        )
        assert torch.isclose(weight.sum(), torch.tensor(1.0), atol=1e-6).item() is unit_sum


def test_factorized_dissipation_is_positive_semidefinite():
    torch.manual_seed(2)
    factor = FiniteRangeFactor(
        patch_size=7,
        factor_radius=1,
        likelihood_radius=1,
        hidden_channels=4,
        n_layers=3,
        dilations=(1, 1, 1),
        parameterization="amplitude_shape",
    )
    field = torch.randn(2, 1, 8, 8)
    force = torch.randn(2, 8, 8)
    coefficients = factor.coefficients_on_field(field)
    action = factor.apply_dissipation(coefficients, force)
    quadratic_form = (force * action).flatten(1).sum(1)
    assert torch.all(quadratic_form >= -1e-5)


def test_local_divergence_matches_autograd_reference():
    torch.manual_seed(3)
    factor = FiniteRangeFactor(
        patch_size=7,
        factor_radius=1,
        likelihood_radius=1,
        hidden_channels=3,
        n_layers=3,
        dilations=(1, 1, 1),
        parameterization="amplitude_shape",
    )
    field = 0.2 * torch.randn(1, 1, 3, 3)
    local = divergence_on_field_local_exact(factor, field)
    reference = factor.divergence_on_field_exact(field, chunk_size=9)
    assert torch.allclose(local, reference, atol=2e-5, rtol=2e-4)
