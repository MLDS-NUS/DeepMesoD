import torch.nn.functional as F

from deepmesod.models import ScalarFreeEnergyInteraction, WeakScalarDrift
from deepmesod.operators import FiniteRangeFactor, divergence_on_field_local_exact

from .base import DynamicalClass
from .registry import register_dynamical_class


@register_dynamical_class("model_a_scalar")
class ScalarModelA(DynamicalClass):
    name = "model_a_scalar"

    def build_drift_model(self, config):
        return WeakScalarDrift(
            patch_size=config.get("patch_size", 15),
            hidden=tuple(config.get("hidden", (128, 128, 128))),
            activation_name=config.get("activation", "softplus"),
        )

    def build_factor(self, config):
        return FiniteRangeFactor(
            patch_size=config.get("patch_size", 15),
            factor_radius=config.get("factor_radius", 1),
            likelihood_radius=config.get("likelihood_radius", 1),
            hidden_channels=config.get("hidden_channels", 32),
            n_layers=config.get("layers", 3),
            activation=config.get("activation", "tanh"),
            dilations=config.get("dilations", (1, 2, 4)),
            spin_flip_even=config.get("spin_flip_even", True),
            parameterization=config.get("parameterization", "amplitude_shape"),
            d4_equivariant=config.get("d4_equivariant", False),
        )

    def build_thermodynamic_model(self, config):
        interaction_constraint = config.get("interaction_constraint")
        if interaction_constraint is None:
            interaction_constraint = (
                "centrosymmetric_sum_one"
                if config.get("centrosymmetric", True)
                else "sum_one"
            )
        return ScalarFreeEnergyInteraction(
            patch_size=config.get("patch_size", 15),
            hidden=tuple(config.get("hidden", (128, 128, 128))),
            activation_name=config.get("activation", "softplus"),
            external_field=config.get("external_field", 0),
            interaction_constraint=interaction_constraint,
        )

    def force(self, thermodynamics, field, *, create_graph):
        state = field[:, 0]
        derivative = thermodynamics.free_energy.derivative(state, create_graph=create_graph)
        padding = thermodynamics.patch_size // 2
        convolved = F.conv2d(
            F.pad(field, (padding, padding, padding, padding), mode="circular"),
            thermodynamics.interaction.weight,
        )[:, 0]
        return derivative - (convolved - state) - thermodynamics.external_field

    def drift(self, factor, thermodynamics, field, beta, coarse_volume):
        force = self.force(thermodynamics, field, create_graph=False)
        divergence, coefficients = divergence_on_field_local_exact(
            factor, field, return_coefficients=True
        )
        return -factor.apply_dissipation(coefficients, force) + divergence / (
            beta * coarse_volume
        )

    def validate_field(self, field):
        super().validate_field(field)
        if field.ndim != 4 or field.shape[1] != 1:
            raise ValueError("model_a_scalar expects (batch, 1, y, x)")
