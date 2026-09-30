"""Model-A adapter for the three-component 2D ferromagnet pipeline."""

import torch

from deepmesod.models import VectorPsiJModel, VectorWeakPriorDrift
from deepmesod.operators import VectorFiniteRangeFactor

from .base import DynamicalClass
from .registry import register_dynamical_class


@register_dynamical_class("model_a_vector")
class VectorModelA(DynamicalClass):
    name = "model_a_vector"

    def build_drift_model(self, config):
        hidden = config.get("mlp_hidden", config.get("hidden", 64))
        if isinstance(hidden, (list, tuple)):
            if len(set(hidden)) != 1:
                raise ValueError("VectorWeakPriorDrift currently uses one hidden width")
            hidden = hidden[0]
        return VectorWeakPriorDrift(
            patch_size=config.get("patch_size", 13),
            kernel_size=config.get("kernel_size", 13),
            kernel_normalize_mode=config.get("kernel_normalize_mode", "sum"),
            mlp_hidden=int(hidden),
            mlp_layers=config.get("mlp_layers", config.get("layers", 3)),
            n_components=config.get("n_components", 3),
        )

    def build_factor(self, config):
        return VectorFiniteRangeFactor(
            patch_size=config.get("patch_size", 15),
            factor_radius=config.get("factor_radius", 1),
            likelihood_radius=config.get("likelihood_radius", 1),
            n_components=config.get("n_components", 3),
            hidden_channels=config.get("hidden_channels", 32),
            n_layers=config.get("n_layers", config.get("layers", 3)),
            activation=config.get("activation", "tanh"),
            structure=config.get("structure", config.get("factor_structure", "full")),
            min_scale=config.get("min_scale", 1e-5),
            spin_flip_even=config.get("spin_flip_even", not config.get("no_spin_flip_even", False)),
            symmetry=config.get("symmetry", "none"),
            axial=config.get("axial", True),
        )

    def build_thermodynamic_model(self, config):
        hidden = config.get("mlp_hidden", config.get("hidden", [64, 64, 64]))
        if isinstance(hidden, int):
            hidden = [hidden] * config.get("mlp_layers", 3)
        return VectorPsiJModel(
            patch_size=config.get("patch_size", 13),
            kernel_size=config.get("kernel_size", 13),
            kernel_normalize_mode=config.get("kernel_normalize_mode", "sum"),
            mlp_hidden=tuple(hidden),
            n_components=config.get("n_components", 3),
            jm_mode=config.get("jm_mode", "raw"),
        )

    def force(self, thermodynamics, field, *, create_graph):
        beta = torch.full((field.shape[0],), 0.1, device=field.device, dtype=field.dtype)
        external_field = torch.zeros_like(beta)
        return thermodynamics.functional_force(
            field, beta, external_field, create_graph=create_graph
        )

    def drift(self, factor, thermodynamics, field, beta, coarse_volume):
        beta_value = torch.as_tensor(beta, device=field.device, dtype=field.dtype)
        beta_value = beta_value.reshape(1).expand(field.shape[0])
        h_value = torch.as_tensor(0.0, device=field.device, dtype=field.dtype)
        h_value = h_value.reshape(1).expand(field.shape[0])
        force = thermodynamics.functional_force(
            field, beta_value, h_value, create_graph=False
        )
        blocks = factor.blocks_on_field(field)
        divergence = factor.divergence_on_field_exact(field)
        return -factor.apply_mobility(blocks, force) + divergence / (beta_value[:, None, None, None] * coarse_volume)

    def validate_field(self, field):
        super().validate_field(field)
        if field.ndim != 4 or field.shape[1] != 3:
            raise ValueError("model_a_vector expects (batch, 3, y, x)")
