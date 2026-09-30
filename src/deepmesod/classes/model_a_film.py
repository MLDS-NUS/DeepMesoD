"""Model-A adapter for the CrBr3 3D film experiment."""

import torch

from deepmesod.models import FilmPsiJModel, FilmWeakPriorDrift
from deepmesod.operators import VectorFiniteRangeFactor
from deepmesod.operators.film import film_apply_mobility, film_divergence_exact

from .base import DynamicalClass
from .registry import register_dynamical_class


@register_dynamical_class("model_a_film")
class FilmModelA(DynamicalClass):
    name = "model_a_film"

    def build_drift_model(self, config):
        hidden = config.get("mlp_hidden", 64)
        if isinstance(hidden, (list, tuple)):
            if len(set(hidden)) != 1:
                raise ValueError("film weak-prior currently uses one hidden width")
            hidden = hidden[0]
        return FilmWeakPriorDrift(
            num_layers=config.get("num_layers", 28),
            num_components=config.get("n_components", 3),
            patch_size=config.get("patch_size", 7),
            kernel_size=config.get("kernel_size", 7),
            kernel_normalize_mode=config.get("kernel_normalize_mode", "sum"),
            jz_normalize_mode=config.get("jz_normalize_mode", "none"),
            mlp_hidden=int(hidden), mlp_layers=config.get("mlp_layers", 3),
        )

    def build_factor(self, config):
        return VectorFiniteRangeFactor(
            patch_size=config.get("patch_size", 7),
            factor_radius=config.get("factor_radius", 1),
            likelihood_radius=config.get("likelihood_radius", 1),
            n_components=config.get("n_components", 3),
            hidden_channels=config.get("hidden_channels", 32),
            n_layers=config.get("n_layers", 3),
            activation=config.get("activation", "tanh"),
            structure=config.get("structure", "full"),
            min_scale=config.get("min_scale", 1e-5),
            spin_flip_even=config.get("spin_flip_even", True),
            symmetry=config.get("symmetry", "none"),
            axial=config.get("axial", True),
            projector_epsilon=config.get("projector_epsilon", 0.05),
        )

    def build_thermodynamic_model(self, config):
        hidden = config.get("mlp_hidden", [64, 64, 64])
        if isinstance(hidden, int):
            hidden = [hidden] * config.get("mlp_layers", 3)
        return FilmPsiJModel(
            num_layers=config.get("num_layers", 28),
            num_components=config.get("n_components", 3),
            patch_size=config.get("patch_size", 7),
            kernel_normalize_mode=config.get("kernel_normalize_mode", "sum"),
            jz_normalize_mode=config.get("jz_normalize_mode", "none"),
            mlp_hidden=hidden,
            spin_flip_even=config.get("spin_flip_even", False),
        )

    def validate_field(self, field):
        super().validate_field(field)
        if field.ndim != 5 or field.shape[1] != 3:
            raise ValueError("model_a_film expects (batch, 3, Lz, height, width)")

    def force(self, thermodynamics, field, *, create_graph):
        return thermodynamics.functional_force(field, create_graph=create_graph)

    def drift(self, factor, thermodynamics, field, beta, coarse_volume):
        beta_value = torch.as_tensor(beta, device=field.device, dtype=field.dtype).reshape(1)
        force = thermodynamics.functional_force(field, create_graph=False)
        result = -film_apply_mobility(factor, field, force)
        return result + film_divergence_exact(factor, field) / (beta_value * coarse_volume)

