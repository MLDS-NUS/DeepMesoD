import json

import numpy as np
import torch

from deepmesod.classes import get_dynamical_class
from deepmesod.data import VectorTrajectoryBank, load_trajectory
from deepmesod.models import VectorPsiJModel
from deepmesod.operators import VectorFiniteRangeFactor


def write_vector_trajectory(path):
    path.mkdir()
    field = np.random.default_rng(4).normal(size=(5, 6, 6, 1, 3)).astype(np.float32)
    np.save(path / "m.npy", field)
    np.save(path / "times.npy", np.arange(5, dtype=np.float64))
    (path / "meta.json").write_text(
        json.dumps({"B": 2, "dataset_meta": {"beta": 0.1, "h": 0.0}}),
        encoding="utf-8",
    )


def test_legacy_vector_layout_is_normalized_and_sampled(tmp_path):
    path = tmp_path / "trajectory"
    write_vector_trajectory(path)
    trajectory = load_trajectory(path)
    assert trajectory.field.shape == (5, 3, 6, 6)
    assert trajectory.metadata["coarse_volume"] == 4.0
    bank = VectorTrajectoryBank([path], bins=2, seed=3)
    factor, drift, current, following, dt = bank.sample_window(4, 2, 1, 1)
    assert factor.shape == (4, 3, 5, 5)
    assert drift.shape == (4, 3, 3, 3)
    assert current.shape == following.shape == (4, 27)
    assert dt.shape == (4,)


def test_vector_model_class_builds_pipeline_components():
    model_class = get_dynamical_class("model_a_vector")()
    factor = model_class.build_factor({"patch_size": 7, "hidden_channels": 3})
    drift = model_class.build_drift_model({"patch_size": 7, "kernel_size": 7,
                                           "mlp_hidden": 4, "n_components": 3})
    thermodynamics = model_class.build_thermodynamic_model(
        {"patch_size": 7, "kernel_size": 7, "mlp_hidden": [4], "jm_mode": "deviation"}
    )
    field = torch.randn(2, 3, 8, 8)
    blocks = factor.blocks_on_field(field)
    assert blocks.shape == (2, 9, 3, 3, 8, 8)
    assert drift(torch.randn(5, 3, 7, 7), torch.ones(5), torch.zeros(5)).shape == (5, 3)
    force = thermodynamics.functional_force(field, torch.ones(2), torch.zeros(2))
    assert force.shape == field.shape


def test_vector_factor_is_positive_semidefinite_and_has_exact_shapes():
    torch.manual_seed(5)
    factor = VectorFiniteRangeFactor(patch_size=7, hidden_channels=3)
    field = torch.randn(2, 3, 8, 8)
    force = torch.randn_like(field)
    blocks = factor.blocks_on_field(field)
    quadratic = (force * factor.apply_mobility(blocks, force)).flatten(1).sum(1)
    assert torch.all(quadratic >= -1e-5)
    divergence = factor.divergence_on_field_exact(field, chunk_size=64)
    assert divergence.shape == field.shape


def test_vector_psi_functional_force_backpropagates_to_parameters():
    model = VectorPsiJModel(patch_size=7, kernel_size=7, mlp_hidden=(4,), jm_mode="raw")
    field = torch.randn(2, 3, 8, 8)
    force = model.functional_force(field, torch.ones(2), torch.zeros(2), create_graph=True)
    force.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
