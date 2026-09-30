import json

import numpy as np
import torch

from deepmesod.data import FilmTrajectory, FilmTrajectoryBank
from deepmesod.models import FilmPsiJModel, FilmWeakPriorDrift


def _write_film(path):
    path.mkdir()
    field = np.random.default_rng(4).normal(size=(5, 4, 6, 3, 3)).astype("float32")
    np.save(path / "m.npy", field)
    (path / "meta.json").write_text(json.dumps({"coarse_volume": 480}), encoding="utf-8")
    return field


def test_film_data_uses_dt_when_times_are_omitted(tmp_path):
    raw = _write_film(tmp_path / "seed1")
    trajectory = FilmTrajectory.load(tmp_path / "seed1", dt=0.002)
    assert tuple(trajectory.field.shape) == (5, 3, 3, 4, 6)
    assert torch.allclose(trajectory.times, torch.arange(5) * 0.002)
    assert torch.allclose(trajectory.field[0], torch.from_numpy(raw[0]).permute(3, 2, 0, 1))
    bank = FilmTrajectoryBank([tmp_path / "seed1"], dt=0.002, seed=5)
    patch, target, dt = bank.sample_drift(3, radius=1)
    assert tuple(patch.shape) == (3, 3, 3, 3, 3)
    assert tuple(target.shape) == (3, 3, 3)
    assert torch.allclose(dt, torch.full((3,), 0.002), atol=1e-6)


def test_film_models_forward_and_functional_force():
    drift = FilmWeakPriorDrift(num_layers=3, patch_size=3, kernel_size=3)
    thermodynamics = FilmPsiJModel(num_layers=3, patch_size=3)
    patch = torch.randn(2, 3, 3, 3, 3)
    assert drift(patch).shape == (2, 3, 3)
    field = torch.randn(2, 3, 3, 4, 4)
    force = thermodynamics.functional_force(field, create_graph=False)
    assert force.shape == field.shape
    assert torch.isfinite(force).all()
