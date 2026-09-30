import json

import numpy as np
import pytest
import torch

from deepmesod.data import (
    UniformFieldSampler,
    convert_legacy_ising_trajectory,
    load_trajectory,
)


def write_trajectory(path, field, times):
    path.mkdir()
    np.save(path / "field.npy", field)
    np.save(path / "times.npy", times)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "boundary": "periodic",
                "beta": 2.0,
                "coarse_volume": 4.0,
            }
        ),
        encoding="utf-8",
    )


def test_forward_increments_use_current_and_next_state(tmp_path):
    field = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    times = np.array([0.0, 0.5, 1.5])
    write_trajectory(tmp_path / "trajectory", field, times)
    trajectory = load_trajectory(tmp_path / "trajectory")
    current, following, dt, rate = trajectory.transitions
    assert torch.equal(current, trajectory.field[:-1])
    assert torch.equal(following, trajectory.field[1:])
    assert torch.allclose(dt, torch.tensor([0.5, 1.0], dtype=torch.float64))
    assert torch.allclose(rate[0], torch.full((2, 2), 8.0))
    assert torch.allclose(rate[1], torch.full((2, 2), 4.0))


def test_invalid_time_axis_is_rejected(tmp_path):
    field = np.zeros((3, 2, 2), dtype=np.float32)
    write_trajectory(tmp_path / "trajectory", field, np.array([0.0, 0.5, 0.5]))
    with pytest.raises(ValueError, match="strictly increasing"):
        load_trajectory(tmp_path / "trajectory")


def test_uniform_sampler_returns_valid_coordinates():
    field = torch.zeros(3, 4, 5)
    sampler = UniformFieldSampler(field, generator=torch.Generator().manual_seed(7))
    time, row, column = sampler.sample(100)
    assert time.min() >= 0 and time.max() < 3
    assert row.min() >= 0 and row.max() < 4
    assert column.min() >= 0 and column.max() < 5


def test_legacy_ising_conversion_normalizes_metadata_without_copying(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    np.save(source / "m.npy", np.zeros((3, 2, 2), dtype=np.float32))
    np.save(source / "times.npy", np.array([0.0, 0.5, 1.0]))
    (source / "meta.json").write_text(
        json.dumps(
            {
                "B": 32,
                "shape": [3, 2, 2],
                "dataset_meta": {"T": 0.75, "h": 0.0},
            }
        ),
        encoding="utf-8",
    )
    output = convert_legacy_ising_trajectory(source, tmp_path / "converted")
    assert (output / "field.npy").is_symlink()
    trajectory = load_trajectory(output)
    assert trajectory.metadata["coarse_volume"] == 1024
    assert trajectory.metadata["temperature"] == 0.75
