import json

import numpy as np

from deepmesod.data import load_trajectory
from deepmesod.data.coarse_graining import coarse_grain_vampire, periodic_block_mean


def test_vampire_float_snapshots_are_coarse_grained_to_contract(tmp_path):
    raw_root = tmp_path / "raw"
    raw = raw_root / "round0"
    raw.mkdir(parents=True)
    snapshots = np.arange(3 * 8 * 8 * 1 * 3, dtype=np.float32).reshape(3, 8, 8, 1, 3)
    np.save(raw / "seed1_snaps.npy", snapshots)
    np.save(raw / "seed1_times.npy", np.array([0.0, 1.0, 2.0]))
    (raw / "seed1_meta.json").write_text(
        json.dumps(
            {
                "snaps_shape": list(snapshots.shape),
                "components": ["x", "y", "z"],
                "beta": 0.1,
                "T": 10.0,
            }
        ),
        encoding="utf-8",
    )

    output = coarse_grain_vampire(
        raw_root,
        tmp_path / "coarse",
        block_size=2,
        coarse_size=4,
        batch_size=2,
    )
    trajectory = load_trajectory(output / "round0")
    expected = periodic_block_mean(snapshots, 2, 4)
    expected = expected[:, :, :, 0, :].transpose(0, 3, 1, 2)

    assert trajectory.field.shape == (3, 3, 4, 4)
    assert np.allclose(trajectory.field.numpy(), expected)
    assert trajectory.metadata["coarse_volume"] == 4.0
    assert trajectory.metadata["simulator"] == "Vampire"
