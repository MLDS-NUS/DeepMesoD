import argparse
import json
from pathlib import Path

import numpy as np


def write_trajectory(path, seed, states=9, size=8, dt=0.05, initial_seed=None):
    path.mkdir(parents=True, exist_ok=False)
    initial_rng = np.random.default_rng(seed if initial_seed is None else initial_seed)
    noise_rng = np.random.default_rng(seed + 100000)
    field = np.empty((states, size, size), dtype=np.float32)
    field[0] = initial_rng.uniform(-0.7, 0.7, size=(size, size))
    for index in range(states - 1):
        neighborhood = 0.25 * (
            np.roll(field[index], 1, 0)
            + np.roll(field[index], -1, 0)
            + np.roll(field[index], 1, 1)
            + np.roll(field[index], -1, 1)
        )
        drift = -field[index] + 0.6 * neighborhood - 0.1 * field[index] ** 3
        noise = 0.01 * noise_rng.normal(size=(size, size))
        field[index + 1] = np.clip(field[index] + dt * drift + noise, -1, 1)
    np.save(path / "field.npy", field)
    np.save(path / "times.npy", np.arange(states, dtype=np.float64) * dt)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system": "synthetic_scalar",
                "boundary": "periodic",
                "beta": 1.0,
                "coarse_volume": 1.0,
                "external_field": 0.0,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "synthetic",
    )
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=True)
    for split, seeds in {"train": (1, 2, 3), "test": (4, 5, 6)}.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            initial_seed = 4 if split == "test" else None
            write_trajectory(directory / f"seed{seed}", seed, initial_seed=initial_seed)
