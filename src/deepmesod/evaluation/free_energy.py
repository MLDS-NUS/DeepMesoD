"""Reusable two-block energy-surface evaluation for spatially local models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


def uniform_layer_energy(model, profiles: torch.Tensor) -> torch.Tensor:
    """Evaluate energy per layer for uniform profiles using ``model.density``."""
    if profiles.ndim == 1:
        profiles = profiles[None]
    batch, layers = profiles.shape
    size = int(model.patch_size)
    mz = profiles[:, :, None, None].expand(batch, layers, size, size)
    zeros = torch.zeros_like(mz)
    density = model.density(torch.stack((zeros, zeros, mz), dim=1))
    center = size // 2
    return density[:, :, center, center].sum(dim=1) / layers


def _minimum_path(energy, contrast_axis):
    contrast_min = np.full(energy.shape[1], np.nan)
    energy_min = np.full(energy.shape[1], np.nan)
    for column in range(energy.shape[1]):
        if np.isfinite(energy[:, column]).any():
            row = np.nanargmin(energy[:, column])
            contrast_min[column] = contrast_axis[row]
            energy_min[column] = energy[row, column]
    return contrast_min, energy_min


def scan_two_block_energy(
    model, output_dir: str | Path, *, split_after_layer: int,
    num_mean: int = 401, num_contrast: int = 401, batch_size: int = 2048,
    device: str = "cpu", dtype: str = "float64", checkpoint: str | Path | None = None,
    artifact_stem: str = "energy_surface",
) -> dict:
    """Evaluate a two-block profile surface for any compatible film model."""
    actual_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    actual_dtype = torch.float64 if dtype == "float64" else torch.float32
    model = model.to(device=actual_device, dtype=actual_dtype).eval()
    layers = int(model.num_layers)
    if not 1 <= split_after_layer < layers:
        raise ValueError(f"split_after_layer must be between 1 and {layers - 1}")
    mean_axis = np.linspace(-1, 1, num_mean)
    contrast_axis = np.linspace(-1, 1, num_contrast)
    mean_grid, contrast_grid = np.meshgrid(mean_axis, contrast_axis, indexing="xy")
    first, second = mean_grid + contrast_grid, mean_grid - contrast_grid
    feasible = np.abs(mean_grid) + np.abs(contrast_grid) <= 1.0 + 1e-12
    selected = np.flatnonzero(feasible.ravel())
    values = np.full(mean_grid.size, np.nan, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            indexes = selected[start : start + batch_size]
            profiles = torch.empty(len(indexes), layers, device=actual_device, dtype=actual_dtype)
            profiles[:, :split_after_layer] = torch.as_tensor(first.ravel()[indexes], device=actual_device, dtype=actual_dtype)[:, None]
            profiles[:, split_after_layer:] = torch.as_tensor(second.ravel()[indexes], device=actual_device, dtype=actual_dtype)[:, None]
            values[indexes] = uniform_layer_energy(model, profiles).cpu().numpy()
    energy = values.reshape(mean_grid.shape)
    reference = float(np.nanmin(energy))
    centered = energy - reference
    contrast_min, energy_min = _minimum_path(energy, contrast_axis)
    minimum = np.unravel_index(np.nanargmin(energy), energy.shape)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / f"{artifact_stem}.npz"
    csv_path = output / f"{artifact_stem}.csv"
    np.savez_compressed(
        npz_path, mean_axis=mean_axis,
        contrast_axis=contrast_axis, mean_grid=mean_grid, contrast_grid=contrast_grid,
        first_grid=first, second_grid=second, feasible_mask=feasible,
        energy_raw=energy, energy_minus_global_min=centered,
        contrast_min_at_fixed_mean=contrast_min, energy_min_at_fixed_mean=energy_min,
        global_min_mean=float(mean_axis[minimum[1]]), global_min_contrast=float(contrast_axis[minimum[0]]),
        global_min_energy=reference, split_after_layer=split_after_layer,
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mean", "contrast", "first", "second", "energy_raw", "energy_minus_global_min"))
        for index in selected:
            row = np.unravel_index(index, energy.shape)
            writer.writerow((mean_grid[row], contrast_grid[row], first[row], second[row], energy[row], centered[row]))
    with (output / "minimum_path.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mean", "contrast_min", "first", "second", "energy_raw", "energy_minus_global_min"))
        for mean, contrast, value in zip(mean_axis, contrast_min, energy_min):
            writer.writerow((mean, contrast, mean + contrast, mean - contrast, value, value - reference))
    metadata = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint else None,
        "num_mean": num_mean, "num_contrast": num_contrast,
        "split_after_layer": split_after_layer, "n_feasible_points": int(feasible.sum()),
        "global_min_energy": reference,
    }
    (output / "free_energy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_dir": output, "npz": npz_path, "csv": csv_path}


__all__ = ["scan_two_block_energy", "uniform_layer_energy"]
