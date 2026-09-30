"""Reusable extraction helpers for factorized interaction kernels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def state_dict(payload: dict) -> dict:
    return payload.get("model_state_dict", payload)


def infer_factorized_layers(state: dict) -> int:
    flat = state.get("kernel.J_z_flat")
    if flat is None:
        raise ValueError("factorized checkpoint is missing kernel.J_z_flat")
    n_tri = int(flat.numel())
    layers = int(round((math.sqrt(8 * n_tri + 1) - 1) / 2))
    if layers * (layers + 1) // 2 != n_tri:
        raise ValueError(f"invalid triangular kernel parameter count: {n_tri}")
    return layers


def extract_factorized_kernel(checkpoint: str | Path, normalize_spatial: str = "sum") -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(layer_kernel, spatial_kernel, metadata)`` from a checkpoint."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state_dict(payload)
    layers = infer_factorized_layers(state)
    rows, cols = torch.triu_indices(layers, layers)
    layer_kernel = torch.zeros(layers, layers)
    layer_kernel[rows, cols] = state["kernel.J_z_flat"].detach().cpu()
    layer_kernel = layer_kernel + layer_kernel.triu(1).T
    spatial_kernel = state["kernel.w_xy"].detach().cpu()
    if normalize_spatial == "sum":
        spatial_kernel = spatial_kernel / (spatial_kernel.sum() + 1e-8)
    elif normalize_spatial != "none":
        raise ValueError("normalize_spatial must be 'sum' or 'none'")
    metadata = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "epoch": payload.get("epoch", payload.get("step")),
        "num_layers": layers,
        "kernel_size": int(spatial_kernel.shape[0]),
        "spatial_normalize": normalize_spatial,
    }
    return layer_kernel.numpy(), spatial_kernel.numpy(), metadata


def export_factorized_kernel(checkpoint: str | Path, output_dir: str | Path,
                             normalize_spatial: str = "sum",
                             layer_name: str = "layer_kernel.csv",
                             spatial_name: str = "spatial_kernel.csv") -> dict:
    """Write extracted factorized kernels for a plotting or audit step."""
    layer_kernel, spatial_kernel, metadata = extract_factorized_kernel(checkpoint, normalize_spatial)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / layer_name, layer_kernel, delimiter=",", fmt="%.6f")
    np.savetxt(output / spatial_name, spatial_kernel, delimiter=",", fmt="%.6f")
    (output / "kernel_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_dir": output, "layer_kernel": output / layer_name, "spatial_kernel": output / spatial_name}


__all__ = ["export_factorized_kernel", "extract_factorized_kernel", "infer_factorized_layers", "state_dict"]
