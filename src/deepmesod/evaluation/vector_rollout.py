"""Reusable deterministic vector-field rollout evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from deepmesod.models import load_vector_thermodynamics
from deepmesod.operators import load_vector_factor


def _load_frame(path):
    values = np.asarray(np.load(path), dtype=np.float32)
    if values.ndim == 4:
        singleton = [axis for axis in range(1, values.ndim) if values.shape[axis] == 1]
        if len(singleton) != 1:
            raise ValueError("initial frame must contain at most one singleton axis")
        values = np.squeeze(values, axis=singleton[0])
    if values.ndim != 3:
        raise ValueError("initial frame must have shape (height, width, components)")
    if values.shape[-1] in (2, 3):
        frame = values
    elif values.shape[0] in (2, 3):
        frame = np.transpose(values, (1, 2, 0))
    else:
        raise ValueError("initial frame must have components on an outer axis")
    if not np.isfinite(frame).all():
        raise ValueError("initial frame contains non-finite values")
    return np.ascontiguousarray(frame)


def _force_field(model, field, beta, external_field):
    beta_values = torch.full((field.shape[0],), beta, device=field.device, dtype=field.dtype)
    field_values = torch.full_like(beta_values, external_field)
    return model.functional_force(field, beta_values, field_values, create_graph=False).detach()


def generate_vector_rollout(
    stage3_checkpoint, factor_checkpoint, initial_frame, output_dir, *,
    t_start=0.0, t_end=125.0, dt=1.0, divk_every=1, jacobian_chunk_size=1024,
    device="cuda", progress_every=0, rollout_name="rollout", metadata=None,
):
    """Generate trajectory and macro-statistics arrays for a vector Model-A model."""
    if dt <= 0 or t_end < t_start or divk_every < 0:
        raise ValueError("invalid rollout interval, dt, or divergence interval")
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    factor, factor_payload = load_vector_factor(factor_checkpoint, target_device)
    model, model_payload = load_vector_thermodynamics(stage3_checkpoint, target_device)
    if model_payload.get("jm_mode", "raw") != "raw":
        raise ValueError("vector rollout evaluation requires the direct raw Stage-3 model")
    frame = _load_frame(initial_frame)
    height, width, components = frame.shape
    if components != 3:
        raise ValueError("vector rollout evaluation expects three components")
    times = np.arange(t_start, t_end + 0.5 * dt, dt, dtype=np.float64)
    field = torch.from_numpy(frame).permute(2, 0, 1)[None].to(target_device)
    beta = float(model_payload.get("beta", factor_payload.get("beta", 0.1)))
    volume = float(model_payload.get("coarse_volume", factor_payload.get("coarse_volume", 256.0)))
    external_field = float(model_payload.get("external_field", factor_payload.get("external_field", 0.0)))
    if beta <= 0 or volume <= 0:
        raise ValueError("checkpoint beta and coarse_volume must be positive")

    output = Path(output_dir).expanduser().resolve() / "data" / rollout_name
    output.mkdir(parents=True, exist_ok=True)
    stem = f"t{int(t_start) if float(t_start).is_integer() else str(t_start).replace('.', 'p')}_{int(t_end) if float(t_end).is_integer() else str(t_end).replace('.', 'p')}"
    initial_name, trajectory_name, macro_name = f"initial_frame_{stem}.npy", f"pred_traj_{stem}.npy", f"macro_statistics_{stem}.csv"
    np.save(output / initial_name, frame)
    trajectory, rows = [], []
    site_count = height * width

    def record(index, value):
        current = value[0]
        trajectory.append(current.permute(1, 2, 0).cpu().numpy().astype(np.float32))
        means = current.mean(dim=(1, 2))
        variance = current.reshape(components, -1).var(dim=1, unbiased=False)
        susceptibility = beta * site_count * variance
        rows.append({"time": float(times[index]), "mx": float(means[0]), "my": float(means[1]),
                     "mz": float(means[2]), "m_norm": float(current.norm(dim=0).mean()),
                     "chi_x": float(susceptibility[0]), "chi_y": float(susceptibility[1]),
                     "chi_z": float(susceptibility[2]),
                     "chi_norm": float(beta * site_count * current.norm(dim=0).reshape(-1).var(unbiased=False))})

    record(0, field)
    for step in range(1, len(times)):
        force = _force_field(model, field, beta, external_field)
        with torch.no_grad():
            blocks = factor.blocks_on_field(field)
            update = -factor.apply_mobility(blocks, force)
            if divk_every and step % divk_every == 0:
                divergence = factor.divergence_on_field_exact(field, chunk_size=jacobian_chunk_size).detach()
                update = update + divergence / (beta * volume)
        field = (field + dt * update).detach()
        record(step, field)
        magnitude = float(field.norm(dim=1).mean())
        if not np.isfinite(magnitude) or magnitude > 10.0:
            raise FloatingPointError(f"vector rollout diverged at step {step}")
        if progress_every and step % progress_every == 0:
            print(f"[rollout] step {step}/{len(times) - 1} |m|={magnitude:.4f}")
    np.save(output / trajectory_name, np.stack(trajectory))
    with (output / macro_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {"stage3_checkpoint": str(Path(stage3_checkpoint).expanduser().resolve()),
                "factor_checkpoint": str(Path(factor_checkpoint).expanduser().resolve()),
                "initial_frame": str(Path(initial_frame).expanduser().resolve()),
                "output_dir": str(output), "t_start": float(t_start), "t_end": float(t_end),
                "dt": float(dt), "divk_every": int(divk_every), "beta": beta,
                "coarse_volume": volume, "frames": len(rows), "files": {
                    "initial_frame": initial_name, "trajectory": trajectory_name, "macro_statistics": macro_name}}
    if metadata:
        manifest.update(metadata)
    (output / "evaluation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"output_dir": output, "initial_frame": output / initial_name,
            "trajectory": output / trajectory_name, "macro_statistics": output / macro_name,
            "metadata": output / "evaluation.json"}


__all__ = ["generate_vector_rollout"]
