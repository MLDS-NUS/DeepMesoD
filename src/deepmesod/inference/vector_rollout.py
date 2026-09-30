"""Deterministic Euler rollout for the vector Model-A pipeline."""

import json
from pathlib import Path

import torch

from deepmesod.data import VectorTrajectory
from deepmesod.models import load_vector_thermodynamics
from deepmesod.operators import load_vector_factor


def rollout_vector(initial, factor_path, thermodynamics_path, output_dir,
                   steps=125, dt=1.0, device="cuda", divergence=True):
    device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    factor, factor_payload = load_vector_factor(factor_path, device)
    model, model_payload = load_vector_thermodynamics(thermodynamics_path, device)
    trajectory = VectorTrajectory.load(initial)
    field = trajectory.field[0:1].to(device)
    beta = float(model_payload.get("beta", factor_payload.get("beta", trajectory.beta)))
    volume = float(model_payload.get("coarse_volume", factor_payload.get("coarse_volume", trajectory.coarse_volume)))
    external_field = float(model_payload.get("external_field", factor_payload.get("external_field", trajectory.external_field)))
    beta_values = torch.full((1,), beta, device=device, dtype=field.dtype)
    h_values = torch.full((1,), external_field, device=device, dtype=field.dtype)
    frames = [field[0].cpu()]
    for _ in range(steps):
        force = model.functional_force(field, beta_values, h_values, create_graph=False)
        with torch.no_grad():
            blocks = factor.blocks_on_field(field)
            update = -factor.apply_mobility(blocks, force)
            if divergence:
                update = update + factor.divergence_on_field_exact(field) / (beta * volume)
        field = (field + dt * update).detach()
        frames.append(field[0].cpu())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result = output / "rollout.pt"
    torch.save({"field": torch.stack(frames), "times": torch.arange(len(frames)) * dt,
                "beta": beta, "coarse_volume": volume, "external_field": external_field}, result)
    (output / "metrics.json").write_text(
        json.dumps({"steps": steps, "final_mean_abs_m": float(field.norm(dim=1).mean())}, indent=2),
        encoding="utf-8",
    )
    return result
