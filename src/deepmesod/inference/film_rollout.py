"""Euler rollout for a CrBr3 finite-thickness film."""

import json
from pathlib import Path
import torch

from deepmesod.data import FilmTrajectory
from deepmesod.models import FilmPsiJModel
from deepmesod.operators import VectorFiniteRangeFactor, film_apply_mobility, film_divergence_exact


def rollout_film(initial, factor_path, thermodynamics_path, output_dir, steps=125,
                 dt=0.002, beta=0.02, coarse_volume=480.0, h=0.0,
                 device="cuda", divergence=True):
    device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    factor_payload = torch.load(factor_path, map_location=device, weights_only=False)
    factor = VectorFiniteRangeFactor(**factor_payload["model_config"]).to(device)
    factor.load_state_dict(factor_payload["model_state_dict"])
    factor.eval()
    model_payload = torch.load(thermodynamics_path, map_location=device, weights_only=False)
    model_config = model_payload.get("model_config") or {
        "num_layers": 28, "num_components": 3, "patch_size": 7,
        "kernel_normalize_mode": "sum", "jz_normalize_mode": "none",
        "mlp_hidden": [64, 64, 64],
    }
    model = FilmPsiJModel(
        num_layers=model_config.get("num_layers", 28),
        num_components=model_config.get("num_components", model_config.get("n_components", 3)),
        patch_size=model_config.get("patch_size", 7),
        kernel_normalize_mode=model_config.get("kernel_normalize_mode", "sum"),
        jz_normalize_mode=model_config.get("jz_normalize_mode", "none"),
        mlp_hidden=model_config.get("mlp_hidden", [64, 64, 64]),
        spin_flip_even=model_config.get("spin_flip_even", False),
    ).to(device)
    model.load_state_dict(model_payload["model_state_dict"])
    model.eval()
    trajectory = FilmTrajectory.load(initial, dt=dt, end_time_idx=2, device=device)
    field = trajectory.field[0:1]
    frames = [field[0].cpu()]
    for _ in range(int(steps)):
        with torch.enable_grad():
            force = model.functional_force(field, h=h, create_graph=False)
        with torch.no_grad():
            update = -film_apply_mobility(factor, field, force)
            if divergence:
                update = update + film_divergence_exact(factor, field) / (float(beta) * float(coarse_volume))
            field = (field + float(dt) * update).detach()
        frames.append(field[0].cpu())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result = output / "rollout.pt"
    torch.save({"field": torch.stack(frames), "times": torch.arange(len(frames)) * float(dt),
                "beta": beta, "coarse_volume": coarse_volume, "external_field": h}, result)
    (output / "metrics.json").write_text(json.dumps({"steps": int(steps)}, indent=2), encoding="utf-8")
    return result

