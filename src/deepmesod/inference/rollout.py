import json
import time
from pathlib import Path

import torch

from deepmesod.classes import get_dynamical_class
from deepmesod.data import inverse_temperature, load_trajectory
from deepmesod.operators.checkpoints import load_factor


def load_thermodynamics(path, model_class, config, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = model_class.build_thermodynamic_model(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def rollout(config, factor_path, thermodynamics_path, output_dir):
    requested = config.get("device", "cuda")
    device = torch.device(
        requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
    )
    builder = get_dynamical_class(config["model"]["class"])
    model_class = builder()
    factor, _ = load_factor(factor_path, device)
    thermodynamics, _ = load_thermodynamics(
        thermodynamics_path, model_class, config, device
    )
    references = [load_trajectory(path) for path in config["data"]["test"]]
    times = references[0].times
    count = min(int(config["inference"]["steps"]), times.shape[0] - 1)
    for item in references[1:]:
        if item.field.shape != references[0].field.shape or not torch.allclose(
            item.times, times
        ):
            raise ValueError("test reference trajectories must share shape and times")
        if not torch.allclose(item.field[0], references[0].field[0]):
            raise ValueError("ensemble evaluation requires a shared initial coarse field")
    field = references[0].field[0][None, None].to(device)
    beta = inverse_temperature(references[0].metadata)
    coarse_volume = float(references[0].metadata["coarse_volume"])
    mode = config["inference"].get("mode", "pde")
    snapshots = [field[0, 0].cpu()]
    mae = []
    reference = torch.stack([item.field for item in references]).mean(0).to(device)
    noise_generator = torch.Generator(device=device).manual_seed(
        int(config["inference"].get("noise_seed", 0))
    )

    def step(current, dt):
        drift = model_class.drift(factor, thermodynamics, current, beta, coarse_volume)
        following = current + dt * drift[:, None]
        if mode == "spde":
            coefficients = factor.coefficients_on_field(current)
            noise = torch.randn(
                current.shape,
                generator=noise_generator,
                device=device,
                dtype=current.dtype,
            )[:, 0]
            following = (
                following
                + (2 * dt / (beta * coarse_volume)).sqrt()[:, None, None, None]
                * factor.apply_factor(coefficients, noise)[:, None]
            )
        return following

    compiled_step = step
    if config["inference"].get("compile", True) and hasattr(torch, "compile"):
        compiled_step = torch.compile(step)
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(count):
            dt = (
                (times[index + 1] - times[index])
                .to(device=device, dtype=field.dtype)
                .reshape(1)
            )
            field = compiled_step(field, dt)
            snapshots.append(field[0, 0].cpu())
            mae.append(float((field[0, 0] - reference[index + 1]).abs().mean()))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "field": torch.stack(snapshots),
            "times": times[: count + 1],
            "mae": torch.tensor(mae),
            "runtime_seconds": time.perf_counter() - started,
            "mode": mode,
        },
        output_dir / "rollout.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "mean_mae": sum(mae) / len(mae),
                "final_mae": mae[-1],
                "steps": count,
                "mode": mode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir / "rollout.pt"
