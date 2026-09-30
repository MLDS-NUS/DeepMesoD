"""Direct vector Stage-3 trainer for ``Psi`` and the shared interaction kernel."""

import json
from pathlib import Path

import torch

from deepmesod.classes import register_trainer
from deepmesod.models import VectorPsiJModel
from deepmesod.operators import load_vector_factor

from .common import artifact_path, device_from_config, dynamical_class, save_history
from .vector_targets import prepare_vector_stage3_targets


def _load_targets(paths):
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    return (torch.cat([item["fields"] for item in payloads]),
            torch.cat([item["right_hand_sides"] for item in payloads]))


@register_trainer("vector_free_energy_interaction")
def train_vector_free_energy_interaction(context, stage):
    config, options = context.config, stage.config
    if options.get("objective", "direct") != "direct":
        raise ValueError("vector pipeline currently supports the direct Stage-3 objective")
    device = device_from_config(config)
    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    factor, factor_payload = load_vector_factor(
        artifact_path(context, stage.requires[-1], "checkpoint"), device
    )
    output = context.stage_dir(stage.name)
    target_files = options.get("target_files") or config.get("data", {}).get("stage3_targets")
    if target_files:
        target_paths = [Path(path) for path in target_files]
    else:
        target_paths = prepare_vector_stage3_targets(
            config["data"]["train"], factor, output / "targets",
            options.get("start_time_idx", 75), options.get("end_time_idx", 150),
            factor_payload.get("coarse_volume"), options.get("target_batch_size", 1), device,
        )
    fields, right_hand_sides = _load_targets(target_paths)
    model_config = config["model"]["thermodynamics"]
    model = dynamical_class(config).build_thermodynamic_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, int(options.get("decay_step", 4000)), float(options.get("decay_factor", .5))
    )
    generator = torch.Generator().manual_seed(seed)
    frames_per_step = int(options.get("frames_per_step", 4))
    steps = int(options.get("steps", 20000))
    beta = float(factor_payload.get("beta", .1))
    volume = float(factor_payload.get("coarse_volume", 256.0))
    rhs_scale = right_hand_sides.square().mean().sqrt().clamp_min(1e-12).to(device)
    history = []
    for step in range(1, steps + 1):
        picks = torch.randint(fields.shape[0], (frames_per_step,), generator=generator)
        field = fields[picks].to(device)
        rhs = right_hand_sides[picks].to(device)
        beta_values = torch.full((field.shape[0],), beta, device=device)
        h_values = torch.full_like(beta_values, float(factor_payload.get("external_field", 0.0)))
        force = model.functional_force(field, beta_values, h_values, create_graph=True)
        with torch.no_grad():
            blocks = factor.blocks_on_field(field)
        prediction = factor.apply_mobility(blocks, force)
        residual = prediction - rhs
        loss = residual.square().mean() / rhs_scale.square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % int(options.get("log_interval", 500)) == 0 or step == steps:
            row = {"step": step, "loss": float(loss.detach()),
                   "relative_rmse": float(residual.square().mean().sqrt() / rhs_scale),
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(json.dumps(row), flush=True)
        interval = int(options.get("checkpoint_interval", 5000))
        if step % interval == 0 or step == steps:
            checkpoint_dir = output / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "step": step,
                        "beta": beta, "external_field": factor_payload.get("external_field", 0.0),
                        "coarse_volume": volume, "force_convention": "functional_derivative",
                        "jm_mode": model.jm_mode}, checkpoint_dir / f"step{step:06d}.pth")
    final = output / "thermodynamics_last.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                "model_class": config["model"]["class"], "step": steps,
                "beta": beta, "external_field": factor_payload.get("external_field", 0.0),
                "coarse_volume": volume, "force_convention": "functional_derivative",
                "jm_mode": model.jm_mode}, final)
    save_history(output / "history.json", history)
    return {"checkpoint": str(final), "target_cache": [str(path) for path in target_paths]}
