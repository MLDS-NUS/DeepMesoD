"""Stage-3 direct force matching for the CrBr3 film Psi/J model."""

import json
import os
from pathlib import Path
import torch

from deepmesod.classes import register_trainer
from deepmesod.data import FilmTrajectoryBank
from deepmesod.models import FilmPsiJModel
from deepmesod.operators import film_apply_mobility, film_divergence_exact
from .common import artifact_path, device_from_config, dynamical_class, save_history


def prepare_film_targets(bank, factor, beta, volume, start, stop, stride, device, chunk_size):
    fields, targets = [], []
    for trajectory in bank.trajectories:
        end = min(int(stop), trajectory.T - 1) if stop is not None else trajectory.T - 1
        for t in range(int(start), end, int(stride)):
            current = trajectory.field[t:t + 1].to(device)
            following = trajectory.field[t + 1:t + 2].to(device)
            dt = trajectory.times[t + 1] - trajectory.times[t]
            divergence = film_divergence_exact(factor, current, chunk_size)
            target = divergence / (float(beta) * float(volume)) - (following - current) / dt
            fields.append(current.cpu()[0])
            targets.append(target.cpu()[0])
    if not fields:
        raise ValueError("Stage-3 target window contains no transitions")
    return torch.stack(fields), torch.stack(targets)


@register_trainer("film_free_energy_interaction")
def train_film_free_energy_interaction(context, stage):
    config, options = context.config, stage.config
    device = device_from_config(config)
    model_class = dynamical_class(config)
    factor_payload = torch.load(artifact_path(context, stage.requires[-1], "checkpoint"),
                                map_location=device, weights_only=False)
    factor = model_class.build_factor(factor_payload["model_config"]).to(device)
    factor.load_state_dict(factor_payload["model_state_dict"])
    factor.eval()
    for parameter in factor.parameters():
        parameter.requires_grad_(False)
    bank = FilmTrajectoryBank(config["data"]["train"], config["data"].get("dt", .002),
                              seed=int(config["run"]["seed"]))
    beta = float(config["physics"]["beta"])
    volume = float(config["physics"]["coarse_volume"])
    target_path = options.get("target_cache")
    if target_path:
        target_path = Path(os.path.expandvars(str(target_path)))
    if target_path and target_path.exists():
        cached = torch.load(target_path, map_location="cpu", weights_only=False)
        fields, targets = cached["fields"], cached["targets"]
    else:
        fields, targets = prepare_film_targets(
            bank, factor, beta, volume, options.get("start_time_idx", 50),
            options.get("end_time_idx", 251), options.get("frame_stride", 3),
            device, int(options.get("divergence_chunk", 512)),
        )
        if target_path:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"fields": fields, "targets": targets}, target_path)
    model = model_class.build_thermodynamic_model(config["model"]["thermodynamics"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(options.get("decay_step", 400)),
                                                float(options.get("decay_factor", .5)))
    scale = float(options.get("target_scale", targets.square().mean().sqrt()))
    generator = torch.Generator().manual_seed(int(config["run"]["seed"]))
    history = []
    for step in range(1, int(options.get("steps", 2000)) + 1):
        idx = torch.randint(fields.shape[0], (int(options.get("frames_per_step", 4)),), generator=generator)
        field, target = fields[idx].to(device), targets[idx].to(device)
        force = model.functional_force(field, h=float(config["physics"].get("external_field", 0)), create_graph=True)
        prediction = film_apply_mobility(factor, field, force)
        residual = prediction - target
        loss = residual.square().mean() / (scale * scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(options.get("grad_clip", 1.0)))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % int(options.get("log_interval", 100)) == 0 or step == int(options.get("steps", 2000)):
            row = {"step": step, "loss": float(loss.detach()),
                   "relative_rmse": float(residual.square().mean().sqrt() / scale),
                   "learning_rate": scheduler.get_last_lr()[0]}
            history.append(row)
            print(json.dumps(row), flush=True)
    output = context.stage_dir(stage.name)
    checkpoint = output / "thermodynamics_last.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": config["model"]["thermodynamics"],
                "model_class": config["model"]["class"], "step": step,
                "beta": beta, "coarse_volume": volume,
                "force_convention": "functional_derivative"}, checkpoint)
    save_history(output / "history.json", history)
    return {"checkpoint": str(checkpoint), "target_cache": str(target_path) if target_path else None}
