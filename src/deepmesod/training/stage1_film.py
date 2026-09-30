"""Stage-1 training for the CrBr3 3D film weak prior."""

import json
import torch

from deepmesod.classes import register_trainer
from deepmesod.data import FilmTrajectoryBank
from .common import device_from_config, dynamical_class, save_history


@register_trainer("film_weak_prior")
def train_film_weak_prior(context, stage):
    config, options = context.config, stage.config
    device = device_from_config(config)
    torch.manual_seed(int(config["run"]["seed"]))
    model_class = dynamical_class(config)
    model = model_class.build_drift_model(config["model"]["drift"]).to(device)
    bank = FilmTrajectoryBank(
        config["data"]["train"], dt=config["data"].get("dt", 0.002),
        start_time_idx=int(options.get("start_time_idx", 0)),
        end_time_idx=options.get("end_time_idx"), seed=int(config["run"]["seed"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, int(options.get("decay_step", 300)), float(options.get("decay_factor", .8))
    )
    scale = float(options.get("target_scale", 1.0))
    history = []
    for epoch in range(1, int(options.get("epochs", 3000)) + 1):
        losses = []
        for _ in range(int(options.get("steps_per_epoch", 100))):
            patches, target, _ = bank.sample_drift(
                int(options.get("batch_size", 256)), model.patch_size // 2,
                time_start=0, time_stop=None,
            )
            prediction = model(patches.to(device))
            loss = ((prediction - target.to(device)) / scale).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {"epoch": epoch, "loss": sum(losses) / len(losses),
               "learning_rate": scheduler.get_last_lr()[0]}
        history.append(row)
        if epoch == 1 or epoch % int(options.get("log_interval", 20)) == 0:
            print(json.dumps(row), flush=True)
    output = context.stage_dir(stage.name)
    checkpoint = output / "drift_last.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": config["model"]["drift"],
                "model_class": config["model"]["class"], "epoch": history[-1]["epoch"]}, checkpoint)
    save_history(output / "history.json", history)
    return {"checkpoint": str(checkpoint)}
