"""Vector Stage-1 trainer migrated from the pipeline_r1 script."""

import json

import torch

from deepmesod.classes import register_trainer
from deepmesod.data import VectorTrajectoryBank

from .common import device_from_config, dynamical_class, save_history


@register_trainer("vector_local_drift")
def train_vector_local_drift(context, stage):
    options = stage.config
    config = context.config
    device = device_from_config(config)
    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    model_class = dynamical_class(config)
    model = model_class.build_drift_model(config["model"]["drift"]).to(device)
    drift_config = config["model"]["drift"]
    bank = VectorTrajectoryBank(
        config["data"]["train"], options.get("start_time_idx", 0),
        options.get("end_time_idx"), options.get("balanced_bins", 0), seed
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(options.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, int(options.get("decay_step", 200)), float(options.get("decay_factor", .5))
    )
    batch = int(options.get("batch_size", 256))
    epochs = int(options.get("epochs", 2000))
    batches = int(options.get("batches_per_epoch", 500))
    patch = int(drift_config.get("patch_size", 13))
    history = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        for _ in range(batches):
            _, drift_context, current, following, dt = bank.sample_window(
                batch, patch // 2, patch // 2, 0
            )
            beta = torch.full((batch,), bank.beta, device=device)
            h = torch.full((batch,), bank.external_field, device=device)
            prediction = model(drift_context.to(device), beta, h)
            target = ((following - current) / dt[:, None]).to(device)
            loss = (prediction - target).square().mean()
            if options.get("ball_projection", True):
                centre = drift_context[:, :, patch // 2, patch // 2].to(device)
                loss = loss + float(options.get("ball_weight", 1e-3)) * torch.relu(
                    (centre + dt[:, None].to(device) * prediction).norm(dim=1) - 1
                ).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        row = {"epoch": epoch, "loss": total / batches,
               "learning_rate": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row), flush=True)
    output = context.stage_dir(stage.name)
    checkpoint = output / "drift_last.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": drift_config,
                "model_class": config["model"]["class"], "selection": "last_epoch",
                "epoch": epochs}, checkpoint)
    save_history(output / "history.json", history)
    return {"checkpoint": str(checkpoint)}
