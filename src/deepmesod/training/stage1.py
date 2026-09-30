import json

import torch

from deepmesod.classes import get_loss, register_trainer
from deepmesod.data import ScalarTrajectoryBank

from .common import device_from_config, dynamical_class, save_history


@register_trainer("local_drift")
def train_local_drift(context, stage):
    config = context.config
    options = stage.config
    device = device_from_config(config)
    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    model_class = dynamical_class(config)
    model = model_class.build_drift_model(config["model"]["drift"]).to(device)
    bank = ScalarTrajectoryBank(
        config["data"]["train"],
        config["data"]["transitions_per_trajectory"],
        bins=options.get("balanced_bins", 10),
        sampling=options.get("sampling", "joint"),
        patch_size=config["model"]["drift"].get("patch_size", 15),
        seed=seed,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(options.get("learning_rate", 1e-3))
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        int(options.get("decay_step", 500)),
        float(options.get("decay_factor", 0.5)),
    )
    loss_function = get_loss(stage.loss)
    batch_size = int(options.get("batch_size", 256))
    patch_size = int(config["model"]["drift"].get("patch_size", 15))
    history = []
    epochs = int(options.get("epochs", 2000))
    batches = int(options.get("batches_per_epoch", 500))
    for epoch in range(1, epochs + 1):
        total = 0.0
        model.train()
        for _ in range(batches):
            patches, target = bank.sample(batch_size, patch_size)
            loss = loss_function(model(patches.to(device)), target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": total / batches,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    output = context.stage_dir(stage.name)
    checkpoint = output / "drift_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config["model"]["drift"],
            "model_class": config["model"]["class"],
            "selection": "last_epoch",
            "epoch": epochs,
        },
        checkpoint,
    )
    save_history(output / "history.json", history)
    return {"checkpoint": str(checkpoint)}
