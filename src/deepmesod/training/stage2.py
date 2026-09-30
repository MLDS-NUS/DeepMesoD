import json

import torch

from deepmesod.classes import register_trainer
from deepmesod.data import ScalarTrajectoryBank

from .common import artifact_path, device_from_config, dynamical_class, save_history
from .likelihood import patch_grid, window_gaussian_nll


@register_trainer("finite_range_factor")
def train_finite_range_factor(context, stage):
    config = context.config
    options = stage.config
    device = device_from_config(config)
    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    model_class = dynamical_class(config)
    drift = model_class.build_drift_model(config["model"]["drift"]).to(device)
    drift_payload = torch.load(
        artifact_path(context, stage.requires[0], "checkpoint"),
        map_location=device,
        weights_only=False,
    )
    drift.load_state_dict(drift_payload["model_state_dict"])
    drift.eval()
    for parameter in drift.parameters():
        parameter.requires_grad_(False)
    factor = model_class.build_factor(config["model"]["factor"]).to(device)
    bank = ScalarTrajectoryBank(
        config["data"]["train"],
        config["data"]["transitions_per_trajectory"],
        bins=options.get("balanced_bins", 10),
        sampling=options.get("sampling", "center"),
        patch_size=config["model"]["factor"].get("patch_size", 15),
        seed=seed,
    )
    optimizer = torch.optim.Adam(
        factor.parameters(), lr=float(options.get("learning_rate", 5e-4))
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        int(options.get("decay_step", 200)),
        float(options.get("decay_factor", 0.5)),
    )
    batch_size = int(options.get("batch_size", 256))
    patch_size = int(config["model"]["factor"].get("patch_size", 15))
    window_radius = int(config["model"]["factor"].get("likelihood_radius", 1))
    history = []
    epochs = int(options.get("epochs", 2000))
    batches = int(options.get("batches_per_epoch", 100))
    for epoch in range(1, epochs + 1):
        totals = []
        factor.train()
        for _ in range(batches):
            context_patch, current, following, dt = bank.sample_window(
                batch_size, patch_size, window_radius
            )
            context_patch = context_patch.to(device)
            current = current.to(device)
            following = following.to(device)
            dt = dt.to(device)
            patches = patch_grid(context_patch, patch_size, factor.n_likelihood_rows)
            with torch.no_grad():
                mean_drift = drift(patches.reshape(-1, 1, patch_size, patch_size)).reshape(
                    batch_size, factor.n_likelihood_rows
                )
            loss, metrics = window_gaussian_nll(
                factor,
                context_patch,
                current,
                following,
                dt,
                mean_drift,
                bank.beta,
                bank.coarse_volume,
                float(options.get("jitter", 1e-8)),
                patches=patches,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(factor.parameters(), 5)
            optimizer.step()
            totals.append(metrics)
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": sum(item["loss"] for item in totals) / batches,
            "quadratic_per_dimension": sum(
                item["quadratic_per_dimension"] for item in totals
            )
            / batches,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    output = context.stage_dir(stage.name)
    checkpoint = output / "factor_last.pt"
    torch.save(
        {
            "model_state_dict": factor.state_dict(),
            "model_config": factor.configuration(),
            "beta": bank.beta,
            "coarse_volume": bank.coarse_volume,
            "selection": "last_epoch",
            "epoch": epochs,
        },
        checkpoint,
    )
    save_history(output / "history.json", history)
    return {"checkpoint": str(checkpoint)}
