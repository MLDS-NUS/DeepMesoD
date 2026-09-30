"""Vector Stage-2 trainer: frozen Stage-1 mean plus full window likelihood."""

import json

import torch

from deepmesod.classes import register_trainer
from deepmesod.data import VectorTrajectoryBank

from .common import artifact_path, device_from_config, dynamical_class, save_history
from .vector_likelihood import vector_patch_grid, vector_window_gaussian_nll


@register_trainer("vector_finite_range_factor")
def train_vector_finite_range_factor(context, stage):
    config, options = context.config, stage.config
    device = device_from_config(config)
    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    model_class = dynamical_class(config)
    drift = model_class.build_drift_model(config["model"]["drift"]).to(device)
    drift_payload = torch.load(
        artifact_path(context, stage.requires[0], "checkpoint"),
        map_location=device, weights_only=False,
    )
    drift.load_state_dict(drift_payload.get("model_state_dict", drift_payload))
    drift.eval()
    for parameter in drift.parameters():
        parameter.requires_grad_(False)
    factor = model_class.build_factor(config["model"]["factor"]).to(device)
    factor_config = config["model"]["factor"]
    bank = VectorTrajectoryBank(
        config["data"]["train"], options.get("start_time_idx", 0),
        options.get("end_time_idx"), options.get("balanced_bins", 10), seed
    )
    optimizer = torch.optim.Adam(factor.parameters(), lr=float(options.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, int(options.get("decay_step", 200)), float(options.get("decay_factor", .5))
    )
    batch = int(options.get("batch_size", 128))
    epochs = int(options.get("epochs", 1000))
    batches = int(options.get("batches_per_epoch", 20))
    factor_patch = factor.patch_size
    drift_patch = int(config["model"]["drift"].get("patch_size", 13))
    history = []
    for epoch in range(1, epochs + 1):
        metrics = []
        for _ in range(batches):
            context_patch, drift_context, current, following, dt = bank.sample_window(
                batch,
                factor_patch // 2 + factor.likelihood_radius,
                drift_patch // 2 + factor.likelihood_radius,
                factor.likelihood_radius,
            )
            context_patch = context_patch.to(device)
            current, following, dt = current.to(device), following.to(device), dt.to(device)
            drift_patches = vector_patch_grid(
                drift_context.to(device), drift_patch, factor.n_likelihood_rows,
                factor.n_components
            )
            with torch.no_grad():
                beta = torch.full((batch * factor.n_likelihood_rows,), bank.beta, device=device)
                h = torch.full_like(beta, bank.external_field)
                mean = drift(drift_patches.flatten(0, 1), beta, h)
                mean = mean.reshape(batch, factor.n_likelihood_rows, factor.n_components)
                mean = mean.reshape(batch, -1)
            loss, row = vector_window_gaussian_nll(
                factor, context_patch, current, following, dt, mean,
                bank.beta, bank.coarse_volume, options.get("jitter", 1e-8),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(factor.parameters(), options.get("grad_clip", 5.0))
            optimizer.step()
            metrics.append(row)
        scheduler.step()
        row = {"epoch": epoch, "loss": sum(m["loss"] for m in metrics) / batches,
               "quadratic_per_dimension": sum(m["quadratic_per_dimension"] for m in metrics) / batches,
               "learning_rate": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row), flush=True)
    output = context.stage_dir(stage.name)
    checkpoint = output / "factor_last.pt"
    torch.save({"model_state_dict": factor.state_dict(), "model_config": factor.configuration(),
                "beta": bank.beta, "external_field": bank.external_field,
                "coarse_volume": bank.coarse_volume, "epoch": epochs,
                "selection": "last_epoch", "final_training_metrics": history[-1]}, checkpoint)
    save_history(output / "history.json", history)
    return {"checkpoint": str(checkpoint)}
