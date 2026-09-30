"""Stage-2 finite-range in-plane mobility training for CrBr3 films."""

import json
import torch
import torch.nn.functional as F

from deepmesod.classes import register_trainer
from deepmesod.data import FilmTrajectoryBank, film_patch_grid
from deepmesod.operators import VectorFiniteRangeFactor
from .common import artifact_path, device_from_config, dynamical_class, save_history
from .vector_likelihood import vector_window_gaussian_nll


@register_trainer("film_finite_range_factor")
def train_film_finite_range_factor(context, stage):
    config, options = context.config, stage.config
    device = device_from_config(config)
    model_class = dynamical_class(config)
    drift = model_class.build_drift_model(config["model"]["drift"]).to(device)
    payload = torch.load(artifact_path(context, stage.requires[0], "checkpoint"),
                         map_location=device, weights_only=False)
    drift.load_state_dict(payload.get("model_state_dict", payload))
    drift.eval()
    for parameter in drift.parameters():
        parameter.requires_grad_(False)
    factor = model_class.build_factor(config["model"]["factor"]).to(device)
    beta = float(config["physics"]["beta"])
    volume = float(config["physics"]["coarse_volume"])
    bank = FilmTrajectoryBank(config["data"]["train"], config["data"].get("dt", .002),
                              seed=int(config["run"]["seed"]))
    optimizer = torch.optim.Adam(factor.parameters(), lr=float(options.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(options.get("decay_step", 200)),
                                                float(options.get("decay_factor", .5)))
    history = []
    for epoch in range(1, int(options.get("epochs", 1000)) + 1):
        rows = []
        for _ in range(int(options.get("steps_per_epoch", options.get("batches_per_epoch", 20)))):
            factor_radius = factor.patch_size // 2 + factor.likelihood_radius
            drift_radius = int(config["model"]["drift"].get("patch_size", 7)) // 2 + factor.likelihood_radius
            fctx, dctx, current, following, dt, z = bank.sample_slab(
                int(options.get("batch_size", 128)), factor_radius, drift_radius,
                factor.likelihood_radius)
            fctx, dctx = fctx.to(device), dctx.to(device)
            rows_count = factor.n_likelihood_rows
            patches = film_patch_grid(dctx, int(config["model"]["drift"].get("patch_size", 7)),
                                      bank.n_components)
            with torch.no_grad():
                pred = drift(patches.flatten(0, 1).to(device)).reshape(
                    patches.shape[0], rows_count, bank.n_components, bank.n_layers)
                mean = pred[torch.arange(pred.shape[0], device=device), :, :, z.to(device)]
                mean = mean.reshape(pred.shape[0], -1)
            loss, metrics = vector_window_gaussian_nll(
                factor, fctx.to(device), current.to(device), following.to(device), dt.to(device),
                mean, beta, volume, float(options.get("jitter", 1e-8)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(factor.parameters(), float(options.get("grad_clip", 5.0)))
            optimizer.step()
            rows.append(metrics)
        scheduler.step()
        row = {"epoch": epoch, "loss": sum(r["loss"] for r in rows) / len(rows),
               "quadratic_per_dimension": sum(r["quadratic_per_dimension"] for r in rows) / len(rows),
               "learning_rate": scheduler.get_last_lr()[0]}
        history.append(row)
        if epoch == 1 or epoch % int(options.get("log_interval", 20)) == 0:
            print(json.dumps(row), flush=True)
    output = context.stage_dir(stage.name)
    checkpoint = output / "factor_last.pt"
    torch.save({"model_state_dict": factor.state_dict(), "model_config": factor.configuration(),
                "beta": beta, "coarse_volume": volume, "dt": bank.dt,
                "epoch": history[-1]["epoch"]}, checkpoint)
    save_history(output / "history.json", history)
    return {"checkpoint": str(checkpoint)}

