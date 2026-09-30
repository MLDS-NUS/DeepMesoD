import json

import torch
import torch.nn.functional as F

from deepmesod.classes import get_loss, register_trainer
from deepmesod.data import periodic_patches
from deepmesod.operators.checkpoints import load_factor

from .common import artifact_path, device_from_config, dynamical_class, save_history
from .losses import state_weights
from .targets import load_training_trajectories, prepare_stage3_cache


@register_trainer("free_energy_interaction")
def train_free_energy_interaction(context, stage):
    config = context.config
    options = stage.config
    objective_by_loss = {
        "direct_forward_rate": "direct",
        "cg_force_target": "cg",
    }
    if stage.loss not in objective_by_loss:
        raise ValueError("the built-in thermodynamic trainer requires a direct or CG loss")
    objective = objective_by_loss[stage.loss]
    configured = options.get("objective", objective)
    if configured != objective:
        raise ValueError("Stage 3 objective and loss disagree")
    device = device_from_config(config)
    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    factor, _ = load_factor(
        artifact_path(context, stage.requires[-1], "checkpoint"), device
    )
    output = context.stage_dir(stage.name)
    cache_paths = prepare_stage3_cache(
        load_training_trajectories(config),
        factor,
        output / "targets",
        objective=objective,
        batch_size=int(options.get("target_batch_size", 4)),
        ridge_relative=float(options.get("ridge_relative", 1e-4)),
        cg_iterations=int(options.get("cg_iterations", 100)),
        cg_tolerance=float(options.get("cg_tolerance", 1e-5)),
        device=device,
    )
    datasets = [
        torch.load(path, map_location="cpu", weights_only=False) for path in cache_paths
    ]
    fields = torch.cat([item["fields"] for item in datasets])
    right_hand_sides = torch.cat([item["right_hand_sides"] for item in datasets])
    dissipation_rows = torch.cat([item["dissipation_rows"] for item in datasets])
    force_targets = None
    if objective == "cg":
        force_targets = torch.cat([item["force_targets"] for item in datasets])
    sampling = options.get("sampling", "center")
    if sampling == "none":
        weights = torch.ones(1)
        bin_index = torch.zeros_like(fields, dtype=torch.long)
    elif sampling == "center":
        weights, bin_index = state_weights(
            fields,
            int(options.get("balanced_bins", 10)),
            float(options.get("max_state_weight", 10)),
        )
    else:
        raise ValueError("Stage 3 sampling must be 'none' or 'center'")
    model_class = dynamical_class(config)
    thermodynamics = model_class.build_thermodynamic_model(
        config["model"]["thermodynamics"]
    ).to(device)
    learning_rate = float(options.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(thermodynamics.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        int(options.get("decay_step", 40000)),
        float(options.get("decay_factor", 0.5)),
    )
    loss_function = get_loss(stage.loss)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(options.get("batch_size", 256))
    patch_size = int(config["model"]["thermodynamics"].get("patch_size", 15))
    patch_radius = patch_size // 2
    context_radius = patch_radius + factor.force_radius
    history = []
    steps = int(options.get("steps", 400000))
    for step in range(1, steps + 1):
        time = torch.randint(fields.shape[0], (batch_size,), generator=generator)
        row = torch.randint(fields.shape[-2], (batch_size,), generator=generator)
        column = torch.randint(fields.shape[-1], (batch_size,), generator=generator)
        weight = weights[bin_index[time, row, column]].to(device)
        if objective == "cg":
            patches = periodic_patches(fields, time, row, column, patch_size).to(device)
            prediction = thermodynamics.force_on_patches(patches, create_graph=True)
            target = force_targets[time, row, column].to(device)
        else:
            context_patch = periodic_patches(
                fields, time, row, column, 2 * context_radius + 1
            ).to(device)
            patch_grid = F.unfold(context_patch, kernel_size=patch_size)
            patch_grid = patch_grid.transpose(1, 2).reshape(
                batch_size,
                factor.n_force_rows,
                1,
                patch_size,
                patch_size,
            )
            force_grid = thermodynamics.force_on_patches(
                patch_grid.flatten(0, 1), create_graph=True
            ).reshape(batch_size, factor.n_force_rows)
            dissipation_row = dissipation_rows[time, row, column].to(device)
            prediction = (dissipation_row * force_grid).sum(1)
            target = right_hand_sides[time, row, column].to(device)
        loss = loss_function(prediction, target, weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(thermodynamics.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == steps:
            row_metrics = {
                "step": step,
                "loss": float(loss.detach()),
                "rmse": float((prediction.detach() - target).square().mean().sqrt()),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(row_metrics)
            print(json.dumps(row_metrics), flush=True)
    checkpoint = output / "thermodynamics_last.pt"
    torch.save(
        {
            "model_state_dict": thermodynamics.state_dict(),
            "model_config": config["model"]["thermodynamics"],
            "model_class": config["model"]["class"],
            "objective": objective,
            "selection": "last_step",
            "step": steps,
        },
        checkpoint,
    )
    torch.save(thermodynamics.free_energy.state_dict(), output / "F_net_model.pth")
    torch.save(thermodynamics.interaction.state_dict(), output / "J_kernel_model.pth")
    save_history(output / "history.json", history)
    return {
        "checkpoint": str(checkpoint),
        "target_cache": [str(path) for path in cache_paths],
        "objective": objective,
    }
