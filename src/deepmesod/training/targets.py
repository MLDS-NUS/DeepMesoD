import json
from pathlib import Path

import torch

from deepmesod.data import inverse_temperature, load_trajectories
from deepmesod.operators import divergence_on_field_local_exact


def dissipation_rows_on_field(factor, coefficients):
    neighbors = torch.stack(
        [
            torch.roll(coefficients, shifts=(-dy, -dx), dims=(-2, -1))
            for dy, dx in factor.force_offsets
        ],
        dim=1,
    )
    batch, _, _, height, width = neighbors.shape
    neighbors = neighbors.permute(0, 3, 4, 1, 2).reshape(
        batch * height * width, factor.n_force_rows, factor.n_factors
    )
    rows = factor._center_covariance_row(neighbors)
    return rows.reshape(batch, height, width, factor.n_force_rows)


def batched_conjugate_gradient(action, right_hand_side, iterations, tolerance):
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_norm = residual.flatten(1).square().sum(1)
    initial_norm = residual_norm.sqrt().clamp_min(1e-20)
    completed = torch.zeros(
        right_hand_side.shape[0], dtype=torch.long, device=right_hand_side.device
    )
    active = torch.ones_like(residual_norm, dtype=torch.bool)
    for iteration in range(1, iterations + 1):
        action_direction = action(direction)
        denominator = (direction * action_direction).flatten(1).sum(1).clamp_min(1e-20)
        step = torch.where(
            active, residual_norm / denominator, torch.zeros_like(residual_norm)
        )
        solution = solution + step[:, None, None] * direction
        next_residual = residual - step[:, None, None] * action_direction
        next_norm = next_residual.flatten(1).square().sum(1)
        just_finished = active & (next_norm.sqrt() <= tolerance * initial_norm)
        completed[just_finished] = iteration
        active = active & ~just_finished
        coefficient = torch.where(
            active,
            next_norm / residual_norm.clamp_min(1e-20),
            torch.zeros_like(next_norm),
        )
        direction = next_residual + coefficient[:, None, None] * direction
        residual = next_residual
        residual_norm = next_norm
        if not active.any():
            break
    completed[completed == 0] = iteration
    relative = residual_norm.sqrt() / initial_norm
    return solution, completed, relative


def prepare_stage3_cache(
    trajectories,
    factor,
    output_dir,
    *,
    objective,
    batch_size,
    ridge_relative,
    cg_iterations,
    cg_tolerance,
    device,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for trajectory_index, trajectory in enumerate(trajectories):
        current, following, dt, rate = trajectory.transitions
        right_hand_sides = []
        force_targets = []
        dissipation_rows = []
        residuals = []
        iterations_used = []
        beta = inverse_temperature(trajectory.metadata)
        coarse_volume = float(trajectory.metadata["coarse_volume"])
        for start in range(0, current.shape[0], batch_size):
            stop = min(start + batch_size, current.shape[0])
            field = current[start:stop, None].to(device)
            observed_rate = rate[start:stop].to(device)
            divergence, coefficients = divergence_on_field_local_exact(
                factor, field, return_coefficients=True
            )
            right_hand_side = divergence / (beta * coarse_volume) - observed_rate
            rows = dissipation_rows_on_field(factor, coefficients)
            right_hand_sides.append(right_hand_side.cpu())
            dissipation_rows.append(rows.cpu())
            if objective == "cg":
                dissipation_scale = coefficients.square().sum(1).mean(dim=(-2, -1))
                ridge = ridge_relative * dissipation_scale

                def regularized_action(force, coefficients=coefficients, ridge=ridge):
                    return (
                        factor.apply_dissipation(coefficients, force)
                        + ridge[:, None, None] * force
                    )

                target, completed, relative = batched_conjugate_gradient(
                    regularized_action,
                    right_hand_side,
                    cg_iterations,
                    cg_tolerance,
                )
                force_targets.append(target.cpu())
                iterations_used.append(completed.cpu())
                residuals.append(relative.cpu())
        payload = {
            "fields": current.contiguous(),
            "following": following.contiguous(),
            "dt": dt.float().contiguous(),
            "right_hand_sides": torch.cat(right_hand_sides),
            "dissipation_rows": torch.cat(dissipation_rows),
            "beta": beta,
            "coarse_volume": coarse_volume,
            "source": str(trajectory.source),
            "objective": objective,
        }
        if objective == "cg":
            payload["force_targets"] = torch.cat(force_targets)
            payload["cg_iterations"] = torch.cat(iterations_used)
            payload["cg_relative_residual"] = torch.cat(residuals)
        path = output_dir / f"trajectory_{trajectory_index}.pt"
        torch.save(payload, path)
        summary = {
            "trajectory": trajectory_index,
            "path": str(path),
            "transitions": int(current.shape[0]),
            "rhs_rms": float(payload["right_hand_sides"].square().mean().sqrt()),
        }
        if objective == "cg":
            summary["cg_max_relative_residual"] = float(
                payload["cg_relative_residual"].max()
            )
        summaries.append(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return [Path(item["path"]) for item in summaries]


def load_training_trajectories(config):
    return load_trajectories(
        config["data"]["train"], config["data"]["transitions_per_trajectory"]
    )
