import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepmesod.classes import get_dynamical_class
from deepmesod.data import load_trajectory
from deepmesod.inference.rollout import load_thermodynamics
from deepmesod.operators.checkpoints import load_factor
from deepmesod.training.common import device_from_config

from .metrics import trajectory_metrics
from .operators import conditional_complete_response, interaction_diagnostics


def _stage(config, trainer):
    for stage in config["training"]["procedure"]:
        if stage["trainer"] == trainer:
            return stage["name"]
    raise KeyError(f"the procedure does not contain trainer '{trainer}'")


def _checkpoint(config, trainer, filename):
    name = _stage(config, trainer)
    explicit = config.get("artifacts", {}).get(name, {}).get("checkpoint")
    return Path(explicit) if explicit else Path(config["run"]["output_dir"]) / name / filename


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_run(config):
    rollout_path = Path(config["run"]["output_dir"]) / "inference" / "rollout.pt"
    payload = torch.load(rollout_path, map_location="cpu", weights_only=False)
    prediction = payload["field"]
    reference = torch.zeros_like(prediction)
    for path in config["data"]["test"]:
        trajectory = load_trajectory(path)
        reference += trajectory.field[: prediction.shape[0]] / len(config["data"]["test"])
    metrics = trajectory_metrics(prediction, reference)
    serializable = {
        key: value.tolist() if torch.is_tensor(value) else value
        for key, value in metrics.items()
    }
    output = rollout_path.parent / "evaluation.json"
    output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output


def _load_histories(config):
    histories = []
    root = Path(config["run"]["output_dir"])
    for stage in config["training"]["procedure"]:
        path = root / stage["name"] / "history.json"
        if path.is_file():
            rows = json.loads(path.read_text(encoding="utf-8"))
            histories.append(
                {
                    "name": stage["name"],
                    "trainer": stage["trainer"],
                    "loss": stage["loss"],
                    "rows": rows,
                    "finite": all(
                        np.isfinite(value)
                        for row in rows
                        for value in row.values()
                        if isinstance(value, (int, float))
                    ),
                }
            )
    return histories


def _reference_and_samples(config, count, max_frames):
    reference = None
    selected = None
    for index, path in enumerate(config["data"]["test"]):
        trajectory = load_trajectory(path)
        field = trajectory.field[:count]
        if reference is None:
            reference = torch.zeros_like(field)
            indices = torch.linspace(
                0, field.shape[0] - 1, min(max_frames, field.shape[0])
            ).round().long()
            selected = field[indices].clone()
        reference += field / len(config["data"]["test"])
    return reference, selected


def _cg_summary(config):
    try:
        stage = _stage(config, "free_energy_interaction")
    except KeyError:
        return None
    path = Path(config["run"]["output_dir"]) / stage / "targets" / "summary.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    residuals = [row["cg_max_relative_residual"] for row in rows if "cg_max_relative_residual" in row]
    return {
        "trajectories": rows,
        "max_relative_residual": max(residuals) if residuals else None,
    }


def _plot_history(axis, history):
    rows = history["rows"]
    coordinate = "epoch" if "epoch" in rows[0] else "step"
    axis.plot([row[coordinate] for row in rows], [row["loss"] for row in rows])
    axis.set_title(history["name"])
    axis.set_xlabel(coordinate.capitalize())
    axis.set_ylabel("Training loss")


def _report_figure(
    output,
    histories,
    rollout_payload,
    reference,
    state,
    energy,
    response,
    kernel,
):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for index, axis in enumerate(axes[0, :3]):
        if index < len(histories):
            _plot_history(axis, histories[index])
        else:
            axis.set_visible(False)

    times = rollout_payload["times"].numpy()
    prediction = rollout_payload["field"]
    error = (prediction - reference).abs().mean(dim=(-2, -1)).numpy()
    axes[0, 3].plot(times, error, color="#1f77b4")
    axes[0, 3].set_title("Rollout error")
    axes[0, 3].set_xlabel("Time")
    axes[0, 3].set_ylabel("MAE")

    axes[1, 0].plot(times, reference.mean(dim=(-2, -1)), color="#2ca02c", label="Reference")
    axes[1, 0].plot(times, prediction.mean(dim=(-2, -1)), color="#1f77b4", label="DeepMesoD")
    axes[1, 0].set_title("Spatial mean")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Mean coarse-grained field")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(state, energy, color="#1f77b4")
    axes[1, 1].set_title("Homogeneous free energy")
    axes[1, 1].set_xlabel("Coarse-grained field")
    axes[1, 1].set_ylabel("g(m), shifted")

    valid = response["count"] > 0
    response_state = response["state"][valid]
    response_mean = response["mean"][valid]
    response_sd = response["standard_deviation"][valid]
    axes[1, 2].plot(response_state, response_mean, color="#1f77b4")
    axes[1, 2].fill_between(
        response_state,
        response_mean - response_sd,
        response_mean + response_sd,
        color="#1f77b4",
        alpha=0.15,
        label="Within-bin SD",
    )
    axes[1, 2].set_title("Dissipation operator")
    axes[1, 2].set_xlabel("Center coarse-grained field")
    axes[1, 2].set_ylabel(r"$[K[m]\mathbf{1}]_x$")
    axes[1, 2].legend(frameon=False)

    limit = float(np.abs(kernel).max())
    image = axes[1, 3].imshow(
        kernel,
        cmap="coolwarm",
        vmin=-limit if limit else None,
        vmax=limit if limit else None,
        interpolation="none",
    )
    axes[1, 3].set_title("Interaction kernel")
    axes[1, 3].set_xlabel("Grid offset")
    axes[1, 3].set_ylabel("Grid offset")
    figure.colorbar(image, ax=axes[1, 3], label="Kernel weight")

    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)


def generate_run_report(config):
    run_root = Path(config["run"]["output_dir"])
    output_dir = run_root / "report"
    output_dir.mkdir(parents=False, exist_ok=False)
    device = device_from_config(config)
    builder = get_dynamical_class(config["model"]["class"])
    model_class = builder()
    factor_path = _checkpoint(config, "finite_range_factor", "factor_last.pt")
    thermodynamics_path = _checkpoint(
        config, "free_energy_interaction", "thermodynamics_last.pt"
    )
    factor, factor_payload = load_factor(factor_path, device)
    thermodynamics, thermodynamics_payload = load_thermodynamics(
        thermodynamics_path, model_class, config, device
    )
    histories = _load_histories(config)
    rollout_path = run_root / "inference" / "rollout.pt"
    rollout_payload = torch.load(rollout_path, map_location="cpu", weights_only=False)
    reference, fields = _reference_and_samples(
        config,
        rollout_payload["field"].shape[0],
        int(config.get("report", {}).get("max_operator_frames", 16)),
    )
    response = conditional_complete_response(
        factor,
        fields,
        bins=int(config.get("report", {}).get("state_bins", 40)),
        batch_size=int(config.get("report", {}).get("operator_batch_size", 4)),
        device=str(device),
    )
    interaction = interaction_diagnostics(thermodynamics)
    state = torch.linspace(-1, 1, 201, device=device)
    with torch.no_grad():
        energy = thermodynamics.free_energy(state)
        energy = energy - energy.min()
        kernel = thermodynamics.interaction.weight[0, 0]
    metrics = trajectory_metrics(rollout_payload["field"], reference)
    report = {
        "status": "pass"
        if metrics["finite"] and all(item["finite"] for item in histories)
        else "fail",
        "run": {
            "name": config["run"].get("name"),
            "seed": config["run"].get("seed"),
            "model_class": config["model"]["class"],
            "stage3_objective": thermodynamics_payload.get("objective"),
            "checkpoint_selection": thermodynamics_payload.get("selection"),
        },
        "rollout": {
            "steps": int(rollout_payload["field"].shape[0] - 1),
            "mean_mae": metrics["mean_mae"],
            "final_mae": metrics["final_mae"],
            "finite": metrics["finite"],
        },
        "training": [
            {
                "name": item["name"],
                "trainer": item["trainer"],
                "loss": item["loss"],
                "finite": item["finite"],
                "final": item["rows"][-1],
            }
            for item in histories
        ],
        "interaction": interaction,
        "dissipation": {
            "representation": "K=B B*",
            "displayed_functional": "conditional complete response [K[m]1]_x",
            "factor_checkpoint_epoch": factor_payload.get("epoch"),
            "occupied_bins": int((response["count"] > 0).sum()),
            "total_bins": int(response["count"].numel()),
        },
        "cg": _cg_summary(config),
        "limitations": [
            "This report covers one training seed and does not estimate model-seed uncertainty.",
            "The learned dissipation response has no microscopic or hydrodynamic reference in this generic report.",
            "Finite losses and rollouts are numerical checks, not evidence that the selected dynamical class is physically complete.",
            "The displayed free-energy density is shifted by its own minimum.",
        ],
        "provenance": {
            "config": config.get("_config_path"),
            "data": list(config["data"]["train"]) + list(config["data"]["test"]),
            "factor_checkpoint": str(factor_path),
            "factor_sha256": _sha256(factor_path),
            "thermodynamics_checkpoint": str(thermodynamics_path),
            "thermodynamics_sha256": _sha256(thermodynamics_path),
            "operator_frames": int(fields.shape[0]),
            "state_bins": int(response["count"].numel()),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez(
        output_dir / "diagnostics.npz",
        free_energy_state=state.cpu().numpy(),
        free_energy=energy.cpu().numpy(),
        dissipation_state=response["state"].numpy(),
        dissipation_mean=response["mean"].numpy(),
        dissipation_standard_deviation=response["standard_deviation"].numpy(),
        dissipation_count=response["count"].numpy(),
        interaction_kernel=kernel.cpu().numpy(),
    )
    _report_figure(
        output_dir / "diagnostics",
        histories,
        rollout_payload,
        reference,
        state.cpu().numpy(),
        energy.cpu().numpy(),
        {key: value.numpy() for key, value in response.items()},
        kernel.cpu().numpy(),
    )
    markdown = f"""# DeepMesoD run report

Run `{report['run']['name']}` used model class `{report['run']['model_class']}`, seed {report['run']['seed']}, and the final scheduled `{report['run']['stage3_objective']}` Stage 3 checkpoint.

## Numerical outcome

- Rollout steps: {report['rollout']['steps']}
- Mean MAE: {report['rollout']['mean_mae']:.6g}
- Final MAE: {report['rollout']['final_mae']:.6g}
- Finite rollout: {report['rollout']['finite']}
- Interaction constraint: `{interaction['constraint']}`
- Kernel sum: {interaction['kernel_sum']:.9g}
- Centrosymmetry relative error: {interaction['centrosymmetry_relative_error']:.6g}
- Occupied dissipation-state bins: {report['dissipation']['occupied_bins']}/{report['dissipation']['total_bins']}

The dissipation panel displays the complete constant-force response `[K[m]1]_x`. Its band is the within-bin standard deviation across sampled field sites, not uncertainty across training seeds. The free-energy curve is shifted by its own minimum. The interaction heatmap displays signed weights with a color scale centered at zero.

## Interpretation boundary

""" + "\n".join(f"- {item}" for item in report["limitations"]) + "\n"
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    return output_dir / "report.json"
