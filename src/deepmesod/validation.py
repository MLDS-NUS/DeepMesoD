from pathlib import Path

import torch

from deepmesod.classes import (
    get_dynamical_class,
    get_loss,
    get_trainer,
    load_extensions,
)
from deepmesod.data import FilmTrajectory, inverse_temperature, load_trajectory
from deepmesod.operators import validate_exact_local_divergence
from deepmesod.training.procedure import TrainingProcedure


def _trajectory_summary(path, split, transitions):
    trajectory = load_trajectory(path, mmap=True)
    available = trajectory.field.shape[0] - 1
    if transitions > available:
        raise ValueError(
            f"{path} provides {available} transitions but the configuration requests "
            f"{transitions}"
        )
    return {
        "split": split,
        "path": str(trajectory.source.resolve()),
        "shape": list(trajectory.field.shape),
        "selected_transitions": transitions if split == "train" else None,
        "available_transitions": available,
        "time_start": float(trajectory.times[0]),
        "time_end": float(trajectory.times[-1]),
        "boundary": trajectory.metadata["boundary"],
        "beta": inverse_temperature(trajectory.metadata),
        "coarse_volume": float(trajectory.metadata["coarse_volume"]),
    }, trajectory


def validate_experiment(config, *, require_fresh_output=False):
    errors = []
    warnings = []
    rows = []
    load_extensions(config.get("extensions"), config.get("_config_dir"))

    try:
        procedure = TrainingProcedure(config["training"]["procedure"])
        for stage in procedure.stages:
            get_trainer(stage.trainer)
            if stage.trainer in {"local_drift", "free_energy_interaction"}:
                get_loss(stage.loss)
    except (ImportError, KeyError, TypeError, ValueError) as error:
        errors.append(f"training procedure: {error}")
        procedure = None

    try:
        model_class = get_dynamical_class(config["model"]["class"])()
        model_class.build_drift_model(config["model"]["drift"])
        factor = model_class.build_factor(config["model"]["factor"])
        model_class.build_thermodynamic_model(config["model"]["thermodynamics"])
        validate_exact_local_divergence(factor)
    except (ImportError, KeyError, TypeError, ValueError) as error:
        errors.append(f"model class: {error}")

    transitions = int(config.get("data", {}).get("transitions_per_trajectory", 0))
    if transitions < 1:
        errors.append("data.transitions_per_trajectory must be positive")
    loaded = {"train": [], "test": []}
    is_film = config.get("model", {}).get("class") == "model_a_film"
    for split, trajectories in loaded.items():
        paths = config.get("data", {}).get(split, [])
        if not paths:
            errors.append(f"data.{split} must contain at least one trajectory")
            continue
        for path in paths:
            try:
                if is_film:
                    trajectory = FilmTrajectory.load(
                        path, dt=float(config.get("data", {}).get("dt", 0.002)),
                        device="cpu",
                    )
                    available = trajectory.field.shape[0] - 1
                    if transitions > available:
                        raise ValueError(
                            f"{path} provides {available} transitions but the configuration requests {transitions}"
                        )
                    row = {
                        "split": split, "path": str(trajectory.source),
                        "shape": list(trajectory.field.shape),
                        "selected_transitions": transitions if split == "train" else None,
                        "available_transitions": available,
                        "time_start": float(trajectory.times[0]),
                        "time_end": float(trajectory.times[-1]),
                        "boundary": trajectory.metadata["boundary"],
                        "beta": float(config["physics"]["beta"]),
                        "coarse_volume": float(config["physics"]["coarse_volume"]),
                    }
                else:
                    row, trajectory = _trajectory_summary(path, split, transitions)
                rows.append(row)
                trajectories.append(trajectory)
            except (OSError, KeyError, TypeError, ValueError) as error:
                errors.append(f"{split} trajectory {path}: {error}")

    for split, trajectories in loaded.items():
        settings = {
            (
                tuple(item.field.shape[1:]),
                (float(config["physics"]["beta"]) if is_film else inverse_temperature(item.metadata)),
                (float(config["physics"]["coarse_volume"]) if is_film else float(item.metadata["coarse_volume"])),
                (float(config["physics"].get("external_field", 0)) if is_film else float(item.metadata.get("external_field", 0))),
            )
            for item in trajectories
        }
        if len(settings) > 1:
            errors.append(f"data.{split} trajectories have inconsistent physical settings")

    tests = loaded["test"]
    if tests:
        reference_shape = tests[0].field.shape
        reference_times = tests[0].times
        reference_initial = tests[0].field[0]
        for trajectory in tests[1:]:
            if trajectory.field.shape != reference_shape or not torch.allclose(
                trajectory.times, reference_times
            ):
                errors.append("test trajectories must share shape and time coordinates")
                break
            if not torch.allclose(trajectory.field[0], reference_initial):
                errors.append("test trajectories must share the initial coarse-grained field")
                break

    output = Path(config["run"]["output_dir"])
    inputs = [Path(row["path"]) for row in rows]
    if any(output == path or output in path.parents or path in output.parents for path in inputs):
        errors.append("run.output_dir must be separate from every trajectory directory")
    if require_fresh_output and output.exists() and any(output.iterdir()):
        errors.append(f"run.output_dir is not empty: {output}")
    elif output.exists() and any(output.iterdir()):
        warnings.append(f"run.output_dir already contains files: {output}")

    stage3 = None
    if procedure is not None:
        for stage in procedure.stages:
            if stage.trainer in {"free_energy_interaction", "vector_free_energy_interaction",
                                 "film_free_energy_interaction"}:
                stage3 = {
                    "name": stage.name,
                    "loss": stage.loss,
                    "objective": stage.config.get("objective"),
                }
                expected = {
                    "direct_forward_rate": "direct",
                    "cg_force_target": "cg",
                }.get(stage.loss)
                if expected is not None and stage.config.get("objective", expected) != expected:
                    errors.append("Stage 3 loss and objective disagree")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "run": {
            "name": config["run"].get("name"),
            "seed": config["run"].get("seed"),
            "output_dir": str(output),
        },
        "model_class": config["model"].get("class"),
        "forward_increment": "single_step_within_each_trajectory",
        "stage3": stage3,
        "trajectories": rows,
    }
