import argparse
import json
from pathlib import Path

from deepmesod.classes import load_extensions, registered_components
from deepmesod.config import apply_overrides, load_config
from deepmesod.data import (
    FilmTrajectory,
    coarse_grain_crbr3,
    coarse_grain_vampire,
    convert_legacy_ising_trajectory,
    load_trajectory,
)
from deepmesod.data_generators import (
    get_data_generator,
    registered_data_generators,
)
from deepmesod.evaluation import evaluate_run, generate_run_report
from deepmesod.inference import rollout, rollout_film, rollout_vector
from deepmesod.training import RunContext, TrainingProcedure
from deepmesod.validation import validate_experiment
from deepmesod.visualization import plot_figure2


def parser():
    root = argparse.ArgumentParser(prog="deepmesod")
    subcommands = root.add_subparsers(dest="command", required=True)
    subcommands.add_parser("components")
    validate_run = subcommands.add_parser("validate")
    validate_run.add_argument("--config", required=True)
    validate_run.add_argument("--set", action="append", default=[])
    data = subcommands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    validate = data_commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    generate = data_commands.add_parser("generate")
    generate.add_argument("--config", required=True)
    convert = data_commands.add_parser("convert-legacy-ising")
    convert.add_argument("--source", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    coarse = data_commands.add_parser("coarse-grain-vampire")
    coarse.add_argument("--input-root", required=True)
    coarse.add_argument("--output-dir", required=True)
    coarse.add_argument("--block-size", type=int, default=16)
    coarse.add_argument("--coarse-size", type=int, default=64)
    coarse.add_argument("--batch-size", type=int, default=10)
    coarse.add_argument("--overwrite", action="store_true")
    crbr3 = data_commands.add_parser(
        "coarse-grain-crbr3",
        help="coarse-grain VAMPIRE CrBr3 frames using absolute lattice coordinates",
    )
    crbr3.add_argument("--input-root", required=True)
    crbr3.add_argument("--output-dir", required=True)
    crbr3.add_argument("--stride-A", type=float, nargs=2, default=(50.0, 50.0), metavar=("X", "Y"))
    crbr3.add_argument("--block-A", type=float, nargs=2, default=None, metavar=("X", "Y"))
    crbr3.add_argument("--z-group", type=int, default=3)
    crbr3.add_argument("--coords-name", default="atoms-coords.data")
    crbr3.add_argument("--frame-stride", type=int, default=1)
    crbr3.add_argument("--max-frames", type=int, default=None)
    crbr3.add_argument("--time-unit", choices=("ns", "s", "index"), default="ns")
    crbr3.add_argument("--overwrite", action="store_true")
    train = subcommands.add_parser("train")
    train.add_argument("stage", nargs="?")
    train.add_argument("--config", required=True)
    train.add_argument("--set", action="append", default=[])
    infer = subcommands.add_parser("infer")
    infer.add_argument("--config", required=True)
    infer.add_argument("--set", action="append", default=[])
    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--set", action="append", default=[])
    report = subcommands.add_parser("report")
    report.add_argument("--config", required=True)
    report.add_argument("--set", action="append", default=[])
    reproduce = subcommands.add_parser("reproduce")
    reproduce.add_argument("--config", required=True)
    reproduce.add_argument("--set", action="append", default=[])
    plot = subcommands.add_parser("plot")
    plot.add_argument("figure", choices=["figure2"])
    plot.add_argument("--manifest", required=True)
    return root


def _load_with_overrides(path, overrides):
    return apply_overrides(load_config(path), overrides)


def _artifact(config, stage, filename):
    explicit = config.get("artifacts", {}).get(stage, {}).get("checkpoint")
    if explicit:
        return Path(explicit)
    return Path(config["run"]["output_dir"]) / stage / filename


def _artifact_for_trainer(config, trainer, filename):
    for stage in config["training"]["procedure"]:
        if stage["trainer"] == trainer:
            return _artifact(config, stage["name"], filename)
    raise KeyError(f"the procedure does not contain trainer '{trainer}'")


def _artifact_for_trainers(config, trainers, filename):
    for trainer in trainers:
        try:
            return _artifact_for_trainer(config, trainer, filename)
        except KeyError:
            continue
    raise KeyError(f"the procedure does not contain any of {trainers}")


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == "components":
        components = registered_components()
        components["data_generators"] = registered_data_generators()
        print(json.dumps(components, indent=2))
        return
    if arguments.command == "validate":
        config = _load_with_overrides(arguments.config, arguments.set)
        report = validate_experiment(config)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["status"] == "pass" else 1)
    if arguments.command == "data" and arguments.data_command == "validate":
        config = load_config(arguments.config)
        rows = []
        for split in ("train", "test"):
            for path in config["data"].get(split, []):
                trajectory = (
                    FilmTrajectory.load(path, dt=config["data"].get("dt", .002))
                    if config.get("model", {}).get("class") == "model_a_film"
                    else load_trajectory(path, mmap=True)
                )
                field = trajectory.field
                rows.append(
                    {
                        "split": split,
                        "path": str(trajectory.source), "shape": list(field.shape),
                        "time_start": float(trajectory.times[0]),
                        "time_end": float(trajectory.times[-1]),
                    }
                )
        print(json.dumps(rows, indent=2))
        return
    if arguments.command == "data" and arguments.data_command == "generate":
        path = Path(arguments.config).resolve()
        config = load_config(path)
        load_extensions(config.get("extensions"), config.get("_config_dir"))
        output = Path(config.pop("output_dir"))
        if not output.is_absolute():
            output = path.parent / output
        generator = get_data_generator(config.pop("generator"))
        config.pop("_config_path", None)
        config.pop("_config_dir", None)
        print(generator(config, output))
        return
    if arguments.command == "data" and arguments.data_command == "convert-legacy-ising":
        print(
            convert_legacy_ising_trajectory(
                arguments.source,
                arguments.output,
                mode=arguments.mode,
            )
        )
        return
    if arguments.command == "data" and arguments.data_command == "coarse-grain-vampire":
        print(
            coarse_grain_vampire(
                arguments.input_root,
                arguments.output_dir,
                block_size=arguments.block_size,
                coarse_size=arguments.coarse_size,
                batch_size=arguments.batch_size,
                overwrite=arguments.overwrite,
            )
        )
        return
    if arguments.command == "data" and arguments.data_command == "coarse-grain-crbr3":
        print(
            coarse_grain_crbr3(
                arguments.input_root,
                arguments.output_dir,
                coords_name=arguments.coords_name,
                stride_A=tuple(arguments.stride_A),
                block_A=tuple(arguments.block_A) if arguments.block_A else None,
                z_group=arguments.z_group,
                frame_stride=arguments.frame_stride,
                max_frames=arguments.max_frames,
                time_unit=arguments.time_unit,
                overwrite=arguments.overwrite,
            )
        )
        return
    if arguments.command == "train":
        config = _load_with_overrides(arguments.config, arguments.set)
        procedure = TrainingProcedure(config["training"]["procedure"])
        context = RunContext(
            config,
            config["run"]["output_dir"],
            resume=arguments.stage is not None,
        )
        only = None if arguments.stage is None else [arguments.stage]
        procedure.run(context, only=only)
        return
    if arguments.command == "infer":
        config = _load_with_overrides(arguments.config, arguments.set)
        if config["model"]["class"] == "model_a_film":
            factor = _artifact_for_trainers(config, ("film_finite_range_factor",), "factor_last.pt")
            thermodynamics = _artifact_for_trainers(
                config, ("film_free_energy_interaction",), "thermodynamics_last.pt"
            )
            output = Path(config["run"]["output_dir"]) / "inference"
            physics = config["physics"]
            print(rollout_film(
                config["data"]["test"][0], factor, thermodynamics, output,
                steps=int(config["inference"].get("steps", 125)),
                dt=float(physics["dt"]), beta=float(physics["beta"]),
                coarse_volume=float(physics["coarse_volume"]),
                h=float(physics.get("external_field", 0.0)), device=config.get("device", "cuda"),
            ))
            return
        if config["model"]["class"] == "model_a_vector":
            factor = _artifact_for_trainers(
                config, ("vector_finite_range_factor",), "factor_last.pt"
            )
            thermodynamics = _artifact_for_trainers(
                config, ("vector_free_energy_interaction",), "thermodynamics_last.pt"
            )
            output = Path(config["run"]["output_dir"]) / "inference"
            print(rollout_vector(
                config["data"]["test"][0], factor, thermodynamics, output,
                steps=int(config["inference"].get("steps", 125)),
                device=config.get("device", "cuda"),
            ))
            return
        factor = _artifact_for_trainer(config, "finite_range_factor", "factor_last.pt")
        thermodynamics = _artifact_for_trainer(
            config, "free_energy_interaction", "thermodynamics_last.pt"
        )
        output = Path(config["run"]["output_dir"]) / "inference"
        print(rollout(config, factor, thermodynamics, output))
        return
    if arguments.command == "evaluate":
        config = _load_with_overrides(arguments.config, arguments.set)
        print(evaluate_run(config))
        return
    if arguments.command == "report":
        config = _load_with_overrides(arguments.config, arguments.set)
        print(generate_run_report(config))
        return
    if arguments.command == "reproduce":
        config = _load_with_overrides(arguments.config, arguments.set)
        preflight = validate_experiment(config, require_fresh_output=True)
        print(json.dumps(preflight, indent=2))
        if preflight["status"] != "pass":
            raise SystemExit(1)
        procedure = TrainingProcedure(config["training"]["procedure"])
        context = RunContext(config, config["run"]["output_dir"])
        procedure.run(context)
        factor = _artifact_for_trainers(
            config, ("finite_range_factor", "vector_finite_range_factor"), "factor_last.pt"
        )
        if config["model"]["class"] == "model_a_film":
            thermodynamics = _artifact_for_trainers(
                config, ("film_free_energy_interaction",), "thermodynamics_last.pt"
            )
            physics = config["physics"]
            print(rollout_film(
                config["data"]["test"][0], factor, thermodynamics,
                Path(config["run"]["output_dir"]) / "inference",
                steps=int(config["inference"].get("steps", 125)),
                dt=float(physics["dt"]), beta=float(physics["beta"]),
                coarse_volume=float(physics["coarse_volume"]),
                h=float(physics.get("external_field", 0.0)), device=config.get("device", "cuda"),
            ))
            return
        if config["model"]["class"] == "model_a_vector":
            thermodynamics = _artifact_for_trainers(
                config, ("vector_free_energy_interaction",), "thermodynamics_last.pt"
            )
            print(rollout_vector(
                config["data"]["test"][0], factor, thermodynamics,
                Path(config["run"]["output_dir"]) / "inference",
                steps=int(config["inference"].get("steps", 125)),
                device=config.get("device", "cuda"),
            ))
            return
        thermodynamics = _artifact_for_trainer(
            config, "free_energy_interaction", "thermodynamics_last.pt"
        )
        rollout(
            config,
            factor,
            thermodynamics,
            Path(config["run"]["output_dir"]) / "inference",
        )
        evaluate_run(config)
        print(generate_run_report(config))
        return
    if arguments.command == "plot":
        print(plot_figure2(arguments.manifest))


if __name__ == "__main__":
    main()
