import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def run(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def generator_config(output, seed):
    return {
        "generator": "ising_kac_tau_leaping",
        "output_dir": str(output),
        "size": 16,
        "block": 2,
        "steps": 3,
        "temperature": 1.5,
        "interaction_strength": 1.0,
        "external_field": 0.0,
        "epsilon": 0.125,
        "domain_length_x": 1.0,
        "domain_length_y": 1.0,
        "tau_fraction": 0.01,
        "seed": seed,
        "device": "cpu",
    }


def experiment_config(work):
    return {
        "run": {
            "name": "source-release-agent-self-test",
            "seed": 42,
            "output_dir": str(work / "private-results"),
        },
        "device": "cpu",
        "data": {
            "train": [str(work / "private-data" / "train")],
            "test": [str(work / "private-data" / "test")],
            "transitions_per_trajectory": 3,
        },
        "model": {
            "class": "model_a_scalar",
            "drift": {"patch_size": 7, "hidden": [4], "activation": "softplus"},
            "factor": {
                "patch_size": 7,
                "factor_radius": 1,
                "likelihood_radius": 1,
                "hidden_channels": 3,
                "layers": 3,
                "dilations": [1, 1, 1],
                "activation": "tanh",
                "parameterization": "amplitude_shape",
                "spin_flip_even": True,
                "d4_equivariant": False,
            },
            "thermodynamics": {
                "patch_size": 7,
                "hidden": [4],
                "activation": "softplus",
                "external_field": 0.0,
                "interaction_constraint": "centrosymmetric_sum_one",
            },
        },
        "training": {
            "procedure": [
                {
                    "name": "drift",
                    "trainer": "local_drift",
                    "loss": "forward_rate_mse",
                    "config": {
                        "sampling": "joint",
                        "balanced_bins": 4,
                        "batch_size": 4,
                        "epochs": 1,
                        "batches_per_epoch": 1,
                        "learning_rate": 0.001,
                        "decay_step": 10,
                        "decay_factor": 1.0,
                    },
                },
                {
                    "name": "dissipation",
                    "trainer": "finite_range_factor",
                    "loss": "window_gaussian_nll",
                    "requires": ["drift"],
                    "config": {
                        "sampling": "center",
                        "balanced_bins": 4,
                        "batch_size": 4,
                        "epochs": 1,
                        "batches_per_epoch": 1,
                        "learning_rate": 0.001,
                        "decay_step": 10,
                        "decay_factor": 1.0,
                        "jitter": 0.0001,
                    },
                },
                {
                    "name": "thermodynamics",
                    "trainer": "free_energy_interaction",
                    "loss": "direct_forward_rate",
                    "requires": ["dissipation"],
                    "config": {
                        "objective": "direct",
                        "sampling": "center",
                        "target_batch_size": 2,
                        "balanced_bins": 4,
                        "max_state_weight": 4,
                        "batch_size": 4,
                        "steps": 1,
                        "learning_rate": 0.001,
                        "decay_step": 10,
                        "decay_factor": 1.0,
                    },
                },
            ]
        },
        "inference": {"mode": "pde", "steps": 1, "compile": False},
        "report": {"max_operator_frames": 2, "state_bins": 4},
    }


def execute(work):
    work.mkdir(parents=True, exist_ok=False)
    for name, seed in (("train", 7), ("test", 8)):
        path = work / f"generate-{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                generator_config(work / "private-data" / name, seed),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        run(
            [sys.executable, "-m", "deepmesod.cli", "data", "generate", "--config", str(path)],
            work,
        )
    config = work / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(experiment_config(work), sort_keys=False), encoding="utf-8"
    )
    run(
        [sys.executable, "-m", "deepmesod.cli", "reproduce", "--config", str(config)],
        work,
    )
    report_path = work / "private-results" / "report" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = (
        report_path,
        report_path.with_name("report.md"),
        report_path.with_name("diagnostics.pdf"),
        report_path.with_name("diagnostics.png"),
    )
    if report["status"] != "pass" or not all(path.is_file() for path in required):
        raise RuntimeError("agent self-test did not produce a complete passing report")
    return {
        "status": "pass",
        "work_dir": str(work),
        "report": str(report_path),
        "rollout_mean_mae": report["rollout"]["mean_mae"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir")
    arguments = parser.parse_args()
    if arguments.work_dir:
        result = execute(Path(arguments.work_dir).resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="deepmesod-agent-") as temporary:
            result = execute(Path(temporary) / "rehearsal")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
