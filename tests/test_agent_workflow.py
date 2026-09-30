import json

import numpy as np
import pytest
import yaml

from deepmesod.cli import main
from deepmesod.config import load_config
from deepmesod.validation import validate_experiment


def write_trajectory(path, seed):
    generator = np.random.default_rng(seed)
    field = np.empty((4, 8, 8), dtype=np.float32)
    field[0] = generator.uniform(-0.4, 0.4, size=(8, 8))
    for index in range(1, field.shape[0]):
        field[index] = np.clip(
            field[index - 1] + 0.02 * generator.standard_normal((8, 8)), -1, 1
        )
    path.mkdir(parents=True)
    np.save(path / "field.npy", field)
    np.save(path / "times.npy", np.arange(4, dtype=np.float64) * 0.1)
    (path / "metadata.json").write_text(
        json.dumps({"boundary": "periodic", "beta": 2.0, "coarse_volume": 4.0}),
        encoding="utf-8",
    )


def workflow_config(tmp_path):
    train = tmp_path / "private-data" / "train"
    test = tmp_path / "private-data" / "test"
    write_trajectory(train, 1)
    write_trajectory(test, 2)
    return {
        "run": {
            "name": "agent-workflow-test",
            "seed": 42,
            "output_dir": str(tmp_path / "private-results"),
        },
        "device": "cpu",
        "data": {
            "train": [str(train)],
            "test": [str(test)],
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


def test_agent_can_validate_reproduce_and_report_private_data(tmp_path):
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(yaml.safe_dump(workflow_config(tmp_path)), encoding="utf-8")
    config = load_config(config_path)
    assert validate_experiment(config, require_fresh_output=True)["status"] == "pass"
    main(["reproduce", "--config", str(config_path)])
    output = tmp_path / "private-results"
    report = json.loads((output / "report" / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["interaction"]["unit_sum_declared"] is True
    assert (output / "report" / "diagnostics.pdf").is_file()
    assert (output / "report" / "diagnostics.png").is_file()
    assert (output / "report" / "report.md").is_file()
    with pytest.raises(SystemExit) as error:
        main(["reproduce", "--config", str(config_path)])
    assert error.value.code == 1


def test_preflight_rejects_unsupported_exact_divergence_architecture(tmp_path):
    config_path = tmp_path / "workflow.yaml"
    config = workflow_config(tmp_path)
    config["model"]["factor"]["layers"] = 2
    config["model"]["factor"]["dilations"] = [1, 1]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = validate_experiment(load_config(config_path))
    assert report["status"] == "fail"
    assert any("three convolution layers" in item for item in report["errors"])
