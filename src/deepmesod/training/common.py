import json
from pathlib import Path

import torch

from deepmesod.classes import get_dynamical_class


def device_from_config(config):
    requested = config.get("device", "cuda")
    return torch.device(
        requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
    )


def dynamical_class(config):
    builder = get_dynamical_class(config["model"]["class"])
    return builder()


def save_history(path, history):
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def artifact_path(context, requirement, key):
    if requirement in context.artifacts:
        return Path(context.artifacts[requirement][key])
    configured = context.config.get("artifacts", {}).get(requirement, {})
    if key not in configured:
        raise KeyError(f"missing artifact '{requirement}.{key}'")
    return Path(configured[key])
