import os
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a mapping")
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return normalize_paths(config)


def resolve_path(config, value):
    path = Path(os.path.expandvars(value))
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def apply_overrides(config, overrides):
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"override must have KEY=VALUE form: {item}")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        final = int(parts[-1]) if isinstance(target, list) else parts[-1]
        target[final] = yaml.safe_load(raw)
    return normalize_paths(config)


def normalize_paths(config):
    extensions = []
    for entry in config.get("extensions", ()):
        if isinstance(entry, str) and (entry.endswith(".py") or entry.startswith(".")):
            extensions.append(str(resolve_path(config, entry).resolve()))
        else:
            extensions.append(entry)
    if extensions:
        config["extensions"] = extensions
    for split in ("train", "test"):
        if split in config.get("data", {}):
            config["data"][split] = [
                str(resolve_path(config, item).resolve()) for item in config["data"][split]
            ]
    if "input_dir" in config.get("simulation", {}):
        config["simulation"]["input_dir"] = str(
            resolve_path(config, config["simulation"]["input_dir"]).resolve()
        )
    if "output_dir" in config.get("run", {}):
        config["run"]["output_dir"] = str(
            resolve_path(config, config["run"]["output_dir"]).resolve()
        )
    for stage in config.get("artifacts", {}).values():
        for key, value in tuple(stage.items()):
            if isinstance(value, str):
                stage[key] = str(resolve_path(config, value).resolve())
    # Evaluation and visualization are allowed to share an experiment root,
    # while their reference export may live in the old repository. Normalize
    # only path-shaped keys so values such as ``device: cuda`` stay untouched.
    path_keys = {
        "data_root", "reference_data_root", "output_dir", "output_stem",
        "initial_frame", "stage3_checkpoint", "factor_checkpoint",
        "kernel_output_dir", "free_energy_output_dir", "kernel_dir",
        "free_energy_dir", "jz_jxy_output_stem", "free_energy_output_stem",
    }

    def normalize_nested_paths(value):
        if isinstance(value, dict):
            for key, item in tuple(value.items()):
                if key in path_keys and isinstance(item, str):
                    value[key] = str(resolve_path(config, item).resolve())
                else:
                    normalize_nested_paths(item)
        elif isinstance(value, list):
            for item in value:
                normalize_nested_paths(item)

    for section in ("evaluation", "visualization"):
        normalize_nested_paths(config.get(section, {}))
    return config
