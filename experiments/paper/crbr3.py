"""Run the CrBr3 paper evaluation workflow.

The reusable numerical functions live in ``deepmesod.evaluation``.  This file
only resolves the CrBr3 configuration, checkpoint convention, and output
layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepmesod.config import apply_overrides, load_config
from deepmesod.evaluation.free_energy import scan_two_block_energy
from deepmesod.evaluation.kernels import export_factorized_kernel, infer_factorized_layers, state_dict
from deepmesod.models import FilmPsiJModel


def load_crbr3_model(checkpoint, config, device="cpu", dtype=torch.float64):
    device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state_dict(payload)
    model_config = dict(payload.get("model_config") or config["model"]["thermodynamics"])
    model_config.setdefault("num_layers", infer_factorized_layers(state))
    model_config.setdefault("num_components", model_config.get("n_components", 3))
    model_config.setdefault("patch_size", model_config.get("kernel_size", 7))
    model_config.setdefault("mlp_hidden", [64, 64, 64])
    model = FilmPsiJModel(
        num_layers=int(model_config["num_layers"]),
        num_components=int(model_config["num_components"]),
        patch_size=int(model_config["patch_size"]),
        kernel_normalize_mode=model_config.get("kernel_normalize_mode", "sum"),
        jz_normalize_mode=model_config.get("jz_normalize_mode", "none"),
        mlp_hidden=model_config["mlp_hidden"],
        spin_flip_even=model_config.get("spin_flip_even", False),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state)
    return model.eval()


def evaluate(config, kind="all"):
    settings = config["evaluation"]
    checkpoint = settings["stage3_checkpoint"]
    result = {}
    if kind in ("kernels", "all"):
        result["kernels"] = export_factorized_kernel(
            checkpoint,
            settings["kernel_output_dir"],
            settings.get("normalize_wxy", "sum"),
            settings.get("layer_kernel_name", "Jz.csv"),
            settings.get("spatial_kernel_name", "w_xy.csv"),
        )
    if kind in ("free-energy", "all"):
        device = config.get("device", "cuda")
        dtype = torch.float64 if settings.get("dtype", "float64") == "float64" else torch.float32
        model = load_crbr3_model(checkpoint, config, device=device, dtype=dtype)
        result["free_energy"] = scan_two_block_energy(
            model,
            settings["free_energy_output_dir"],
            split_after_layer=int(settings.get("split_after_layer", 14)),
            num_mean=int(settings.get("num_mean", 401)),
            num_contrast=int(settings.get("num_ell", 401)),
            batch_size=int(settings.get("batch_size", 2048)),
            device=device,
            dtype=settings.get("dtype", "float64"),
            checkpoint=checkpoint,
            artifact_stem=settings.get("energy_output_stem", "two_block_free_energy_2d"),
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("kernels", "free-energy", "all"), default="all")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args.set)
    print(json.dumps(evaluate(config, args.kind), indent=2, default=str))


if __name__ == "__main__":
    main()
