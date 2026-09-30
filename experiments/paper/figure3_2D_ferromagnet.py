"""Paper-specific Figure 3-A evaluation for ``2D-ferromagnet``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepmesod.config import apply_overrides, load_config
from deepmesod.evaluation.vector_rollout import generate_vector_rollout


def generate_figure3_rollout(stage3_checkpoint, factor_checkpoint, initial_frame, output_dir, **kwargs):
    """Keep the historical Figure 3 function name at the experiment boundary."""
    return generate_vector_rollout(
        stage3_checkpoint, factor_checkpoint, initial_frame, output_dir,
        rollout_name="rollout_deepmesoD",
        metadata={"experiment": "2D-ferromagnet", "stage3_objective": "direct"},
        **kwargs,
    )


def evaluate_from_config(config, *, device=None):
    settings = config["evaluation"]["figure3"]
    return generate_figure3_rollout(
        settings["stage3_checkpoint"], settings["factor_checkpoint"], settings["initial_frame"],
        settings["output_dir"], t_start=settings.get("t_start", 20.0),
        t_end=settings.get("t_end", 25000.0), dt=settings.get("dt", 1.0),
        divk_every=settings.get("divk_every", 50),
        jacobian_chunk_size=settings.get("jacobian_chunk_size", 1024),
        device=device or settings.get("device", config.get("device", "cuda")),
        progress_every=settings.get("progress_every", 2500),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    print(json.dumps(evaluate_from_config(config, device=args.device), indent=2, default=str))


if __name__ == "__main__":
    main()
