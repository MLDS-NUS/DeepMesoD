"""Stage-3 direct targets for the vector pipeline."""

import json
from pathlib import Path

import torch

from deepmesod.data import VectorTrajectory


def prepare_vector_stage3_targets(paths, factor, output_dir, start_time_idx=0,
                                  end_time_idx=None, coarse_volume=None,
                                  batch_size=1, device="cpu"):
    """Cache ``q = Div(K)/(beta V_c) - dm/dt`` without an ill-conditioned inverse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, path in enumerate(paths):
        trajectory = VectorTrajectory.load(path, start_time_idx, end_time_idx)
        beta = trajectory.beta
        volume = trajectory.coarse_volume if coarse_volume is None else coarse_volume
        fields, rhs = [], []
        for start in range(0, trajectory.steps, batch_size):
            stop = min(start + batch_size, trajectory.steps)
            current = trajectory.field[start:stop].to(device)
            following = trajectory.field[start + 1:stop + 1].to(device)
            dt = (trajectory.times[start + 1:stop + 1] - trajectory.times[start:stop]).to(device)
            divergence = factor.divergence_on_field_exact(current)
            rate = (following - current) / dt[:, None, None, None]
            fields.append(current.cpu())
            rhs.append((divergence / (beta * volume) - rate).cpu())
        target = output_dir / f"trajectory_{index:02d}.pt"
        torch.save({"fields": torch.cat(fields), "right_hand_sides": torch.cat(rhs),
                    "beta": beta, "coarse_volume": volume, "source": str(path),
                    "factor_checkpoint": str(getattr(factor, "checkpoint_path", ""))}, target)
        summaries.append({"path": str(target), "transitions": trajectory.steps})
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return [output_dir / f"trajectory_{i:02d}.pt" for i in range(len(paths))]
