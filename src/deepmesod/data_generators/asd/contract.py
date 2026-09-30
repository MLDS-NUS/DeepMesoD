import json
from pathlib import Path

import numpy as np


def write_asd_trajectory(field, times, metadata, output_dir):
    field = np.asarray(field)
    times = np.asarray(times)
    if field.ndim != 4 or field.shape[1] != 3:
        raise ValueError("ASD field must have shape (time, 3, y, x)")
    if field.shape[0] != times.shape[0]:
        raise ValueError("field and times must have the same time dimension")
    metadata = {
        **metadata,
        "schema_version": 1,
        "system": "asd",
        "channels": 3,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "field.npy", field.astype(np.float32, copy=False))
    np.save(output_dir / "times.npy", times.astype(np.float64, copy=False))
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir
