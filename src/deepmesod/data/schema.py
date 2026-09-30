import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Trajectory:
    field: torch.Tensor
    times: torch.Tensor
    metadata: dict
    source: Path

    @property
    def transitions(self):
        dt = self.times[1:] - self.times[:-1]
        shape = (dt.shape[0],) + (1,) * (self.field.ndim - 1)
        rate = (self.field[1:] - self.field[:-1]) / dt.to(self.field.dtype).reshape(shape)
        return self.field[:-1], self.field[1:], dt, rate

    def first_transitions(self, count):
        if count < 1 or count >= self.field.shape[0]:
            raise ValueError(f"transition count must be in [1, {self.field.shape[0] - 1}]")
        return Trajectory(
            field=self.field[: count + 1],
            times=self.times[: count + 1],
            metadata=self.metadata,
            source=self.source,
        )


def inverse_temperature(metadata):
    if "beta" in metadata:
        return float(metadata["beta"])
    dataset = metadata.get("dataset_meta", {})
    if "beta" in dataset:
        return float(dataset["beta"])
    return 1 / float(metadata["temperature"])


def _normalize_field(array):
    if array.ndim < 3:
        raise ValueError(
            "field.npy must have shape (time, *space) or (time, channels, *space)"
        )
    values = np.asarray(array)
    # Legacy VAMPIRE files use (time, y, x, z, component), with z=1.  Accept
    # both that layout and the package layouts (time, component, y, x) and
    # (time, y, x, component), while retaining scalar (time, y, x) files.
    if values.ndim == 5:
        singleton = [axis for axis in range(1, values.ndim) if values.shape[axis] == 1]
        if len(singleton) != 1:
            raise ValueError(
                "5D field.npy must contain exactly one singleton spatial axis"
            )
        values = np.squeeze(values, axis=singleton[0])
    channels = None
    if values.ndim == 4:
        if values.shape[1] == 1:
            values = values[:, 0]
        elif values.shape[-1] == 1:
            values = values[..., 0]
        elif values.shape[1] in (2, 3):
            channels = int(values.shape[1])
        elif values.shape[-1] in (2, 3):
            values = np.transpose(values, (0, 3, 1, 2))
            channels = int(values.shape[1])
        else:
            raise ValueError(
                "4D field.npy must have components on axis 1 or the last axis"
            )
    if not values.flags.writeable:
        values = values.copy()
    return torch.from_numpy(values).float(), channels


def load_trajectory(path, *, mmap=False):
    path = Path(path)
    field_path = path / "field.npy"
    if not field_path.exists() and (path / "m.npy").exists():
        field_path = path / "m.npy"
    times_path = path / "times.npy"
    metadata_path = path / "metadata.json"
    if not metadata_path.exists() and (path / "meta.json").exists():
        metadata_path = path / "meta.json"
    missing = [
        item for item in (field_path, times_path, metadata_path) if not item.exists()
    ]
    if missing:
        raise FileNotFoundError(f"trajectory is missing: {[str(item) for item in missing]}")
    mode = "r" if mmap else None
    field, channels = _normalize_field(np.load(field_path, mmap_mode=mode))
    times = torch.from_numpy(np.asarray(np.load(times_path))).double()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Normalize the metadata shape used by the legacy VAMPIRE export.  This is
    # deliberately done at the I/O boundary so model and trainer code only
    # consumes the public trajectory contract.
    nested = metadata.get("dataset_meta", {})
    if "boundary" not in metadata:
        metadata["boundary"] = "periodic"
    if "coarse_volume" not in metadata and "B" in metadata:
        metadata["coarse_volume"] = float(metadata["B"]) ** 2
    if "coarse_volume" not in metadata and "B" in nested:
        metadata["coarse_volume"] = float(nested["B"]) ** 2
    if "beta" not in metadata and "beta" in nested:
        metadata["beta"] = float(nested["beta"])
    if "temperature" not in metadata and "T" in nested:
        metadata["temperature"] = float(nested["T"])
    if "external_field" not in metadata:
        metadata["external_field"] = float(
            nested.get("h", nested.get("h_field", 0.0))
        )
    validate_trajectory(field, times, metadata)
    if channels is not None:
        declared = metadata.get("channels")
        if declared is not None and int(declared) != channels:
            raise ValueError("metadata channel count does not match field.npy")
    return Trajectory(field=field, times=times, metadata=metadata, source=path)


def validate_trajectory(field, times, metadata):
    if field.shape[0] != times.shape[0]:
        raise ValueError("field and times must have the same time dimension")
    if field.shape[0] < 2:
        raise ValueError("a trajectory must contain at least two states")
    if not torch.isfinite(field).all() or not torch.isfinite(times).all():
        raise ValueError("trajectory contains non-finite values")
    if not torch.all(times[1:] > times[:-1]):
        raise ValueError("times must be strictly increasing")
    required = {"boundary", "coarse_volume"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"metadata is missing {sorted(missing)}")
    if metadata["boundary"] != "periodic":
        raise ValueError(
            "the built-in field operators currently require periodic boundaries"
        )
    if "beta" not in metadata and "temperature" not in metadata:
        raise ValueError("metadata must contain beta or temperature")


def load_trajectories(paths, transitions_per_trajectory):
    trajectories = []
    for path in paths:
        trajectory = load_trajectory(path)
        trajectories.append(trajectory.first_transitions(transitions_per_trajectory))
    return trajectories
