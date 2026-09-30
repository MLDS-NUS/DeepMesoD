"""Vector-field trajectory access used by the 2D ferromagnet pipeline."""

from dataclasses import dataclass

import numpy as np
import torch

from .schema import inverse_temperature, load_trajectory


def vector_square(field, time, row, column, radius):
    """Gather periodic vector patches as ``(batch, C, 2r+1, 2r+1)``."""
    time = torch.as_tensor(time, dtype=torch.long)
    row = torch.as_tensor(row, dtype=torch.long)
    column = torch.as_tensor(column, dtype=torch.long)
    offsets = torch.arange(-radius, radius + 1)
    rows = (row[:, None, None] + offsets[None, :, None]) % field.shape[-2]
    columns = (column[:, None, None] + offsets[None, None, :]) % field.shape[-1]
    return field[time[:, None, None], :, rows, columns].permute(0, 3, 1, 2)


def flatten_vector_window(window):
    """Flatten ``(batch, C, S, S)`` in site-major/component-minor order."""
    return window.permute(0, 2, 3, 1).reshape(window.shape[0], -1)


class MagnitudeBalancedSampler:
    """Sample uniformly from non-empty centre-magnitude bins."""

    def __init__(self, field, bins):
        if bins < 1:
            raise ValueError("bins must be positive")
        magnitude = field[:-1].norm(dim=1).reshape(-1).numpy()
        edges = np.linspace(float(magnitude.min()), float(magnitude.max()), bins + 1)
        index = np.clip(np.digitize(magnitude, edges) - 1, 0, bins - 1)
        self.order = torch.from_numpy(np.argsort(index, kind="stable")).long()
        counts = np.bincount(index, minlength=bins)
        self.starts = torch.from_numpy(np.concatenate(([0], np.cumsum(counts)))).long()
        self.valid_bins = torch.from_numpy(np.flatnonzero(counts > 0)).long()
        self.height, self.width = field.shape[-2:]

    def sample(self, count, generator):
        choices = torch.randint(
            self.valid_bins.numel(), (count,), generator=generator
        )
        bins = self.valid_bins[choices]
        starts = self.starts[bins]
        counts = self.starts[bins + 1] - starts
        offsets = torch.floor(
            torch.rand(count, generator=generator) * counts.float()
        ).long()
        flat = self.order[starts + offsets]
        plane = self.height * self.width
        return flat // plane, (flat % plane) // self.width, flat % self.width


@dataclass
class VectorTrajectory:
    field: torch.Tensor
    times: torch.Tensor
    metadata: dict
    source: object

    @classmethod
    def load(cls, path, start_time_idx=0, end_time_idx=None):
        trajectory = load_trajectory(path)
        field = trajectory.field
        if field.ndim != 4:
            raise ValueError(
                "vector trajectory must have shape (time, components, height, width)"
            )
        stop = field.shape[0] if end_time_idx is None else min(end_time_idx, field.shape[0])
        if not 0 <= start_time_idx < stop - 1:
            raise ValueError("time range must contain at least one transition")
        return cls(
            field=field[start_time_idx:stop],
            times=trajectory.times[start_time_idx:stop],
            metadata=trajectory.metadata,
            source=trajectory.source,
        )

    @property
    def steps(self):
        return self.field.shape[0] - 1

    @property
    def beta(self):
        return inverse_temperature(self.metadata)

    @property
    def coarse_volume(self):
        return float(self.metadata["coarse_volume"])

    @property
    def external_field(self):
        return float(self.metadata.get("external_field", 0.0))


class VectorTrajectoryBank:
    """A bank of vector trajectories with the pipeline's magnitude sampling."""

    def __init__(self, paths, start_time_idx=0, end_time_idx=None, bins=0, seed=42):
        self.generator = torch.Generator().manual_seed(seed)
        self.trajectories = [
            VectorTrajectory.load(path, start_time_idx, end_time_idx) for path in paths
        ]
        settings = {
            (t.field.shape[1:], round(t.beta, 12), t.coarse_volume, t.external_field)
            for t in self.trajectories
        }
        if len(settings) != 1:
            raise ValueError("all vector trajectories must share physical settings")
        first = self.trajectories[0]
        self.n_components, self.height, self.width = first.field.shape[1:]
        self.beta = first.beta
        self.coarse_volume = first.coarse_volume
        self.external_field = first.external_field
        self.samplers = [
            MagnitudeBalancedSampler(t.field, bins) if bins else None
            for t in self.trajectories
        ]

    def sample_window(self, batch_size, factor_context_radius, drift_context_radius,
                      window_radius):
        index = int(torch.randint(len(self.trajectories), (1,), generator=self.generator))
        trajectory = self.trajectories[index]
        sampler = self.samplers[index]
        if sampler is None:
            time = torch.randint(trajectory.steps, (batch_size,), generator=self.generator)
            row = torch.randint(self.height, (batch_size,), generator=self.generator)
            column = torch.randint(self.width, (batch_size,), generator=self.generator)
        else:
            time, row, column = sampler.sample(batch_size, self.generator)
        factor_context = vector_square(
            trajectory.field, time, row, column, factor_context_radius
        )
        drift_context = vector_square(
            trajectory.field, time, row, column, drift_context_radius
        )
        current = vector_square(trajectory.field, time, row, column, window_radius)
        following = vector_square(
            trajectory.field, time + 1, row, column, window_radius
        )
        dt = trajectory.times[time + 1] - trajectory.times[time]
        return (
            factor_context,
            drift_context,
            flatten_vector_window(current),
            flatten_vector_window(following),
            dt.float(),
        )
