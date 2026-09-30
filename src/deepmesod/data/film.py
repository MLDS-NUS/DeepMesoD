"""Data access for the CrBr3 coarse-grained film trajectories.

The archive stores fields as ``(T, My, Mx, Lz, C)`` and intentionally omits
``times.npy``.  DeepMesoD keeps the public tensor layout as
``(T, C, Lz, My, Mx)`` and uses the configured ``dt`` when no time file is
present.  This makes the nanosecond-unit training data unambiguous.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _finite_difference(field, times):
    result = torch.empty_like(field)
    result[0] = (field[1] - field[0]) / (times[1] - times[0])
    result[-1] = (field[-1] - field[-2]) / (times[-1] - times[-2])
    if field.shape[0] > 2:
        result[1:-1] = (field[2:] - field[:-2]) / (times[2:] - times[:-2]).reshape(-1, *([1] * (field.ndim - 1)))
    return result


@dataclass
class FilmTrajectory:
    """One film trajectory in canonical ``(T,C,Lz,H,W)`` layout."""

    field: torch.Tensor
    dmdt: torch.Tensor
    times: torch.Tensor
    metadata: dict
    source: Path

    @classmethod
    def load(cls, path, dt=0.002, start_time_idx=0, end_time_idx=None,
             device="cpu", dtype=torch.float32):
        path = Path(path)
        if path.is_file():
            path = path.parent
        m_path = path / "m.npy"
        if not m_path.is_file():
            raise FileNotFoundError(f"film trajectory is missing {m_path}")
        raw = np.load(m_path, mmap_mode="r")
        if raw.ndim != 5 or raw.shape[-1] != 3:
            raise ValueError(f"expected m.npy with shape (T,My,Mx,Lz,3), got {raw.shape}")
        # (T,My,Mx,Lz,C) -> (T,C,Lz,My,Mx)
        series = np.array(raw[start_time_idx:end_time_idx], dtype=np.float32, copy=True)
        field = torch.from_numpy(np.ascontiguousarray(series)).permute(0, 4, 3, 1, 2).contiguous()
        times_path = path / "times.npy"
        if times_path.exists():
            times_np = np.asarray(np.load(times_path), dtype=np.float32)[start_time_idx:end_time_idx]
        else:
            times_np = np.arange(field.shape[0], dtype=np.float32) * float(dt)
        if field.shape[0] < 2:
            raise ValueError("a film trajectory must contain at least two frames")
        times = torch.from_numpy(np.ascontiguousarray(times_np))
        field = field.to(device=device, dtype=dtype)
        times = times.to(device=device, dtype=dtype)
        dmdt = _finite_difference(field, times)
        metadata = {}
        meta_path = path / "meta.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.setdefault("boundary", "periodic_xy_nonperiodic_z")
        metadata.setdefault("coarse_volume", 480.0)
        metadata.setdefault("external_field", 0.0)
        metadata.setdefault("shape", list(raw.shape))
        return cls(field, dmdt, times, metadata, path.resolve())

    @property
    def T(self): return self.field.shape[0]
    @property
    def C(self): return self.field.shape[1]
    @property
    def Lz(self): return self.field.shape[2]
    @property
    def height(self): return self.field.shape[3]
    @property
    def width(self): return self.field.shape[4]


class FilmTrajectoryBank:
    """CPU-resident sampler for the three CrBr3 training stages."""

    def __init__(self, paths, dt=0.002, start_time_idx=0, end_time_idx=None,
                 seed=122, device="cpu"):
        self.generator = torch.Generator().manual_seed(seed)
        self.trajectories = [FilmTrajectory.load(p, dt, start_time_idx, end_time_idx, device)
                             for p in paths]
        first = self.trajectories[0]
        shape = (first.C, first.Lz, first.height, first.width)
        if any((t.C, t.Lz, t.height, t.width) != shape for t in self.trajectories):
            raise ValueError("all film trajectories must share (C,Lz,H,W)")
        self.n_components, self.n_layers, self.height, self.width = shape
        self.dt = float(dt)

    def _coordinates(self, count, radius, time_start=0, time_stop=None):
        eligible = []
        for ti, trajectory in enumerate(self.trajectories):
            stop = trajectory.T - 1 if time_stop is None else min(time_stop, trajectory.T - 1)
            eligible.append((ti, max(0, time_start), stop))
        picks = torch.randint(len(eligible), (count,), generator=self.generator)
        t = torch.empty(count, dtype=torch.long)
        y = torch.randint(self.height, (count,), generator=self.generator)
        x = torch.randint(self.width, (count,), generator=self.generator)
        traj = torch.empty(count, dtype=torch.long)
        for i, pick in enumerate(picks.tolist()):
            ti, lo, hi = eligible[pick]
            if hi <= lo:
                raise ValueError("time window has no valid one-step transitions")
            traj[i] = ti
            t[i] = torch.randint(lo, hi, (1,), generator=self.generator)
        return traj, t, y, x

    @staticmethod
    def _patch(field, t, y, x, radius):
        offsets = torch.arange(-radius, radius + 1, device=field.device)
        yy = (y[:, None] + offsets[None, :]) % field.shape[-2]
        xx = (x[:, None] + offsets[None, :]) % field.shape[-1]
        patch = field[t[:, None, None], :, :, yy[:, :, None], xx[:, None, :]]
        # Advanced indexing yields (batch, patch_y, patch_x, C, Lz).
        return patch.permute(0, 3, 4, 1, 2).contiguous()

    def sample_drift(self, count, radius, time_start=0, time_stop=None):
        traj, t, y, x = self._coordinates(count, radius, time_start, time_stop)
        patches = torch.empty(count, self.n_components, self.n_layers,
                              2 * radius + 1, 2 * radius + 1)
        targets = torch.empty(count, self.n_components, self.n_layers)
        dts = torch.empty(count)
        for ti in torch.unique(traj).tolist():
            sel = torch.nonzero(traj == ti, as_tuple=False).flatten()
            item = self.trajectories[ti]
            patches[sel] = self._patch(item.field, t[sel], y[sel], x[sel], radius).cpu()
            targets[sel] = item.dmdt[t[sel], :, :, y[sel], x[sel]].cpu()
            dts[sel] = (item.times[t[sel] + 1] - item.times[t[sel]]).cpu()
        return patches, targets, dts

    def sample_slab(self, count, factor_context_radius, drift_context_radius,
                    likelihood_radius, time_start=0, time_stop=None):
        traj, t, y, x = self._coordinates(count, drift_context_radius, time_start, time_stop)
        z = torch.randint(self.n_layers, (count,), generator=self.generator)
        factor_context = torch.empty(count, self.n_components, 2 * factor_context_radius + 1,
                                     2 * factor_context_radius + 1)
        drift_context = torch.empty(count, self.n_components, self.n_layers,
                                    2 * drift_context_radius + 1, 2 * drift_context_radius + 1)
        current = torch.empty(count, self.n_components, 2 * likelihood_radius + 1,
                              2 * likelihood_radius + 1)
        following = torch.empty_like(current)
        dts = torch.empty(count)
        for ti in torch.unique(traj).tolist():
            sel = torch.nonzero(traj == ti, as_tuple=False).flatten()
            item = self.trajectories[ti]
            slab = z[sel]
            fp = self._patch(item.field, t[sel], y[sel], x[sel], factor_context_radius)
            dp = self._patch(item.field, t[sel], y[sel], x[sel], drift_context_radius)
            cp = self._patch(item.field, t[sel], y[sel], x[sel], likelihood_radius)
            np_ = self._patch(item.field, t[sel] + 1, y[sel], x[sel], likelihood_radius)
            local = torch.arange(sel.numel())
            factor_context[sel] = fp[local, :, slab].cpu()
            drift_context[sel] = dp.cpu()
            current[sel] = cp[local, :, slab].cpu()
            following[sel] = np_[local, :, slab].cpu()
            dts[sel] = (item.times[t[sel] + 1] - item.times[t[sel]]).cpu()
        def flatten(value):
            return value.permute(0, 2, 3, 1).reshape(count, -1)
        return (factor_context, drift_context, flatten(current), flatten(following), dts, z)


def film_patch_grid(context, patch_size, num_components, num_layers=None):
    """Extract all in-plane patches from ``(B,C,Lz,H,W)`` contexts."""
    b, c, lz, _, _ = context.shape
    patches = F.unfold(context.reshape(b, c * lz, context.shape[-2], context.shape[-1]), patch_size)
    rows = patches.shape[-1]
    return patches.transpose(1, 2).reshape(b, rows, c, lz, patch_size, patch_size)
