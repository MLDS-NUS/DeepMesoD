"""CrBr3 preprocessing: VAMPIRE I/O and absolute-position geometry.

This module stops before coarse-graining.  It discovers atomistic frames,
loads absolute coordinates, reconstructs the honeycomb lattice, and builds the
atom-to-cell map consumed by the CrBr3 entry point in
:mod:`deepmesod.data.coarse_graining`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# VAMPIRE input boundary
# ---------------------------------------------------------------------------

def load_coords(path: str | Path) -> np.ndarray:
    """Load ``atoms-coords.data`` as ``(material, category, x, y, z)``."""
    values = np.loadtxt(Path(path), skiprows=1, ndmin=2, dtype=np.float64)
    if values.shape[1] < 5:
        raise ValueError(f"expected at least five coordinate columns, got {values.shape}")
    return values[:, :5]


def load_spins(path: str | Path, n_atoms: int) -> np.ndarray:
    """Load one ``spins-*.data`` frame as ``(N, 3)``."""
    values = np.loadtxt(Path(path), skiprows=1, ndmin=2, dtype=np.float32)
    if values.shape[0] != n_atoms or values.shape[1] < 3:
        raise ValueError(f"{Path(path).name}: expected ({n_atoms}, >=3), got {values.shape}")
    return values[:, :3]


def read_time(path: str | Path) -> float:
    """Read VAMPIRE's ``Time:`` value, normally expressed in seconds."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("Time:"):
            return float(line.split()[1])
    return float("nan")


@dataclass(frozen=True)
class VampireRun:
    """Discover and load the atomistic files belonging to one seed/run."""

    root: Path
    coords_name: str = "atoms-coords.data"

    @classmethod
    def open(cls, root: str | Path, coords_name: str = "atoms-coords.data") -> "VampireRun":
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if not (root / coords_name).is_file():
            raise FileNotFoundError(root / coords_name)
        return cls(root, coords_name)

    @property
    def coords(self) -> np.ndarray:
        return load_coords(self.root / self.coords_name)

    def frames(self, stride: int = 1, maximum: int | None = None) -> list[Path]:
        if stride < 1:
            raise ValueError("frame stride must be >= 1")
        paths = sorted(self.root.glob("spins-*.data"))[::stride]
        if maximum is not None:
            paths = paths[:maximum]
        if not paths:
            raise FileNotFoundError(f"no spins-*.data frames under {self.root}")
        return paths

    def load_frame(self, path: str | Path, n_atoms: int) -> np.ndarray:
        path = Path(path)
        return load_spins(path if path.is_absolute() else self.root / path, n_atoms)

    def time_seconds(self, frame: str | Path) -> float:
        path = Path(frame)
        return read_time(self.root / f"{path.stem}.meta")


# ---------------------------------------------------------------------------
# Absolute-position honeycomb geometry
# ---------------------------------------------------------------------------

def z_groups(n_layers: int, layers_per_slab: int) -> tuple[list[list[int]], int]:
    """Group layers outward from the mid-plane stacking fault."""
    if layers_per_slab < 1:
        raise ValueError("z_group must be >= 1")
    if n_layers % 2:
        raise ValueError(f"stacking-fault grouping requires even layer count, got {n_layers}")
    half = n_layers // 2
    lower = list(range(half))[::-1]
    upper = list(range(half, n_layers))
    below = [sorted(lower[i : i + layers_per_slab]) for i in range(0, half, layers_per_slab)][::-1]
    above = [sorted(upper[i : i + layers_per_slab]) for i in range(0, half, layers_per_slab)]
    return below + above, len(below)


def assign_z_slabs(z: np.ndarray, layers_per_slab: int) -> dict:
    unique = np.unique(np.round(np.asarray(z), 3)).astype(np.float32)
    if unique.size == 1:
        layers = np.zeros(len(z), dtype=np.int64)
    else:
        right = np.clip(np.searchsorted(unique, z), 1, unique.size - 1)
        layers = np.where(
            np.abs(z - unique[right - 1]) <= np.abs(z - unique[right]), right - 1, right
        ).astype(np.int64)
    groups, below_count = z_groups(int(unique.size), layers_per_slab)
    layer_to_slab = np.empty(unique.size, dtype=np.int64)
    for slab, layer_group in enumerate(groups):
        layer_to_slab[layer_group] = slab
    return {
        "slab_index": layer_to_slab[layers],
        "z_layers": np.asarray([unique[group].mean() for group in groups], dtype=np.float32),
        "z_groups": groups,
        "n_below_groups": below_count,
    }


def detect_axis(values: np.ndarray, decimals: int = 3) -> dict:
    """Infer basis multiplicity and repeat length from absolute coordinates."""
    unique = np.unique(np.round(np.asarray(values), decimals))
    for basis_count in range(1, unique.size // 2 + 1):
        if unique.size % basis_count:
            continue
        gaps = unique[basis_count:] - unique[:-basis_count]
        if gaps.size and np.allclose(gaps, gaps[0], atol=1e-2):
            return {
                "values": unique,
                "basis_count": basis_count,
                "repeat_count": unique.size // basis_count,
                "period_A": float(gaps[0]),
            }
    raise ValueError("could not detect a periodic lattice repeat from coordinates")


def snap_stride(target_A: float, period_A: float, repeat_count: int) -> int:
    divisors = [d for d in range(1, repeat_count + 1) if repeat_count % d == 0]
    return min(divisors, key=lambda d: abs(d * period_A - target_A))


def snap_block(target_A: float, period_A: float, stride: int, repeat_count: int) -> int:
    maximum = max(1, repeat_count // stride)
    multiple = min(range(1, maximum + 1), key=lambda k: abs(k * stride * period_A - target_A))
    return multiple * stride


def build_lattice_map(
    coords: np.ndarray,
    *,
    stride_A: tuple[float, float] = (50.0, 50.0),
    block_A: tuple[float, float] | None = None,
    z_group: int = 3,
) -> dict:
    """Map atoms to lattice-commensurate cells using absolute positions."""
    x, y, z = coords[:, 2], coords[:, 3], coords[:, 4]
    axis_x, axis_y = detect_axis(x), detect_axis(y)
    sx = snap_stride(stride_A[0], axis_x["period_A"], axis_x["repeat_count"])
    sy = snap_stride(stride_A[1], axis_y["period_A"], axis_y["repeat_count"])
    requested_block = block_A or stride_A
    bx = snap_block(requested_block[0], axis_x["period_A"], sx, axis_x["repeat_count"])
    by = snap_block(requested_block[1], axis_y["period_A"], sy, axis_y["repeat_count"])
    nx, ny = axis_x["repeat_count"], axis_y["repeat_count"]
    if nx % sx or ny % sy or bx % sx or by % sy:
        raise ValueError("stride and block must tile the periodic lattice")
    ix_repeat = np.searchsorted(axis_x["values"], np.round(x, 3)) // axis_x["basis_count"]
    iy_repeat = np.searchsorted(axis_y["values"], np.round(y, 3)) // axis_y["basis_count"]
    mx, my = nx // sx, ny // sy
    zmap = assign_z_slabs(z, z_group)
    ix, iy = ix_repeat // sx, iy_repeat // sy
    flat = (zmap["slab_index"] * my + iy) * mx + ix
    n_bins = mx * my * len(zmap["z_layers"])
    fine_counts = np.bincount(flat, minlength=n_bins).astype(np.int32)
    block_factor = (bx // sx) * (by // sy)
    counts = fine_counts * block_factor
    if np.any(counts != counts.flat[0]):
        raise RuntimeError("absolute lattice mapping did not produce uniform atom counts")
    return {
        "flat_index": flat, "Mx": mx, "My": my, "Lz": len(zmap["z_layers"]),
        "n_bins": n_bins, "fine_counts": fine_counts, "counts": counts,
        "z_layers": zmap["z_layers"], "z_groups": zmap["z_groups"],
        "n_below_groups": zmap["n_below_groups"], "block_over_stride": (bx // sx, by // sy),
        "lattice": {
            "ax": axis_x["period_A"], "ay": axis_y["period_A"],
            "Nx_cells": nx, "Ny_cells": ny, "stride_cells": [sx, sy],
            "stride_A": [sx * axis_x["period_A"], sy * axis_y["period_A"]],
            "block_cells": [bx, by], "block_A": [bx * axis_x["period_A"], by * axis_y["period_A"]],
        },
    }


__all__ = [
    "VampireRun", "assign_z_slabs", "build_lattice_map", "detect_axis",
    "load_coords", "load_spins", "read_time", "snap_block", "snap_stride", "z_groups",
]
