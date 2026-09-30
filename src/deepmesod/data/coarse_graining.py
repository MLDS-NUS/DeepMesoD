"""Convert VAMPIRE spin snapshots into the DeepMesoD trajectory contract.

The original 2D-ferromagnet preprocessing reads either floating-point
``*_snaps.npy`` files or bit-packed ``*_snaps.bin`` files, applies a periodic
``B x B`` block mean on an ``M x M`` coarse grid, and writes a trajectory. This
module keeps that behavior while removing the old repository-local imports.

The public output is canonical DeepMesoD data:

* scalar snapshots: ``(time, height, width)``;
* vector snapshots: ``(time, components, height, width)``;
* ``times.npy`` and ``metadata.json`` alongside ``field.npy``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .crbr3_preprocessing import VampireRun, build_lattice_map


def convert_legacy_ising_trajectory(source, output, *, mode="symlink"):
    """Convert the legacy Ising ``m.npy`` layout to the public trajectory format."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    field, times, metadata_path = source / "m.npy", source / "times.npy", source / "meta.json"
    missing = [path for path in (field, times, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"legacy trajectory is missing: {missing}")
    if output.exists():
        raise FileExistsError(f"conversion output already exists: {output}")
    legacy = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = legacy.get("dataset_meta", {})
    block = int(legacy.get("B", dataset.get("block")))
    output.mkdir(parents=True)
    for source_path, name in ((field, "field.npy"), (times, "times.npy")):
        target = output / name
        if mode == "symlink":
            target.symlink_to(source_path)
        elif mode == "copy":
            shutil.copy2(source_path, target)
        else:
            raise ValueError("conversion mode must be 'symlink' or 'copy'")
    metadata = {
        "boundary": "periodic", "temperature": float(dataset["T"]),
        "coarse_volume": float(block * block), "external_field": float(dataset.get("h", 0.0)),
        "source_format": "legacy_ising_m", "source": str(source),
        "legacy_shape": legacy.get("shape"), "block": block,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def _periodic_indices(length, coarse_size, block_size):
    if length % coarse_size:
        raise ValueError(
            f"coarse_size={coarse_size} must divide fine-grid length={length}"
        )
    if block_size < 1:
        raise ValueError("block_size must be positive")
    stride = length // coarse_size
    half = block_size // 2
    centers = np.arange(coarse_size, dtype=np.int64) * stride
    offsets = np.arange(-half, block_size - half, dtype=np.int64)
    return (centers[:, None] + offsets[None, :]) % length


def periodic_block_mean(snapshots, block_size, coarse_size):
    """Compute periodic block means for scalar or vector snapshots.

    ``snapshots`` must have shape ``(T,L,L)``, ``(T,L,L,C)`` or
    ``(T,L,L,Lz,C)``. The two spatial axes are deliberately kept in their
    original positions so VAMPIRE's vector and slab layouts are supported.
    """
    values = np.asarray(snapshots)
    if values.ndim not in (3, 4, 5):
        raise ValueError("snapshots must have 3, 4, or 5 dimensions")
    if values.shape[1] != values.shape[2]:
        raise ValueError("snapshots must contain square LxL frames")
    rows = _periodic_indices(values.shape[1], coarse_size, block_size)
    columns = _periodic_indices(values.shape[2], coarse_size, block_size)

    # Average x first and y second. This avoids materializing a
    # (T,M,B,M,B,...) array, which is too large for the 1024x1024 Vampire data.
    horizontal = np.take(values, columns, axis=2).mean(axis=3, dtype=np.float32)
    coarse = np.take(horizontal, rows, axis=1).mean(axis=2, dtype=np.float32)
    return np.asarray(coarse, dtype=np.float32)


def _canonical_field(coarse):
    """Convert ``(T,M,M,...)`` output into the public field layout."""
    if coarse.ndim == 3:
        return coarse
    if coarse.ndim == 4:
        return np.ascontiguousarray(np.transpose(coarse, (0, 3, 1, 2)))
    # The current 2D-ferromagnet is a one-layer film. Rejecting a non-singleton
    # z axis keeps the core trajectory contract unambiguous for this pipeline.
    if coarse.shape[3] != 1:
        raise ValueError("2D-ferromagnet coarse-graining requires a singleton z axis")
    return np.ascontiguousarray(np.transpose(coarse[:, :, :, 0, :], (0, 3, 1, 2)))


def _nested_value(metadata, *keys, default=None):
    for key in keys:
        if key in metadata:
            return metadata[key]
        nested = metadata.get("dataset_meta", {})
        if key in nested:
            return nested[key]
    return default


def _trajectory_metadata(source_metadata, snapshots, block_size, coarse_size, source):
    z_layers = snapshots.shape[3] if snapshots.ndim == 5 else 1
    result = {
        "schema_version": 1,
        "system": "2D-ferromagnet",
        "simulator": "Vampire",
        "source_format": "vampire_spin_snapshots",
        "source": str(Path(source).expanduser().resolve()),
        "boundary": "periodic",
        "fine_shape": list(snapshots.shape[1:]),
        "coarse_shape": [coarse_size, coarse_size],
        "block_size": int(block_size),
        "coarse_volume": float(block_size * block_size * z_layers),
        "components": (list(_nested_value(source_metadata, "components", default=[]))
                        or (int(snapshots.shape[-1]) if snapshots.ndim >= 4 else 1)),
        "source_metadata": source_metadata,
    }
    beta = _nested_value(source_metadata, "beta")
    temperature = _nested_value(source_metadata, "T", "temperature")
    external_field = _nested_value(source_metadata, "h", "external_field", default=0.0)
    if beta is not None:
        result["beta"] = float(beta)
    if temperature is not None:
        result["temperature"] = float(temperature)
    result["external_field"] = float(external_field)
    return result


def write_coarse_trajectory(field, times, metadata, output_dir, *, overwrite=False):
    """Write one canonical coarse-grained trajectory."""
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    field = np.asarray(field, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    if field.shape[0] != times.shape[0]:
        raise ValueError("field and times must have the same time dimension")
    if field.shape[0] < 2:
        raise ValueError("a trajectory must contain at least two frames")
    np.save(output_dir / "field.npy", field)
    np.save(output_dir / "times.npy", times)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def _load_float_snapshots(input_root):
    input_root = Path(input_root)
    candidates = sorted(input_root.glob("*_snaps.npy"))
    if not candidates:
        return []
    loaded = []
    for snaps_path in candidates:
        stem = snaps_path.name[: -len("_snaps.npy")]
        times_path = input_root / f"{stem}_times.npy"
        metadata_path = input_root / f"{stem}_meta.json"
        if not times_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"{snaps_path} requires {times_path.name} and {metadata_path.name}"
            )
        snapshots = np.load(snaps_path, mmap_mode="r")
        times = np.asarray(np.load(times_path), dtype=np.float64)
        if len(times) < len(snapshots):
            raise ValueError(f"{times_path} has fewer times than {snaps_path} snapshots")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        loaded.append((snapshots, times[: len(snapshots)], metadata, snaps_path))
    return loaded


def _load_bitpacked_snapshots(input_root):
    input_root = Path(input_root)
    candidates = sorted(input_root.glob("*_meta.json"))
    if not candidates:
        return []
    loaded = []
    for metadata_path in candidates:
        stem = metadata_path.name[: -len("_meta.json")]
        root = input_root / stem
        snapshots_path = Path(f"{root}_snaps.bin")
        times_path = Path(f"{root}_times.npy")
        if not snapshots_path.is_file() or not times_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        n_snap = int(metadata["n_snap"])
        length = int(metadata["L"])
        times = np.asarray(np.load(times_path), dtype=np.float64)[:n_snap]
        if len(times) < n_snap:
            raise ValueError(f"{times_path} has fewer times than {snapshots_path}")
        n_bytes = (length * length + 7) // 8
        packed = np.memmap(
            snapshots_path, dtype=np.uint8, mode="r", shape=(n_snap, n_bytes)
        )
        snapshots = np.unpackbits(np.asarray(packed), axis=1)[:, : length * length]
        snapshots = snapshots.reshape(n_snap, length, length).astype(np.float32)
        snapshots = snapshots * 2.0 - 1.0
        loaded.append((snapshots, times, metadata, metadata_path))
    return loaded


def _input_parts(input_root):
    loaded = _load_float_snapshots(input_root)
    return loaded or _load_bitpacked_snapshots(input_root)


def _coarse_grain_one(input_root, output_dir, block_size, coarse_size,
                      batch_size, overwrite):
    parts = _input_parts(input_root)
    if not parts:
        raise FileNotFoundError(
            f"no VAMPIRE *_snaps.npy or *_meta.json input found in {input_root}"
        )
    coarse_parts = []
    time_parts = []
    source_metadata = None
    source_path = None
    fine_shape = None
    for snapshots, times, metadata, source in parts:
        if fine_shape is None:
            fine_shape = snapshots.shape[1:]
            source_metadata = metadata
            source_path = source
        elif snapshots.shape[1:] != fine_shape:
            raise ValueError("all VAMPIRE chunks must have the same snapshot shape")
        chunks = []
        for start in range(0, len(snapshots), batch_size):
            chunks.append(
                periodic_block_mean(
                    snapshots[start : start + batch_size], block_size, coarse_size
                )
            )
        coarse_parts.append(np.concatenate(chunks, axis=0))
        time_parts.append(times)
    coarse = np.concatenate(coarse_parts, axis=0)
    times = np.concatenate(time_parts, axis=0)
    field = _canonical_field(coarse)
    metadata = _trajectory_metadata(
        source_metadata, np.empty((1, *fine_shape)), block_size, coarse_size, source_path
    )
    return write_coarse_trajectory(field, times, metadata, output_dir, overwrite=overwrite)


def coarse_grain_vampire(
    input_root,
    output_dir,
    *,
    block_size=16,
    coarse_size=64,
    batch_size=10,
    overwrite=False,
):
    """Convert one VAMPIRE snapshot directory into a standard trajectory.

    The input may contain a floating-point ``*_snaps.npy`` file or a
    bit-packed ``*_snaps.bin`` file with matching time/metadata sidecars. Data
    are processed in batches for the floating-point path.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    input_root = Path(input_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    round_dirs = sorted(
        path for path in input_root.glob("round*") if path.is_dir()
    )
    if round_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for round_dir in round_dirs:
            _coarse_grain_one(
                round_dir, output_dir / round_dir.name, block_size, coarse_size,
                batch_size, overwrite,
            )
        return output_dir
    return _coarse_grain_one(
        input_root, output_dir, block_size, coarse_size, batch_size, overwrite
    )


# ---------------------------------------------------------------------------
# CrBr3 absolute-position coarse graining
# ---------------------------------------------------------------------------

def coarse_grain_crbr3_frame(spins: np.ndarray, geometry: dict) -> np.ndarray:
    """Aggregate one CrBr3 atomistic frame into ``(My, Mx, Lz, 3)``."""
    spins = np.asarray(spins, dtype=np.float32)
    flat = geometry["flat_index"]
    sums = np.stack(
        [np.bincount(flat, weights=spins[:, c], minlength=geometry["n_bins"]) for c in range(3)],
        axis=-1,
    ).astype(np.float32)
    sums /= geometry["fine_counts"][:, None].astype(np.float32)
    frame = sums.reshape(geometry["Lz"], geometry["My"], geometry["Mx"], 3).transpose(1, 2, 0, 3)
    kx, ky = geometry["block_over_stride"]
    if (kx, ky) != (1, 1):
        averaged = np.zeros_like(frame)
        for dy in range(ky):
            for dx in range(kx):
                averaged += np.roll(frame, (dy - ky // 2, dx - kx // 2), axis=(0, 1))
        frame = averaged / float(kx * ky)
    return frame


def _crbr3_times(run: VampireRun, frames: list[Path], unit: str) -> np.ndarray:
    seconds = np.asarray([run.time_seconds(frame) for frame in frames], dtype=np.float64)
    if unit == "ns":
        return seconds * 1e9
    if unit == "s":
        return seconds
    if unit == "index":
        return np.arange(len(frames), dtype=np.float64)
    raise ValueError("time_unit must be one of: ns, s, index")


def coarse_grain_crbr3(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    coords_name: str = "atoms-coords.data",
    stride_A: tuple[float, float] = (50.0, 50.0),
    block_A: tuple[float, float] | None = None,
    z_group: int = 3,
    frame_stride: int = 1,
    max_frames: int | None = None,
    time_unit: str = "ns",
    overwrite: bool = False,
) -> Path:
    """Convert one CrBr3 VAMPIRE seed into the film trajectory contract."""
    run = VampireRun.open(input_root, coords_name)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    coords = run.coords
    geometry = build_lattice_map(coords, stride_A=stride_A, block_A=block_A, z_group=z_group)
    frames = run.frames(frame_stride, max_frames)
    field = np.stack([
        coarse_grain_crbr3_frame(run.load_frame(frame, len(coords)), geometry)
        for frame in frames
    ])
    times = _crbr3_times(run, frames, time_unit)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "m.npy", field.astype(np.float32))
    counts = geometry["counts"].reshape(geometry["Lz"], geometry["My"], geometry["Mx"])
    np.save(output / "counts.npy", counts.transpose(1, 2, 0).astype(np.int32))
    np.save(output / "times.npy", times)
    np.save(output / "z_layers.npy", geometry["z_layers"])
    metadata = {
        "schema_version": 1, "system": "CrBr3", "simulator": "Vampire",
        "source": str(run.root), "source_format": "vampire_atom_spin_text",
        "mapping": "absolute_position_lattice_commensurate", "shape": list(field.shape),
        "dims": "(T, My, Mx, Lz, 3)", "num_frames": len(frames), "n_atoms": len(coords),
        "z_group": z_group, "z_groups": geometry["z_groups"],
        "n_below_groups": geometry["n_below_groups"], "periodic_xy": True,
        "time_unit": time_unit, "coarse_volume": float(geometry["counts"].flat[0]),
        "lattice": geometry["lattice"],
        "count_stats": {"min": int(geometry["counts"].min()), "max": int(geometry["counts"].max()),
                         "mean": float(geometry["counts"].mean()), "std": float(geometry["counts"].std())},
    }
    (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


__all__ = [
    "convert_legacy_ising_trajectory", "coarse_grain_crbr3", "coarse_grain_crbr3_frame", "coarse_grain_vampire",
    "periodic_block_mean", "write_coarse_trajectory",
]
