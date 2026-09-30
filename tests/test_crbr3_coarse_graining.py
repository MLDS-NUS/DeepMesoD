import json

import numpy as np

from deepmesod.data import coarse_grain_crbr3


def _write_vampire_run(path):
    path.mkdir()
    # Two basis values per repeat on each in-plane axis; two atoms per z layer
    # would also work, but the full Cartesian product keeps occupancy obvious.
    xs, ys, zs = [0.0, 1.0, 3.0, 4.0], [0.0, 2.0, 5.0, 7.0], [0.0, 1.0, 2.0, 3.0]
    coords = np.array([(1, 1, x, y, z) for z in zs for y in ys for x in xs], dtype=float)
    with (path / "atoms-coords.data").open("w") as handle:
        handle.write("material category x y z\n")
        np.savetxt(handle, coords, fmt="%.3f")
    for frame, time in enumerate((0.0, 2e-12)):
        spins = np.zeros((len(coords), 3), dtype=float)
        spins[:, 2] = coords[:, 2] + 10 * coords[:, 3] + 100 * coords[:, 4] + frame
        with (path / f"spins-{frame:08d}.data").open("w") as handle:
            handle.write("sx sy sz\n")
            np.savetxt(handle, spins, fmt="%.6f")
        (path / f"spins-{frame:08d}.meta").write_text(f"Time: {time}\n")


def test_crbr3_absolute_lattice_converter(tmp_path):
    source = tmp_path / "seed1"
    output = tmp_path / "processed"
    _write_vampire_run(source)
    coarse_grain_crbr3(source, output, stride_A=(3, 5), z_group=2, time_unit="ns")
    field = np.load(output / "m.npy")
    counts = np.load(output / "counts.npy")
    metadata = json.loads((output / "meta.json").read_text())
    assert field.shape == (2, 2, 2, 2, 3)
    assert counts.shape == (2, 2, 2)
    assert np.all(counts == 8)
    assert np.allclose(np.load(output / "times.npy"), [0.0, 0.002])
    assert metadata["mapping"] == "absolute_position_lattice_commensurate"
    assert metadata["coarse_volume"] == 8.0
