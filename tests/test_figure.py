from pathlib import Path

import numpy as np
import yaml

from deepmesod.visualization import plot_figure2


def plot_payload(offset):
    time = np.linspace(0, 1, 6)
    state = np.linspace(-0.9, 0.9, 7)
    radius = np.arange(1, 5)
    seed_curve = np.stack([time + offset + value for value in (0, 0.01, -0.01)])
    state_curve = np.stack([state**2 + offset + value for value in (0, 0.01, -0.01)])
    kernel_curve = np.stack([np.exp(-radius) + value for value in (0, 0.001, -0.001)])
    return {
        "times": time,
        "mae_deepmesod": seed_curve,
        "mae_hydrodynamic": time + 0.03,
        "mae_allen_cahn": time + 0.05,
        "mae_cnn": seed_curve + 0.02,
        "mean_reference": seed_curve,
        "mean_deepmesod": seed_curve + 0.01,
        "mean_hydrodynamic": time + 0.02,
        "mean_allen_cahn": time + 0.04,
        "energy_state": state,
        "energy_reference": state**2,
        "energy_deepmesod": state_curve,
        "dissipation_state": state,
        "dissipation_reference": 1 - state**2,
        "dissipation_deepmesod": 1 - state_curve,
        "kernel_radius": radius,
        "kernel_reference": np.exp(-radius),
        "kernel_deepmesod": kernel_curve,
    }


def test_figure2_plotter_writes_vector_and_raster_outputs(tmp_path):
    for temperature, offset in ((0.75, 0.0), (1.5, 0.1)):
        np.savez(tmp_path / f"T{temperature}.npz", **plot_payload(offset))
    manifest = {
        "plot_data": {"0.75": "T0.75.npz", "1.5": "T1.5.npz"},
        "output": "generated/figure2",
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    output = plot_figure2(manifest_path)
    assert output == Path(tmp_path / "generated/figure2.pdf")
    assert output.exists()
    assert output.with_suffix(".png").exists()
    assert output.with_suffix(".provenance.json").exists()
