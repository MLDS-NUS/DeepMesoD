import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .style import COLORS, apply_paper_style


def _band(axis, x, values, *, color, linestyle, label):
    values = np.asarray(values)
    mean = values.mean(0)
    standard_deviation = values.std(0)
    axis.plot(x, mean, color=color, linestyle=linestyle, label=label)
    axis.fill_between(
        x,
        mean - standard_deviation,
        mean + standard_deviation,
        color=color,
        alpha=0.15,
    )


def _trajectory_panels(error_axis, mean_axis, data):
    times = data["times"]
    _band(
        error_axis,
        times,
        data["mae_deepmesod"],
        color=COLORS["deepmesod"],
        linestyle="-",
        label="DeepMesoD",
    )
    error_axis.plot(
        times, data["mae_hydrodynamic"], color="black", linestyle="--", label="Hydrodynamic"
    )
    error_axis.plot(
        times,
        data["mae_allen_cahn"],
        color=COLORS["allen_cahn"],
        linestyle=":",
        label="Allen--Cahn",
    )
    _band(
        error_axis,
        times,
        data["mae_cnn"],
        color=COLORS["cnn"],
        linestyle="-.",
        label="CNN",
    )
    _band(
        mean_axis,
        times,
        data["mean_reference"],
        color=COLORS["reference"],
        linestyle="-",
        label="Microscopic",
    )
    _band(
        mean_axis,
        times,
        data["mean_deepmesod"],
        color=COLORS["deepmesod"],
        linestyle="-",
        label="DeepMesoD",
    )
    mean_axis.plot(
        times,
        data["mean_hydrodynamic"],
        color="black",
        linestyle="--",
        label="Hydrodynamic",
    )
    mean_axis.plot(
        times,
        data["mean_allen_cahn"],
        color=COLORS["allen_cahn"],
        linestyle=":",
        label="Allen--Cahn",
    )


def _energy_panel(axis, data):
    axis.plot(
        data["energy_state"],
        data["energy_reference"],
        color="black",
        linestyle="--",
        label="Hydro",
    )
    _band(
        axis,
        data["energy_state"],
        data["energy_deepmesod"],
        color=COLORS["deepmesod"],
        linestyle="-",
        label="DeepMesoD",
    )


def plot_figure2(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    apply_paper_style()
    figure, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    loaded = {}
    for temperature in (0.75, 1.5):
        path = Path(os.path.expandvars(manifest["plot_data"][str(temperature)]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        loaded[temperature] = np.load(path)
    _trajectory_panels(axes[0, 0], axes[0, 2], loaded[0.75])
    _trajectory_panels(axes[0, 1], axes[0, 3], loaded[1.5])
    _energy_panel(axes[1, 0], loaded[0.75])
    _energy_panel(axes[1, 1], loaded[1.5])
    for temperature, linestyle, marker in (
        (0.75, (0, (2.0, 1.4)), "o"),
        (1.5, (0, (5.0, 1.6, 1.2, 1.6)), "^"),
    ):
        data = loaded[temperature]
        axes[1, 2].plot(
            data["dissipation_state"],
            data["dissipation_reference"],
            color=COLORS["reference"],
            linestyle=linestyle,
            marker=marker,
            markevery=5,
            label=f"Microscopic T={temperature:g}",
        )
        _band(
            axes[1, 2],
            data["dissipation_state"],
            data["dissipation_deepmesod"],
            color=COLORS["deepmesod"],
            linestyle=linestyle,
            label=f"DeepMesoD T={temperature:g}",
        )
        _band(
            axes[1, 3],
            data["kernel_radius"],
            data["kernel_deepmesod"],
            color=COLORS["deepmesod"],
            linestyle=linestyle,
            label=f"DeepMesoD T={temperature:g}",
        )
    axes[1, 3].plot(
        loaded[0.75]["kernel_radius"],
        loaded[0.75]["kernel_reference"],
        color="black",
        linestyle="--",
        label="Hydro",
    )
    titles = (
        "Trajectory Error: T=0.75",
        "Trajectory Error: T=1.5",
        "Mean magnetization: T=0.75",
        "Mean magnetization: T=1.5",
        "Free-energy density: T=0.75",
        "Free-energy density: T=1.5",
        "Dissipation operator",
        "Interaction kernel",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontweight="bold")
    axes[0, 0].set_ylabel("MAE")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 2].set_ylabel(r"$\langle m\rangle(t)$")
    axes[0, 3].set_ylabel(r"$\langle m\rangle(t)$")
    axes[1, 0].set_ylabel(r"$g(m)$")
    axes[1, 1].set_ylabel(r"$g(m)$")
    axes[1, 2].set_ylabel("Effective dissipation")
    axes[1, 3].set_ylabel("Kernel weight")
    for axis in axes[0]:
        axis.set_xlabel("Time")
    axes[1, 0].set_xlabel("m")
    axes[1, 1].set_xlabel("m")
    axes[1, 2].set_xlabel("m")
    axes[1, 3].set_xlabel("Radial distance (grid cells)")
    for axis in axes.flat:
        axis.legend(frameon=False)
    for label, axis in zip("ABCDEFGH", axes.flat):
        axis.text(
            -0.16, 1.08, label, transform=axis.transAxes, fontsize=28, fontweight="bold"
        )
    output = manifest_path.parent / manifest["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    provenance = {
        "manifest": str(manifest_path),
        "plot_data": {
            str(key): str((manifest_path.parent / value).resolve())
            for key, value in manifest["plot_data"].items()
        },
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return output.with_suffix(".pdf")
