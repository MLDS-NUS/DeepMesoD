"""CrBr3 Figure 4 kernel and two-block free-energy rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deepmesod.config import apply_overrides, load_config


def plot_kernels(kernel_dir, output_stem, vlim=1.0):
    root = Path(kernel_dir)
    layer, spatial = np.loadtxt(root / "Jz.csv", delimiter=","), np.loadtxt(root / "w_xy.csv", delimiter=",")
    layers, size = layer.shape[0], spatial.shape[0]
    half, off = size // 2, np.diag(layer, k=1)
    fault = int(np.argmin(off)) if off.size else 0
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    image = axes[0].imshow(layer, cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="lower", interpolation="nearest")
    axes[0].axvline(fault + .5, color="gray", ls="--"); axes[0].axhline(fault + .5, color="gray", ls="--")
    axes[0].set(title=r"$J_z$", xlabel=r"layer $b$", ylabel=r"layer $a$", xticks=np.arange(0, layers, 8), yticks=np.arange(0, layers, 8))
    fig.colorbar(image, ax=axes[0], fraction=.046, pad=.04)
    image = axes[1].imshow(spatial, cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="lower",
                            extent=[-half - .5, half + .5, -half - .5, half + .5], interpolation="nearest")
    axes[1].scatter([0], [0], marker="+", color="black", s=100)
    axes[1].set(title=r"$J_{xy}$", xlabel=r"$\Delta x$", ylabel=r"$\Delta y")
    fig.colorbar(image, ax=axes[1], fraction=.046, pad=.04)
    index = np.arange(layers)
    axes[2].plot(index, np.diag(layer), marker="o", color="tab:red", label="diagonal")
    if off.size:
        axes[2].plot(index[:-1] + .5, off, marker="s", color="tab:blue", label="off-diagonal")
    axes[2].axhline(0, color="gray", ls="--"); axes[2].axvline(fault + .5, color="gray", ls="--")
    axes[2].set(title=r"$J_z$ diagonals", xlabel="layer", ylabel=r"$J_z$", xticks=np.arange(0, layers, 8))
    axes[2].legend()
    fig.tight_layout()
    stem = Path(output_stem).expanduser().resolve(); stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    return stem.with_suffix(".pdf")


def plot_free_energy(data_dir, output_stem):
    data = np.load(Path(data_dir) / "two_block_free_energy_2d.npz")
    mean_axis, contrast_axis = data["mean_axis"], data["contrast_axis"]
    energy, contrast_min = data["energy_minus_global_min"], data["contrast_min_at_fixed_mean"]
    finite = energy[np.isfinite(energy)]
    fig, axis = plt.subplots(figsize=(11.2, 8.8))
    axis.pcolormesh(mean_axis, contrast_axis, np.ma.masked_invalid(energy), shading="auto",
                    cmap="viridis", vmin=0, vmax=float(np.nanmax(finite)), rasterized=True)
    if np.any(finite > 0):
        levels = np.linspace(0, float(np.nanpercentile(finite, 95)), 9)[1:]
        axis.contour(mean_axis, contrast_axis, energy, levels=levels, colors="white", linewidths=.65, alpha=.62)
    axis.plot([-1, 0, 1, 0, -1], [0, 1, 0, -1, 0], color="black", lw=1.5)
    axis.axhline(0, color="white", ls="--", lw=2); axis.axvline(0, color="#ffcc66", ls=":", lw=2)
    axis.plot(mean_axis, contrast_min, color="#d73027", lw=2.5, label="minimum-energy path")
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal",
             xlabel="mean magnetization", ylabel="block contrast")
    axis.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False, fontsize=24)
    fig.subplots_adjust(left=.17, right=.97, bottom=.16, top=.84)
    stem = Path(output_stem).expanduser().resolve(); stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".pdf")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    settings = config["visualization"]
    result = {"kernels": plot_kernels(settings["kernel_dir"], settings["jz_jxy_output_stem"], settings.get("vlim", 1.0)),
              "free_energy": plot_free_energy(settings["free_energy_dir"], settings["free_energy_output_stem"])}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
