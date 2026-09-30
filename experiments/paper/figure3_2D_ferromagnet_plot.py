"""Reproduce the 2D-ferromagnet Figure 3 composite plot.

This is the DeepMesoD package version of
``MesoscopicModel/exports/fig3_deepmesoD/plot_composite_grid2x4_v15.py``.
The plot has one wide A column and six square B panels:

  A: mean ``m_z`` versus time, with observed/extrapolation split
  B: K spatial decay, K coupling constants, learned kernel,
     free-energy landscape, force along Jm_z, additive ansatz

The Figure 3-A rollout files are produced by the matching evaluation module.
The unchanged baseline/CNN/B-panel tables can be supplied from the old export
directory; this module never embeds those arrays in the DeepMesoD source tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter


TEXT_FS = 20
TITLE_FS = 24
TITLE_PAD = 17
AB_FS = 36
KERNEL_CMAP = "RdBu_r"
KERNEL_VLIM = 1.0
C_DATA = "#B8B8B8"
C_FNO = "#2ca02c"
C_CNN = "#FF7F0E"
C_MODEL = "#1f77b4"
C_DATA_LINE = "#008300"
SNAP_TIMES = [0, 100, 1000, 3000, 6000, 10000, 15000, 25000]
SNAP_LABELS = [
    r"$0$", r"$10^2$", r"$10^3$", r"$3\!\times\!10^3$",
    r"$6\!\times\!10^3$", r"$10^4$", r"$1.5\!\times\!10^4$",
    r"$2.5\!\times\!10^4$",
]
SNAP_NCOL, SNAP_NROW = 4, 2
SNAP_GAP_IN = 0.22
SNAP_CAP_IN = 0.34


def _paths(data_root: Path, reference_data_root: Path | None = None):
    reference_root = reference_data_root or data_root
    return {
        "macro_dir": reference_root / "data/baseline",
        "pg_dir": data_root / "data/rollout_deepmesoD",
        "km_dir": reference_root / "data/km_csv",
        "cnn_csv": reference_root / "data/rollout_cnn/macro_statistics_t20_25000.csv",
    }


def _load_snapshots(rollout_dir: Path, macro: pd.DataFrame):
    initial_mz = np.load(rollout_dir / "initial_frame_t20.npy")[..., 2]
    trajectory = np.load(rollout_dir / "pred_traj_t20_25000.npy", mmap_mode="r")
    times = macro["time"].values

    def frame(time):
        if time <= times[0]:
            return initial_mz
        index = min(int(np.searchsorted(times, time, side="left")), len(trajectory) - 1)
        return np.array(trajectory[index, :, :, 2])

    return [frame(time) for time in SNAP_TIMES]


def _load_km_data(km_dir: Path):
    kp = pd.read_csv(km_dir / "km_row2_col1_k_perp_1d.csv")
    kparallel = pd.read_csv(km_dir / "km_row2_col2_k_parallel_1d.csv")
    kernel = pd.read_csv(km_dir / "kernel_row1_col3_heatmap.csv")
    kernel = kernel.pivot_table(index="y", columns="x", values="kernel_value")
    decay = np.load(km_dir / "kdecay_row1_col1.npz")
    free_energy = pd.read_csv(km_dir / "free_energy_row2_col3_F.csv")
    force = pd.read_csv(km_dir / "free_energy_row2_col5_dF_dJmz.csv")
    free_energy = free_energy.pivot_table(index="Jm_z", columns="m_z", values="F")
    force = force.pivot_table(index="Jm_z", columns="m_z", values="dF_dJm_z")
    return {
        "kd_model_d": decay["model_distance"],
        "kd_model_pct": decay["model_pct"],
        "kd_data_d": decay["data_distance"],
        "kd_data_pct": decay["data_pct"],
        "m_center_z": kp["m_center_z"].values,
        "k_perp_1d": kp["K_perp"].values,
        "k_parallel_1d": kparallel["K_parallel"].values,
        "kernel": kernel.values,
        "kern_xs": np.asarray(kernel.columns, dtype=float),
        "kern_ys": np.asarray(kernel.index, dtype=float),
        "mz_vals": np.asarray(free_energy.columns, dtype=float),
        "jmz_vals": np.asarray(free_energy.index, dtype=float),
        "fe_grid": free_energy.values,
        "dfjmz_grid": force.values,
    }


def _box(axis, square=True):
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_linewidth(plt.rcParams["axes.linewidth"])
    if square:
        axis.set_box_aspect(1)
    axis.tick_params(labelsize=TEXT_FS)


def _legend(axis, **kwargs):
    axis.legend(fontsize=TEXT_FS, frameon=True, framealpha=0.9,
                edgecolor="0.8", handlelength=1.5, **kwargs)


def _attach_colorbar(figure, axis, mappable, vmin, vmax, width, gap):
    position = axis.get_position()
    colorbar_axis = figure.add_axes(
        [position.x1 + gap, position.y0, width, position.height]
    )
    colorbar = figure.colorbar(mappable, cax=colorbar_axis)
    ticks = [vmin] if np.isclose(vmin, vmax) else np.linspace(vmin, vmax, 3).tolist()
    colorbar.set_ticks(ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=TEXT_FS)
    return colorbar


def _era_marks(axis):
    axis.axvline(1000, color="gray", ls="--", lw=1.0, zorder=3)
    transform = axis.get_xaxis_transform()
    kwargs = dict(transform=transform, ha="center", va="top", fontsize=TEXT_FS,
                  color="gray", zorder=10,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1))
    axis.text(141, 0.99, "Observed", **kwargs)
    axis.text(5000, 0.99, "Extrapolation", **kwargs)


def _draw_mz(axis, macro, baseline, downsample, cnn_long):
    data_slice = slice(None, None, downsample)
    baseline_times = baseline["times"]
    axis.plot(baseline_times, baseline["data_mag_mz"], color=C_DATA, lw=6.2,
              alpha=0.95, label="Data")
    if "fno_mag_mz" in baseline:
        axis.plot(baseline_times, baseline["fno_mag_mz"], color=C_FNO,
                  lw=1.8, ls="-.", label="FNO")
    if cnn_long is not None:
        axis.plot(cnn_long["time"].values[data_slice], cnn_long["mz"].values[data_slice],
                  color=C_CNN, lw=1.8, ls="--", label="CNN", zorder=4)
    elif "patchcnn_mag_mz" in baseline:
        axis.plot(baseline_times, baseline["patchcnn_mag_mz"], color=C_CNN,
                  lw=1.8, ls="--", label="CNN")
    axis.plot(macro["time"].values[data_slice], macro["mz"].values[data_slice],
              color=C_MODEL, lw=1.8, label="DeepMesoD", zorder=5)
    axis.set_xscale("log")
    axis.set_xlim(float(macro["time"].values[0]), float(macro["time"].values[-1]))
    axis.set_ylim(-1, 1)
    _era_marks(axis)
    axis.set_ylabel(r"$m_z$", fontsize=TEXT_FS)
    axis.set_title(r"Mean magnetisation $m_z$", fontsize=TITLE_FS, pad=TITLE_PAD)
    axis.grid(alpha=0.3, linewidth=0.6, which="both")
    _legend(axis, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    _box(axis, square=False)


def _draw_kdecay(axis, km):
    axis.plot(km["kd_data_d"], km["kd_data_pct"], color=C_DATA_LINE, lw=2.4,
              marker="o", ms=5, label="Data")
    axis.plot(km["kd_model_d"], km["kd_model_pct"], color=C_MODEL, lw=2.0,
              ls="--", marker="o", ms=5, label="DeepMesoD")
    axis.set_ylim(-6, 132)
    axis.set_xlabel(r"Distance  $|d|$", fontsize=TEXT_FS)
    axis.set_ylabel(r"$\|K_{i,i+d}\|_F$  (%)", fontsize=TEXT_FS)
    axis.set_title("K spatial correlation", fontsize=TITLE_FS, pad=TITLE_PAD)
    axis.grid(alpha=0.3)
    _legend(axis, loc="upper right")
    _box(axis)


def _draw_coupling(axis, km):
    axis.plot(km["m_center_z"], km["k_perp_1d"], color="tab:orange", lw=2,
              label=r"$K_\perp$")
    axis.plot(km["m_center_z"], km["k_parallel_1d"], color=C_MODEL, lw=2,
              label=r"$K_\parallel$")
    axis.set_xlabel(r"Center spin $m_z$", fontsize=TEXT_FS)
    axis.set_ylabel("Coupling constant", fontsize=TEXT_FS)
    axis.set_title(r"Coupling constants: $K_\perp$ and $K_\parallel$",
                   fontsize=TITLE_FS, pad=TITLE_PAD)
    axis.grid(alpha=0.3)
    _legend(axis, loc="upper right")
    _box(axis)


def _draw_kernel(axis, km):
    half_range = float(KERNEL_VLIM)
    image = axis.imshow(km["kernel"], cmap=KERNEL_CMAP, origin="lower",
                        vmin=-half_range, vmax=half_range,
                        extent=[km["kern_xs"][0], km["kern_xs"][-1],
                                km["kern_ys"][0], km["kern_ys"][-1]])
    center_x = (km["kern_xs"][0] + km["kern_xs"][-1]) / 2
    center_y = (km["kern_ys"][0] + km["kern_ys"][-1]) / 2
    axis.plot(center_x, center_y, "x", ms=7, mew=1.8, color="white")
    axis.set_xticks([km["kern_xs"][0], 0, km["kern_xs"][-1]])
    axis.set_yticks([km["kern_ys"][0], 0, km["kern_ys"][-1]])
    axis.set_xlabel(r"Offset $d_x$ (cells)", fontsize=TEXT_FS)
    axis.set_ylabel(r"Offset $d_y$ (cells)", fontsize=TEXT_FS)
    axis.set_title("Learned kernel", fontsize=TITLE_FS, pad=TITLE_PAD)
    _box(axis)
    return image, -half_range, half_range


def _draw_field(axis, km, grid, title, cmap, symmetric, show_ylabel):
    vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    if symmetric:
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
    levels = np.linspace(vmin, vmax, 41)
    filled = axis.contourf(km["mz_vals"], km["jmz_vals"], grid, levels=levels,
                           cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    axis.contour(km["mz_vals"], km["jmz_vals"], grid,
                 levels=np.linspace(vmin, vmax, 13),
                 colors="white" if cmap == "viridis" else "black",
                 linewidths=0.4, alpha=0.6)
    axis.set_xlabel(r"$m_z$", fontsize=TEXT_FS)
    if show_ylabel:
        axis.set_ylabel(r"$Jm_z$", fontsize=TEXT_FS)
    else:
        axis.tick_params(labelleft=False)
    axis.set_title(title, fontsize=TITLE_FS, pad=TITLE_PAD)
    _box(axis)
    return filled, vmin, vmax


def plot_figure3(data_root, output_stem="figure3_2D_ferromagnet", dpi=300,
                 downsample=10, reference_data_root=None):
    """Generate the v15 Figure 3 PDF and JPG from an export data directory.

    ``data_root`` contains the generated ``data/rollout_deepmesoD`` directory.
    ``reference_data_root`` optionally supplies the original baseline/CNN/B
    panel tables. If omitted, all data are read below ``data_root``.
    """
    data_root = Path(data_root).expanduser().resolve()
    reference_data_root = (Path(reference_data_root).expanduser().resolve()
                           if reference_data_root is not None else None)
    paths = _paths(data_root, reference_data_root)
    output_stem = Path(output_stem).expanduser()
    macro = pd.read_csv(paths["pg_dir"] / "macro_statistics_t20_25000.csv")
    baseline = np.load(
        paths["macro_dir"] / "03_triangular_domain_configuration_macro_stats.npz"
    )
    cnn_long = (pd.read_csv(paths["cnn_csv"]) if paths["cnn_csv"].exists() else None)
    km = _load_km_data(paths["km_dir"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": TEXT_FS, "axes.titlesize": TITLE_FS,
        "axes.titlepad": TITLE_PAD, "axes.titleweight": "bold",
        "axes.labelsize": TEXT_FS, "xtick.labelsize": TEXT_FS,
        "ytick.labelsize": TEXT_FS, "legend.fontsize": TEXT_FS,
        "lines.linewidth": 1.5, "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    })

    figure_width = 22.0
    left, right = 0.045, 0.975
    top, bottom = 0.91, 0.085
    between_b, between_main, between_rows = 0.42, 0.12, 0.42
    aspect_a = 1.0
    aspect_b = aspect_a * (3 + 2 * between_b) / (SNAP_NCOL / SNAP_NROW)
    usable_width = (right - left) * figure_width
    b_layout = (1 + between_main / 2) * (3 + 2 * between_b)
    panel_width = aspect_b * usable_width / ((aspect_a + aspect_b) * b_layout)
    figure_height = panel_width * (2 + between_rows) / (top - bottom)

    figure = plt.figure(figsize=(figure_width, figure_height))
    main_grid = GridSpec(2, 2, figure=figure, width_ratios=[aspect_a, aspect_b],
                         wspace=between_main, hspace=between_rows,
                         left=left, right=right, top=top, bottom=bottom)
    b_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=main_grid[0, 1],
                                    wspace=between_b)
    b_bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=main_grid[1, 1],
                                       wspace=between_b)

    axis_mz = figure.add_subplot(main_grid[0, 0])
    _draw_mz(axis_mz, macro, baseline, downsample, cnn_long)
    axis_mz.set_xlabel("Time", fontsize=TEXT_FS)
    axis_kdecay = figure.add_subplot(b_top[0, 0])
    axis_coupling = figure.add_subplot(b_top[0, 1])
    axis_kernel = figure.add_subplot(b_top[0, 2])
    _draw_kdecay(axis_kdecay, km)
    _draw_coupling(axis_coupling, km)
    kernel_image, kernel_min, kernel_max = _draw_kernel(axis_kernel, km)

    hydro = np.outer(np.ones(len(km["jmz_vals"])), -km["mz_vals"] / 2)
    specifications = [
        (km["fe_grid"], "Free-energy landscape", "viridis", False),
        (km["dfjmz_grid"], r"Force along $Jm_z$", KERNEL_CMAP, True),
        (hydro, r"Additive ansatz along $Jm_z$", KERNEL_CMAP, True),
    ]
    field_axes, field_images, field_limits = [], [], []
    for column, (grid, title, cmap, symmetric) in enumerate(specifications):
        axis = figure.add_subplot(b_bottom[0, column])
        image, vmin, vmax = _draw_field(axis, km, grid, title, cmap, symmetric,
                                        show_ylabel=(column == 0))
        field_axes.append(axis)
        field_images.append(image)
        field_limits.append((vmin, vmax))

    figure.canvas.draw()
    colorbar_width, colorbar_gap = 0.004, 0.004
    _attach_colorbar(figure, axis_kernel, kernel_image, kernel_min, kernel_max,
                     colorbar_width, colorbar_gap)
    for axis, image, (vmin, vmax) in zip(field_axes, field_images, field_limits):
        _attach_colorbar(figure, axis, image, vmin, vmax, colorbar_width, colorbar_gap)

    snapshots = _load_snapshots(paths["pg_dir"], macro)
    slot = main_grid[1, 0].get_position(figure)
    slot_width, slot_height = slot.width * figure_width, slot.height * figure_height
    side = min((slot_width - (SNAP_NCOL - 1) * SNAP_GAP_IN) / SNAP_NCOL,
               (slot_height - (SNAP_NROW - 1) * SNAP_CAP_IN) / SNAP_NROW)
    grid_x = (slot_width - SNAP_NCOL * side) / (SNAP_NCOL - 1)
    grid_y = (slot_height - SNAP_NROW * side) / (SNAP_NROW - 1)
    height_fraction = side / figure_height
    vertical_gap_fraction = grid_y / figure_height
    heatmap_axes, heatmap_image = [], None
    for index, snapshot in enumerate(snapshots):
        row, column = divmod(index, SNAP_NCOL)
        x = slot.x0 + column * (side + grid_x) / figure_width
        y = slot.y1 - (row + 1) * height_fraction - row * vertical_gap_fraction
        axis = figure.add_axes([x, y, side / figure_width, height_fraction])
        image = axis.imshow(snapshot, cmap=KERNEL_CMAP, vmin=-1.0, vmax=1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for side_name in ("top", "right", "bottom", "left"):
            axis.spines[side_name].set_linewidth(plt.rcParams["axes.linewidth"])
        axis.text(0.5, -0.06, SNAP_LABELS[index], transform=axis.transAxes,
                  ha="center", va="top", fontsize=TEXT_FS)
        heatmap_axes.append(axis)
        heatmap_image = image if heatmap_image is None else heatmap_image

    figure.canvas.draw()
    first = heatmap_axes[0].get_position()
    heatmap_colorbar_axis = figure.add_axes(
        [first.x0 - 0.16 / figure_width, first.y0, 0.06 / figure_width, first.height]
    )
    heatmap_colorbar = figure.colorbar(heatmap_image, cax=heatmap_colorbar_axis)
    heatmap_colorbar.set_ticks([-1, 0, 1])
    heatmap_colorbar.ax.yaxis.set_ticks_position("left")
    heatmap_colorbar.ax.tick_params(labelsize=TEXT_FS)

    figure.canvas.draw()
    for label, reference_axis in (("A", axis_mz), ("B", axis_kdecay)):
        position = reference_axis.get_position()
        reference_axis.text(-0.55 / (position.width * figure_width),
                            1.0 + 0.10 / (position.height * figure_height), label,
                            transform=reference_axis.transAxes, fontsize=AB_FS,
                            ha="left", va="bottom", clip_on=False)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    jpg_path = output_stem.with_suffix(".jpg")
    figure.savefig(jpg_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi,
                   bbox_inches="tight", facecolor="white")
    figure.clear()
    plt.close(figure)
    output_stem.with_suffix(".provenance.json").write_text(
        json.dumps({"data_root": str(data_root), "source_plot":
                    "MesoscopicModel/exports/fig3_deepmesoD/plot_composite_grid2x4_v15.py",
                    "reference_data_root": (str(reference_data_root)
                                             if reference_data_root else None),
                    "downsample": downsample, "dpi": dpi}, indent=2),
        encoding="utf-8",
    )
    return pdf_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        help="pipeline config containing visualization.figure3 paths")
    parser.add_argument("--data-root", type=Path,
                        help="directory containing generated data/rollout_deepmesoD")
    parser.add_argument("--reference-data-root", type=Path,
                        help="optional old export root for baseline and B-panel tables")
    parser.add_argument("--output-stem", type=Path,
                        help="output filename stem; defaults to the config value")
    parser.add_argument("--set", action="append", default=[],
                        help="override a config value, e.g. visualization.figure3.dpi=600")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--downsample", type=int, default=10)
    args = parser.parse_args(argv)
    data_root = args.data_root
    reference_root = args.reference_data_root
    output_stem = args.output_stem
    if args.config:
        from deepmesod.config import apply_overrides, load_config

        config = apply_overrides(load_config(args.config), args.set)
        settings = config["visualization"]["figure3"]
        data_root = data_root or Path(settings["data_root"])
        reference_root = reference_root or Path(settings["reference_data_root"])
        output_stem = output_stem or Path(settings["output_stem"])
    if data_root is None:
        parser.error("--data-root or --config is required")
    print(plot_figure3(data_root, output_stem or "figure3_2D_ferromagnet",
                       args.dpi, args.downsample, reference_root))


if __name__ == "__main__":
    main()
