import matplotlib.pyplot as plt

COLORS = {
    "reference": "#2ca02c",
    "hydrodynamic": "#000000",
    "allen_cahn": "#8c564b",
    "cnn": "#ff7f0e",
    "deepmesod": "#1f77b4",
}


def apply_paper_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 20,
            "axes.titlesize": 21,
            "axes.titlepad": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 16,
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
        }
    )
