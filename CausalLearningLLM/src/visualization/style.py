"""Matplotlib style configuration for publication-quality figures."""

import matplotlib
import matplotlib.pyplot as plt
from typing import Dict

# Colorblind-safe palette with semantic meaning
COLORS = {
    "clinical": "#1f77b4",       # blue
    "sensitive": "#d62728",      # red
    "spurious": "#2ca02c",       # green
    "cra": "#9467bd",            # purple
    "random": "#7f7f7f",         # gray
    "baseline": "#17becf",       # teal
    "nuisance": "#c7c7c7",       # light gray
    "mixed": "#ff7f0e",          # orange
    "mean_difference": "#8c564b",
    "pca": "#e377c2",
}

CMAPS = {
    "diverging": "RdBu_r",
    "sequential": "viridis",
    "heatmap": "cividis",
}


def set_style(font_size: int = 12, dpi: int = 150) -> None:
    """Apply CRA paper style to matplotlib."""
    matplotlib.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.dpi": dpi,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": dpi,
    })


def save_figure(fig, output_dir: str, name: str, formats=("pdf", "svg", "png")) -> None:
    import os
    for fmt in formats:
        path = os.path.join(output_dir, "figures", fmt, f"{name}.{fmt}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
    plt.close(fig)


def get_color(key: str) -> str:
    return COLORS.get(key, "#333333")
