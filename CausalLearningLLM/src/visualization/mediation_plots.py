"""Mediation analysis heatmap (Fig 12)."""

import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.style import CMAPS, set_style, save_figure


def plot_mediation_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    metric: str = "mediation_ratio",
    dpi: int = 150,
) -> None:
    if df.empty or metric not in df.columns:
        return

    set_style(dpi=dpi)
    pivot = df.pivot_table(index="layer", columns="cf_type", values=metric, aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(4, pivot.shape[1] * 1.5), max(3, pivot.shape[0] * 0.5 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap=CMAPS["sequential"])
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"L{l}" for l in pivot.index])
    ax.set_title(f"Mediation Analysis ({metric})")
    save_figure(fig, output_dir, "fig12_mediation_heatmap")
