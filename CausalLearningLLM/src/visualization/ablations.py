"""Ablation study summary plot (Fig 16)."""

import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.style import COLORS, set_style, save_figure


def plot_ablation_summary(
    df: pd.DataFrame,
    output_dir: str,
    metric: str = "privacy_utility_score",
    dpi: int = 150,
) -> None:
    if df.empty or metric not in df.columns:
        return

    set_style(dpi=dpi)
    summary = df.groupby("direction_type")[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [COLORS.get(r.replace("_probe", ""), COLORS["baseline"]) for r in summary["direction_type"]]
    ax.barh(summary["direction_type"], summary[metric], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Ablation Study: Method Comparison")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig16_ablation_summary")
