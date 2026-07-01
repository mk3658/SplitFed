"""Pareto frontier plots (Figs 6, 7)."""

import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.style import COLORS, set_style, save_figure


def plot_privacy_utility_pareto(
    df: pd.DataFrame,
    output_dir: str,
    dpi: int = 150,
) -> None:
    if df.empty:
        return

    set_style(dpi=dpi)
    fig, ax = plt.subplots(figsize=(7, 5))

    for d_type, grp in df.groupby("direction_type"):
        color = COLORS.get(d_type.replace("_probe", ""), COLORS["baseline"])
        x = grp.get("delta_task_loss", grp.iloc[:, -1])
        y = grp.get("delta_leak", grp.iloc[:, 1])
        ax.scatter(x, y, label=d_type, color=color, alpha=0.7)

    ax.set_xlabel("Utility loss (ΔTask loss)")
    ax.set_ylabel("Leakage reduction (ΔLeak)")
    ax.set_title("Privacy–Utility Pareto Frontier")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "fig6_privacy_utility_pareto")


def plot_robustness_spuriousness_pareto(
    df: pd.DataFrame,
    output_dir: str,
    dpi: int = 150,
) -> None:
    if df.empty:
        return

    set_style(dpi=dpi)
    fig, ax = plt.subplots(figsize=(7, 5))

    for d_type, grp in df.groupby("direction_type"):
        color = COLORS.get(d_type.replace("_probe", ""), COLORS["baseline"])
        x = grp.get("delta_task_loss", grp.iloc[:, -1])
        y = grp.get("delta_spur", grp.iloc[:, 1])
        ax.scatter(x, y, label=d_type, color=color, alpha=0.7)

    ax.set_xlabel("Utility loss (ΔTask loss)")
    ax.set_ylabel("Spuriousness reduction (ΔSpur)")
    ax.set_title("Robustness–Spuriousness Pareto Frontier")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "fig7_robustness_spuriousness_pareto")
