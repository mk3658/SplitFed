"""Heatmap plots for direction geometry and causal effects (Figs 3, 5)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.visualization.style import CMAPS, set_style, save_figure


def plot_direction_geometry_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    dpi: int = 150,
) -> None:
    if df.empty:
        return

    set_style(dpi=dpi)
    cos_cols = [c for c in df.columns if c.startswith("cos_") and c not in ("cos_y_s", "cos_y_c", "cos_s_c")]
    if not cos_cols:
        cos_cols = ["cos_y_s", "cos_y_c", "cos_s_c"]
    cos_cols = [c for c in cos_cols if c in df.columns]
    if not cos_cols:
        return

    mat = df[cos_cols].values  # [layers, pairs]

    fig, ax = plt.subplots(figsize=(max(4, len(cos_cols)), max(3, len(df) * 0.4 + 1)))
    im = ax.imshow(mat, aspect="auto", cmap=CMAPS["diverging"], vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(len(cos_cols)))
    ax.set_xticklabels(cos_cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"L{l}" for l in df.get("layer", range(len(df)))], fontsize=9)
    ax.set_title("Direction Geometry (Cosine Similarities)")
    save_figure(fig, output_dir, "fig3_direction_geometry")


def plot_causal_effect_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    metric: str = "cra_score",
    dpi: int = 150,
) -> None:
    if df.empty or metric not in df.columns:
        return

    set_style(dpi=dpi)
    pivot = df.pivot_table(index="layer", columns="direction_type", values=metric, aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(4, pivot.shape[1] * 1.2), max(3, pivot.shape[0] * 0.5 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap=CMAPS["heatmap"])
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"L{l}" for l in pivot.index], fontsize=9)
    ax.set_title(f"Causal Effect Heatmap ({metric})")
    save_figure(fig, output_dir, "fig5_causal_effect_heatmap")
