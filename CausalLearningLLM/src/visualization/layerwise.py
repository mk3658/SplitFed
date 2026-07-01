"""Layer-wise information profile plot (Figure 2)."""

import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.style import COLORS, set_style, save_figure


def plot_layer_information(
    df: pd.DataFrame,
    output_dir: str,
    dpi: int = 150,
) -> None:
    if df.empty:
        return

    set_style(dpi=dpi)
    fig, ax = plt.subplots(figsize=(8, 5))

    if "layer" in df.columns:
        x = df["layer"]
        if "clinical_auroc" in df.columns:
            ax.plot(x, df["clinical_auroc"], label="Clinical (AUROC)", color=COLORS["clinical"], marker="o")
        if "sensitive_auroc" in df.columns:
            ax.plot(x, df["sensitive_auroc"], label="Sensitive leakage (AUROC)", color=COLORS["sensitive"], marker="s")
        if "spurious_auroc" in df.columns:
            ax.plot(x, df["spurious_auroc"], label="Spurious (AUROC)", color=COLORS["spurious"], marker="^")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Layer-wise Information Profile")
    ax.legend()
    save_figure(fig, output_dir, "fig2_layer_information")
