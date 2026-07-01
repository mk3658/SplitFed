"""CRA framework overview diagram (Fig 1)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.visualization.style import COLORS, set_style, save_figure


def plot_framework_diagram(output_dir: str, dpi: int = 150) -> None:
    set_style(dpi=dpi)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.2, 1.5, "Clinical\nInput", COLORS["clinical"]),
        (2.0, 1.5, "Clinical\nLLM", "#444444"),
        (4.0, 1.5, "Layer-wise\nHidden States", COLORS["nuisance"]),
        (6.0, 1.5, "Probing &\nAttribution", COLORS["baseline"]),
        (8.0, 1.5, "Representation\nIntervention", COLORS["cra"]),
        (10.0, 1.5, "Taxonomy &\nEvaluation", COLORS["mixed"]),
    ]

    for x, y, label, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - 0.7, y - 0.55), 1.4, 1.1,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor=color, facecolor=color + "33",
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color="#222222")

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i + 1][0] - 0.7
        y_mid = boxes[i][1]
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    # Taxonomy labels at bottom
    tax_labels = ["Clinical", "Privacy-sensitive", "Spurious", "Mixed", "Nuisance"]
    tax_colors = [COLORS["clinical"], COLORS["sensitive"], COLORS["spurious"], COLORS["mixed"], COLORS["nuisance"]]
    for i, (lbl, col) in enumerate(zip(tax_labels, tax_colors)):
        ax.text(9.0 + (i % 3) * 1.1, 0.3 + (i // 3) * 0.5, lbl,
                ha="center", va="center", fontsize=7, color=col, fontweight="bold")

    ax.set_title("CRA: Causal Representation Attribution Framework", fontsize=13, pad=10)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig1_framework_diagram")
