"""Theorem verification multi-panel figure (Fig 15)."""

import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.style import COLORS, set_style, save_figure


def plot_theorem_verification(
    claim_dfs: dict,
    output_dir: str,
    dpi: int = 150,
) -> None:
    set_style(dpi=dpi)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    # Claim 1: leakage before/after
    c1 = claim_dfs.get("claim1")
    if c1 is not None and not c1.empty and "leak_before" in c1.columns:
        ax = axes[0]
        ax.bar(c1["layer"], c1["leak_before"], label="Before", color=COLORS["sensitive"], alpha=0.6)
        ax.bar(c1["layer"], c1["leak_after"], label="After", color=COLORS["cra"], alpha=0.8)
        ax.set_title("Claim 1: Leakage Reduction")
        ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
        ax.legend()

    # Claim 2: utility vs orthogonality
    c2 = claim_dfs.get("claim2")
    if c2 is not None and not c2.empty and "cos_y_s" in c2.columns:
        ax = axes[1]
        ax.scatter(c2["cos_y_s"], c2["delta_task_loss"], color=COLORS["cra"], alpha=0.8)
        ax.set_title("Claim 2: Utility vs Orthogonality")
        ax.set_xlabel("|cos(v_y, v_s)|"); ax.set_ylabel("ΔTask loss")

    # Claim 3: robustness before/after
    c3 = claim_dfs.get("claim3")
    if c3 is not None and not c3.empty and "robustness_before" in c3.columns:
        ax = axes[2]
        ax.bar(c3["layer"], c3["robustness_before"], label="Before", color=COLORS["spurious"], alpha=0.6)
        ax.bar(c3["layer"], c3["robustness_after"], label="After", color=COLORS["cra"], alpha=0.8)
        ax.set_title("Claim 3: Robustness After Spurious Removal")
        ax.set_xlabel("Layer"); ax.set_ylabel("Consistency")
        ax.legend()

    # Claim 4: random projection
    c4 = claim_dfs.get("claim4")
    if c4 is not None and not c4.empty and "empirical_mean_sq_proj" in c4.columns:
        ax = axes[3]
        ax.bar(c4["layer"], c4["empirical_mean_sq_proj"], label="Empirical", color=COLORS["random"], alpha=0.8)
        ax.bar(c4["layer"], c4["theoretical_mean"], label="Theoretical k/d", color=COLORS["clinical"], alpha=0.5)
        ax.set_title("Claim 4: Random Projection")
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean sq. projection")
        ax.legend()

    # Claim 5: PUS comparison
    c5 = claim_dfs.get("claim5")
    if c5 is not None and not c5.empty and "privacy_utility_score" in c5.columns:
        ax = axes[4]
        ax.barh(c5["direction_type"], c5["privacy_utility_score"], color=COLORS["cra"], alpha=0.8)
        ax.set_title("Claim 5: PUS by Method")
        ax.set_xlabel("Privacy-Utility Score")

    # Claim 6: marginal vs conditional leakage
    c6 = claim_dfs.get("claim6")
    if c6 is not None and not c6.empty and "marginal_leakage" in c6.columns:
        ax = axes[5]
        ax.plot(c6["layer"], c6["marginal_leakage"], label="Marginal", color=COLORS["sensitive"], marker="o")
        ax.plot(c6["layer"], c6["conditional_leakage"], label="Conditional", color=COLORS["cra"], marker="s")
        ax.set_title("Claim 6: Conditional Leakage")
        ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
        ax.legend()

    fig.tight_layout()
    save_figure(fig, output_dir, "fig15_theorem_verification")
