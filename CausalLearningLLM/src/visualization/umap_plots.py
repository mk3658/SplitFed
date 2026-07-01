"""UMAP visualisation of representations (Figs 9-11)."""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from src.visualization.style import set_style, save_figure

logger = logging.getLogger(__name__)


def plot_umap(
    Z: np.ndarray,
    color_labels,
    title: str,
    output_dir: str,
    name: str,
    dpi: int = 150,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP plot.")
        return

    set_style(dpi=dpi)
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    emb = reducer.fit_transform(Z)  # [N, 2]

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=color_labels, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Label")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    save_figure(fig, output_dir, name)
