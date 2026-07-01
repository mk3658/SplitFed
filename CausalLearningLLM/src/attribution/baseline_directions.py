"""Baseline direction methods for comparison with CRA."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.probing.directions import (
    get_mean_difference_direction,
    get_pca_direction,
    get_random_direction,
    run_inlp,
)


def get_all_baseline_directions(
    Z: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Return dict of method_name -> direction vector."""
    d = Z.shape[1]
    return {
        "random": get_random_direction(d, seed=seed),
        "mean_difference": get_mean_difference_direction(Z, labels),
        "pca": get_pca_direction(Z),
    }


def get_inlp_directions(
    Z: np.ndarray,
    labels: np.ndarray,
    n_rounds: int = 5,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run INLP and return (projection_matrix, removed_directions)."""
    return run_inlp(Z, labels, n_rounds=n_rounds)
