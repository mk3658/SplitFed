"""Direction-extraction helpers (probes, PCA, mean-diff, INLP, random)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def get_random_direction(d: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(d)
    return v / (np.linalg.norm(v) + 1e-12)


def get_mean_difference_direction(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    if len(classes) < 2:
        return get_random_direction(Z.shape[1])
    mu0 = Z[labels == classes[0]].mean(0)
    mu1 = Z[labels == classes[1]].mean(0)
    diff = mu1 - mu0
    return diff / (np.linalg.norm(diff) + 1e-12)


def get_pca_direction(Z: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=1)
    pca.fit(Z)
    return pca.components_[0]


def run_inlp(
    Z: np.ndarray,
    labels: np.ndarray,
    n_rounds: int = 5,
    C: float = 1.0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Iterative Nullspace Projection.

    Returns
    -------
    P : ndarray [d, d]
        Accumulated projection matrix.
    directions : list of ndarray [d]
        Directions removed in each round.
    """
    d = Z.shape[1]
    P = np.eye(d)
    directions: List[np.ndarray] = []
    Z_curr = Z.copy()

    for _ in range(n_rounds):
        clf = LogisticRegression(C=C, max_iter=500, class_weight="balanced")
        try:
            clf.fit(Z_curr, labels)
        except Exception:
            break

        W = clf.coef_
        if W.shape[0] == 1:
            v = W[0]
        else:
            _, _, Vt = np.linalg.svd(W, full_matrices=False)
            v = Vt[0]

        v = v / (np.linalg.norm(v) + 1e-12)
        directions.append(v)

        P_v = np.outer(v, v)
        P = (np.eye(d) - P_v) @ P
        Z_curr = Z_curr @ (np.eye(d) - P_v)

    return P, directions
