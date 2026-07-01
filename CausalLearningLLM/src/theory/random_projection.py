"""Random projection theory utilities."""

import numpy as np
from typing import Dict


def expected_squared_projection(k: int, d: int) -> float:
    """E[||P_V r||^2] = k/d for random unit vector r and k-dim subspace."""
    return k / d


def empirical_random_projections(
    subspace_directions: np.ndarray,
    n_random: int = 1000,
    seed: int = 0,
) -> Dict:
    """Empirically estimate mean squared projection of random vectors onto subspace."""
    rng = np.random.RandomState(seed)
    d = subspace_directions.shape[-1]
    Q, _ = np.linalg.qr(subspace_directions.T)
    V = Q.T  # [k, d]

    projections = []
    for _ in range(n_random):
        r = rng.randn(d)
        r /= np.linalg.norm(r) + 1e-12
        proj = r @ V.T  # [k]
        projections.append(float(np.sum(proj ** 2)))

    k = V.shape[0]
    return {
        "empirical_mean": float(np.mean(projections)),
        "theoretical_mean": k / d,
        "k": k,
        "d": d,
        "ratio": float(np.mean(projections)) / (k / d),
    }
