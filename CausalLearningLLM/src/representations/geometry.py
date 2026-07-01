"""Geometric utilities for representation analysis."""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.dot(u, v))


def compute_mean_difference_direction(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    if len(classes) < 2:
        v = np.random.randn(Z.shape[1])
        return v / (np.linalg.norm(v) + 1e-12)
    mu0 = Z[labels == classes[0]].mean(0)
    mu1 = Z[labels == classes[1]].mean(0)
    diff = mu1 - mu0
    return diff / (np.linalg.norm(diff) + 1e-12)


def compute_pca_direction(Z: np.ndarray, k: int = 1) -> np.ndarray:
    pca = PCA(n_components=k)
    pca.fit(Z)
    return pca.components_[0] if k == 1 else pca.components_


def explained_variance_ratio(Z: np.ndarray, direction: np.ndarray) -> float:
    v = direction / (np.linalg.norm(direction) + 1e-12)
    proj_vals = Z @ v
    return float(np.var(proj_vals) / (np.var(Z).sum() + 1e-12))


def compute_direction_geometry(directions: Dict[str, np.ndarray]) -> Dict:
    names = list(directions.keys())
    result = {}
    for n1 in names:
        for n2 in names:
            result[(n1, n2)] = cosine_similarity(directions[n1], directions[n2])
    return result
