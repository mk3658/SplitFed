"""Privacy leakage metrics."""

import numpy as np
from typing import Dict
from sklearn.feature_selection import mutual_info_classif


def estimate_leakage(probe_sensitive, Z: np.ndarray, s_labels: np.ndarray) -> Dict:
    from src.causal.effects import sensitive_leakage
    return sensitive_leakage(probe_sensitive, Z, s_labels)


def estimate_mi_knn(Z: np.ndarray, labels: np.ndarray, n_neighbors: int = 3) -> float:
    mi = mutual_info_classif(Z, labels, n_neighbors=n_neighbors, random_state=42)
    return float(mi.mean())


def conditional_leakage(
    probe_sensitive,
    Z: np.ndarray,
    s_labels: np.ndarray,
    y_labels: np.ndarray,
) -> Dict:
    """Leak(Z;S|Y) estimated as mean per-class leakage."""
    classes = np.unique(y_labels)
    scores = []
    for c in classes:
        mask = y_labels == c
        if mask.sum() < 5 or len(np.unique(s_labels[mask])) < 2:
            continue
        try:
            res = estimate_leakage(probe_sensitive, Z[mask], s_labels[mask])
            scores.append(res["auroc"])
        except Exception:
            pass
    return {
        "conditional_leakage_auroc": float(np.mean(scores)) if scores else 0.5,
        "n_classes_evaluated": len(scores),
    }
