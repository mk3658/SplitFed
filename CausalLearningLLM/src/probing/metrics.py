"""Probe metric helpers — single source of truth for all probe evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from typing import Optional


def probe_accuracy(probe, Z: np.ndarray, labels: np.ndarray) -> float:
    preds = probe.predict(Z)
    le = probe.label_encoder
    return float(accuracy_score(le.transform(labels), le.transform(preds)))


def chance_level(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))


def probe_auroc(probe, Z: np.ndarray, y) -> float:
    """Fixed-probe AUROC evaluated on Z.  Probe weights are NOT updated.

    Works for binary and multiclass probes.  Returns 0.5 on failure.
    This is the canonical implementation — import from here, do not redefine.
    """
    n_cls = len(probe.label_encoder.classes_)
    yl = y.tolist() if hasattr(y, "tolist") else list(y)
    unique = sorted(set(yl))
    remap = {v: i for i, v in enumerate(unique)}
    y_enc = np.array([remap[v] for v in yl])
    mask = y_enc < n_cls
    if mask.sum() < 10:
        return 0.5
    proba = probe.clf.predict_proba(Z[mask])
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_enc[mask], proba[:, 1]))
        return float(roc_auc_score(
            y_enc[mask], proba, multi_class="ovr", average="macro",
            labels=list(range(min(n_cls, proba.shape[1]))),
        ))
    except Exception:
        return 0.5


def bootstrap_auroc_ci(
    probe, Z: np.ndarray, y,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Bootstrap 95% CI for probe_auroc.  Returns (lower, mean, upper)."""
    rng = np.random.RandomState(seed)
    n = len(Z)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            a = probe_auroc(probe, Z[idx], np.array(y)[idx] if not hasattr(y, "iloc") else y.iloc[idx])
        except Exception:
            a = 0.5
        boot.append(a)
    lo = (1 - ci) / 2
    boot = np.array(boot)
    return float(np.quantile(boot, lo)), float(boot.mean()), float(np.quantile(boot, 1 - lo))
