"""Probe metric helpers."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def probe_accuracy(probe, Z: np.ndarray, labels: np.ndarray) -> float:
    preds = probe.predict(Z)
    le = probe.label_encoder
    return float(accuracy_score(le.transform(labels), le.transform(preds)))


def chance_level(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))
