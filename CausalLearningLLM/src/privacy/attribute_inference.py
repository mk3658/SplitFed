"""Attribute inference attack baseline."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict


def attribute_inference_attack(
    Z: np.ndarray, s_labels: np.ndarray, cv: int = 5
) -> Dict:
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    n_classes = len(np.unique(s_labels))
    cv_folds = min(cv, n_classes, int(np.bincount(s_labels).min()))
    cv_folds = max(cv_folds, 2)

    try:
        scores = cross_val_score(clf, Z, s_labels, cv=cv_folds, scoring="balanced_accuracy")
        mean_score = float(scores.mean())
        std_score = float(scores.std())
    except Exception:
        mean_score, std_score = 1.0 / n_classes, 0.0

    chance = 1.0 / n_classes
    return {
        "attack_balanced_accuracy": mean_score,
        "attack_std": std_score,
        "chance_level": chance,
        "attack_success": mean_score > chance + 0.05,
    }
