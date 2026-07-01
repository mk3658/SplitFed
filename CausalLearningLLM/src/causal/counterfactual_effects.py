"""Counterfactual effect computation."""

import numpy as np
from typing import Dict


def compute_counterfactual_robustness(
    probe_clinical,
    Z_orig: np.ndarray,
    Z_cf: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    proba_orig = probe_clinical.predict_proba(Z_orig)
    proba_cf = probe_clinical.predict_proba(Z_cf)

    pred_orig = proba_orig.argmax(1)
    pred_cf = proba_cf.argmax(1)

    n = len(labels)
    correct_orig = np.array([proba_orig[i, labels[i]] for i in range(n)])
    correct_cf = np.array([proba_cf[i, labels[i]] for i in range(n)])

    return {
        "prediction_consistency": float((pred_orig == pred_cf).mean()),
        "spurious_reliance": float((pred_orig != pred_cf).mean()),
        "mean_prob_diff": float(np.mean(np.abs(correct_orig - correct_cf))),
        "acc_orig": float((pred_orig == labels).mean()),
        "acc_cf": float((pred_cf == labels).mean()),
    }
