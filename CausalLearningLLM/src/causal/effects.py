"""Causal effect estimators (probe-space)."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def clinical_effect_probe(
    probe_clinical,
    Z: np.ndarray,
    Z_prime: np.ndarray,
    labels: np.ndarray,
) -> float:
    """CE_Y_probe: mean change in correct-class probability after intervention."""
    p_orig = probe_clinical.predict_proba(Z)
    p_int = probe_clinical.predict_proba(Z_prime)
    n = len(labels)
    orig = np.array([p_orig[i, labels[i]] for i in range(n)])
    intr = np.array([p_int[i, labels[i]] for i in range(n)])
    return float(np.mean(intr - orig))


def task_loss_probe(probe_clinical, Z: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy task loss from probe predictions."""
    proba = probe_clinical.predict_proba(Z)
    le = probe_clinical.label_encoder
    y_enc = le.transform(labels)
    return float(log_loss(y_enc, proba))


def _leakage_metrics(probe, Z: np.ndarray, s_labels: np.ndarray) -> Dict:
    le = probe.label_encoder
    try:
        s_enc = le.transform(s_labels)
    except Exception:
        s_enc = s_labels.astype(int)

    preds_enc = probe.clf.predict(Z)
    proba = probe.predict_proba(Z)
    n_classes = len(le.classes_)

    acc = float(accuracy_score(s_enc, preds_enc))
    bal = float(balanced_accuracy_score(s_enc, preds_enc))
    ce = float(log_loss(s_enc, proba))

    try:
        if n_classes == 2:
            auroc = float(roc_auc_score(s_enc, proba[:, 1]))
        else:
            auroc = float(roc_auc_score(s_enc, proba, multi_class="ovr", average="macro"))
    except Exception:
        auroc = 0.5

    return {"accuracy": acc, "balanced_accuracy": bal, "auroc": auroc, "cross_entropy": ce}


def sensitive_leakage(probe_sensitive, Z: np.ndarray, s_labels: np.ndarray) -> Dict:
    return _leakage_metrics(probe_sensitive, Z, s_labels)


def spurious_decodability(probe_spurious, Z: np.ndarray, c_labels: np.ndarray) -> Dict:
    return _leakage_metrics(probe_spurious, Z, c_labels)


def compute_pus(delta_leak: float, delta_task: float, beta: float = 1.0) -> float:
    """Privacy-Utility Score."""
    return delta_leak - beta * delta_task


def compute_rus(delta_spur: float, delta_task: float, eta: float = 1.0) -> float:
    """Robustness-Utility Score."""
    return delta_spur - eta * delta_task


def compute_cra_score(
    ce_y: float,
    delta_leak: float,
    delta_spur: float,
    delta_task: float,
    alpha_y: float = 1.0,
    alpha_p: float = 1.0,
    alpha_c: float = 1.0,
    gamma: float = 0.5,
) -> float:
    return alpha_y * abs(ce_y) + alpha_p * delta_leak + alpha_c * delta_spur - gamma * delta_task


def counterfactual_robustness(
    proba_orig: np.ndarray,
    proba_cf: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    pred_orig = proba_orig.argmax(1)
    pred_cf = proba_cf.argmax(1)
    consistency = float((pred_orig == pred_cf).mean())
    n = len(labels)
    orig_p = np.array([proba_orig[i, labels[i]] for i in range(n)])
    cf_p = np.array([proba_cf[i, labels[i]] for i in range(n)])
    return {
        "prediction_consistency": consistency,
        "spurious_reliance": 1.0 - consistency,
        "prob_consistency_rate": float(np.mean(np.abs(orig_p - cf_p))),
    }
