"""Spurious shortcut metrics."""

import numpy as np
from typing import Dict


def spurious_decodability(probe_spurious, Z: np.ndarray, c_labels: np.ndarray) -> Dict:
    from src.causal.effects import sensitive_leakage
    return sensitive_leakage(probe_spurious, Z, c_labels)


def counterfactual_flip_rate(probe_clinical, Z_orig: np.ndarray, Z_cf_spurious: np.ndarray) -> float:
    pred_orig = probe_clinical.predict(Z_orig)
    pred_cf = probe_clinical.predict(Z_cf_spurious)
    return float((pred_orig != pred_cf).mean())
