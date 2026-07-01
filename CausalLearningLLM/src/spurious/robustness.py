"""Robustness evaluation helpers."""
import numpy as np
from typing import Dict


def compute_robustness_metrics(probe_clinical, Z_orig, Z_cf, labels) -> Dict:
    from src.causal.counterfactual_effects import compute_counterfactual_robustness
    return compute_counterfactual_robustness(probe_clinical, Z_orig, Z_cf, labels)
