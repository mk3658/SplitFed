"""Activation patching baseline."""

import numpy as np
from typing import Dict


def activation_patch_probe(
    probe_clinical,
    Z_clean: np.ndarray,
    Z_corrupt: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Patch Z_clean[layer] with Z_corrupt and measure prediction change."""
    from src.causal.effects import clinical_effect_probe, task_loss_probe

    ce = clinical_effect_probe(probe_clinical, Z_clean, Z_corrupt, labels)
    loss_clean = task_loss_probe(probe_clinical, Z_clean, labels)
    loss_patched = task_loss_probe(probe_clinical, Z_corrupt, labels)

    return {
        "clinical_effect": ce,
        "loss_clean": loss_clean,
        "loss_patched": loss_patched,
        "delta_task": loss_patched - loss_clean,
    }
