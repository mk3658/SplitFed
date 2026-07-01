"""Causal tracing: corrupt and restore hidden states at specific positions."""

import numpy as np
from typing import Dict, List, Optional


def causal_trace_probe(
    probe_clinical,
    Z_clean: np.ndarray,
    Z_corrupt: np.ndarray,
    labels: np.ndarray,
    restore_indices: Optional[List[int]] = None,
) -> Dict:
    """Probe-space causal tracing: restore selected samples and measure recovery."""
    if restore_indices is None:
        restore_indices = list(range(len(Z_clean)))

    Z_trace = Z_corrupt.copy()
    Z_trace[restore_indices] = Z_clean[restore_indices]

    from src.causal.effects import clinical_effect_probe, task_loss_probe

    ce_corrupt = clinical_effect_probe(probe_clinical, Z_clean, Z_corrupt, labels)
    ce_restore = clinical_effect_probe(probe_clinical, Z_clean, Z_trace, labels)

    return {
        "ce_corrupt": ce_corrupt,
        "ce_restore": ce_restore,
        "recovery": ce_restore - ce_corrupt,
    }
