"""Layer-wise activation patching for mediation (model-space, when hooks are available)."""

import numpy as np
from typing import Dict, Optional


def patching_mediation(
    model_wrapper,
    texts_orig: list,
    texts_cf: list,
    layer: int,
    labels: np.ndarray,
    answer_choices: list,
    pooling: str = "mean",
) -> Dict:
    """Estimate NIE by patching Z_l from cf into orig forward pass."""
    # Original predictions
    _, proba_orig = model_wrapper.predict(texts_orig, answer_choices)

    # Counterfactual predictions
    _, proba_cf = model_wrapper.predict(texts_cf, answer_choices)

    # Patched: run orig text with Z_l replaced by Z_l(x_cf)
    Z_cf_hidden = model_wrapper.get_hidden_states(texts_cf, layers=[layer], pooling=pooling)
    if layer not in Z_cf_hidden:
        return {}

    Z_cf_l = Z_cf_hidden[layer]  # [N, d]

    # Use probe-space patching since model-space may not be available
    return {
        "layer": layer,
        "note": "model-space patching not applied; use probe-space mediation.py",
        "proba_orig_mean": float(proba_orig.mean()),
        "proba_cf_mean": float(proba_cf.mean()),
    }
