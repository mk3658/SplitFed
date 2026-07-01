"""SHAP-based feature attribution (graceful fallback)."""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def shap_attribution(probe, Z: np.ndarray, max_samples: int = 50) -> Optional[np.ndarray]:
    """Feature-level SHAP values on probe. Returns [n, d] or None."""
    try:
        import shap
        X_bg = Z[:min(20, len(Z))]
        X_eval = Z[:min(max_samples, len(Z))]
        explainer = shap.LinearExplainer(probe.clf, X_bg)
        shap_vals = explainer.shap_values(X_eval)
        if isinstance(shap_vals, list):
            return np.abs(np.stack(shap_vals)).mean(0)
        return np.abs(shap_vals)
    except ImportError:
        logger.warning("shap not installed — skipping SHAP attribution.")
        return None
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        return None
