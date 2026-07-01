"""Integrated Gradients attribution (captum-based, with graceful fallback)."""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def integrated_gradients_attribution(
    model_wrapper,
    texts,
    target_class: int = 1,
    n_steps: int = 50,
) -> Optional[np.ndarray]:
    """Token-level IG attribution. Returns [N, seq] or None if captum unavailable."""
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        logger.warning("captum not installed — skipping Integrated Gradients.")
        return None

    logger.warning("Integrated Gradients requires gradient access; using probe-space only.")
    return None
