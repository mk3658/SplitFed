"""Pooling helpers for numpy hidden-state arrays."""

import numpy as np
from typing import Optional


def pool_hidden(
    hidden: np.ndarray,
    attention_mask: Optional[np.ndarray] = None,
    method: str = "mean",
) -> np.ndarray:
    """Pool [batch, seq, hidden] -> [batch, hidden]."""
    if method == "mean":
        if attention_mask is not None:
            mask = attention_mask[:, :, None].astype(float)
            return (hidden * mask).sum(1) / mask.sum(1).clip(min=1e-9)
        return hidden.mean(1)
    elif method == "last":
        if attention_mask is not None:
            lengths = attention_mask.sum(-1).astype(int) - 1
            return hidden[np.arange(len(hidden)), lengths]
        return hidden[:, -1]
    elif method == "max":
        return hidden.max(1)
    return hidden.mean(1)
