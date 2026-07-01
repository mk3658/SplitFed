"""Attention rollout for token-level attribution (fallback implementation)."""

import numpy as np
from typing import Optional


def attention_rollout(
    attentions,  # tuple of [batch, heads, seq, seq]
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """Compute attention rollout importance scores [batch, seq]."""
    result = None
    for attn in attentions:
        if hasattr(attn, "numpy"):
            attn = attn.numpy()
        attn = attn.mean(1)  # [batch, seq, seq]

        # Flatten small attentions
        flat = attn.reshape(attn.shape[0], -1)
        threshold = np.quantile(flat, discard_ratio, axis=1, keepdims=True).reshape(-1, 1, 1)
        attn = np.where(attn >= threshold, attn, 0.0)

        # Add residual
        eye = np.eye(attn.shape[-1])[None]
        attn = (attn + eye) / 2.0
        attn = attn / (attn.sum(-1, keepdims=True) + 1e-12)

        result = attn if result is None else np.matmul(attn, result)

    if result is None:
        return np.array([])
    return result[:, 0, :]  # [batch, seq] from CLS to all tokens
