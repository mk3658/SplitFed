"""Context-manager and factory helpers for model-space interventions."""

from typing import Callable
import numpy as np


class InterventionHook:
    """Context manager that registers / removes a forward hook."""

    def __init__(self, model_wrapper, layer: int, intervention_fn: Callable):
        self.wrapper = model_wrapper
        self.layer = layer
        self.fn = intervention_fn

    def __enter__(self):
        self.wrapper.register_intervention_hook(self.layer, self.fn)
        return self

    def __exit__(self, *_):
        self.wrapper._remove_hooks()


def make_projection_fn(direction: np.ndarray, lam: float = 1.0, mode: str = "remove") -> Callable:
    """Return an intervention function operating on [batch, seq, hidden] arrays."""
    v = direction / (np.linalg.norm(direction) + 1e-12)

    def fn(hidden: np.ndarray) -> np.ndarray:
        orig = hidden.shape
        h = hidden.reshape(-1, orig[-1])
        proj = (h @ v[:, None]) * v[None, :]  # [N, d]
        if mode == "remove":
            out = h - lam * proj
        elif mode == "amplify":
            out = h + lam * proj
        else:
            out = h - proj
        return out.reshape(orig)

    return fn
