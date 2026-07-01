"""Reproducibility utilities for CRA experiments."""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed value.
    deterministic:
        If True, enable deterministic CUDA operations.  This may slow down
        training but ensures full reproducibility on GPU.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (imported lazily to avoid hard dependency at import time)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch >= 1.8 supports CUBLAS determinism
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                pass  # older PyTorch versions

    except ImportError:
        pass  # torch not installed; skip silently

    # Environment variable used by some HuggingFace / Jax internals
    os.environ["PYTHONHASHSEED"] = str(seed)
