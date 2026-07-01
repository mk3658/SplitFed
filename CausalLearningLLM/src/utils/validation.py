"""Input validation helpers for CRA experiments."""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Sample-level validation
# ---------------------------------------------------------------------------

_REQUIRED_SAMPLE_KEYS = {"sample_id", "input_text", "target_label"}


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Return True if *sample* contains all required keys with non-empty values.

    Parameters
    ----------
    sample:
        Dictionary representing a single data sample.

    Returns
    -------
    bool
        ``True`` if valid, ``False`` otherwise.

    Raises
    ------
    TypeError
        If *sample* is not a dict.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"Expected dict, got {type(sample)}")

    for key in _REQUIRED_SAMPLE_KEYS:
        if key not in sample:
            return False
        value = sample[key]
        if value is None:
            return False
        if isinstance(value, str) and len(value.strip()) == 0:
            return False

    if not isinstance(sample["target_label"], (int, np.integer)):
        return False

    return True


# ---------------------------------------------------------------------------
# Representation-level validation
# ---------------------------------------------------------------------------

def validate_representations(
    Z: np.ndarray,
    labels: np.ndarray,
    min_samples: int = 2,
) -> None:
    """Assert that hidden-state matrix *Z* and *labels* are compatible.

    Parameters
    ----------
    Z:
        2-D array of shape ``(n_samples, n_features)``.
    labels:
        1-D integer array of length ``n_samples``.
    min_samples:
        Minimum number of samples required.

    Raises
    ------
    ValueError
        On any shape / type / size mismatch.
    """
    if not isinstance(Z, np.ndarray):
        raise ValueError(f"Z must be np.ndarray, got {type(Z)}")
    if not isinstance(labels, np.ndarray):
        raise ValueError(f"labels must be np.ndarray, got {type(labels)}")

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")

    n_samples = Z.shape[0]
    if n_samples != labels.shape[0]:
        raise ValueError(
            f"Z has {n_samples} rows but labels has {labels.shape[0]} elements"
        )

    if n_samples < min_samples:
        raise ValueError(
            f"At least {min_samples} samples required, got {n_samples}"
        )

    if not np.isfinite(Z).all():
        raise ValueError("Z contains NaN or Inf values")


# ---------------------------------------------------------------------------
# Generic shape checking
# ---------------------------------------------------------------------------

def check_shapes(**named_arrays: np.ndarray) -> None:
    """Assert that all named arrays have finite values and report mismatches.

    Usage::

        check_shapes(Z_train=Z_train, Z_test=Z_test)

    Raises
    ------
    ValueError
        If any array contains non-finite values.
    """
    for name, arr in named_arrays.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"'{name}' must be np.ndarray, got {type(arr)}")
        if not np.isfinite(arr).all():
            raise ValueError(f"'{name}' contains NaN or Inf values (shape={arr.shape})")


def check_consistent_lengths(*arrays: np.ndarray) -> None:
    """Raise ValueError if not all arrays have the same first-dimension length."""
    lengths = [a.shape[0] for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Arrays have inconsistent lengths: {lengths}")
