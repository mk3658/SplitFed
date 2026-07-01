"""Low-level representation intervention operations."""

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def project_direction(Z: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Component of Z along direction. Returns [N, d]."""
    v = _unit(direction)
    return np.outer(Z @ v, v)


def remove_direction(Z: np.ndarray, direction: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return Z - lam * project_direction(Z, direction)


def amplify_direction(Z: np.ndarray, direction: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return Z + lam * project_direction(Z, direction)


def scale_direction(Z: np.ndarray, direction: np.ndarray, scale: float = 0.0) -> np.ndarray:
    proj = project_direction(Z, direction)
    return Z + (scale - 1.0) * proj


def remove_subspace(Z: np.ndarray, directions: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Remove projection onto subspace spanned by directions [k, d]."""
    if directions.ndim == 1:
        return remove_direction(Z, directions, lam)
    Q, _ = np.linalg.qr(directions.T)
    V = Q.T  # [k, d] orthonormal
    proj = Z @ V.T @ V
    return Z - lam * proj


def replace_direction(
    Z: np.ndarray,
    Z_cf: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Replace the component of Z along direction with that from Z_cf."""
    v = _unit(direction)
    return Z - np.outer(Z @ v, v) + np.outer(Z_cf @ v, v)


def patch_representation(Z: np.ndarray, Z_cf: np.ndarray) -> np.ndarray:
    return Z_cf.copy()


def normalize(v: np.ndarray) -> np.ndarray:
    return _unit(v)
