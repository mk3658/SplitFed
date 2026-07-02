"""
Low-level representation intervention operations.

Joint simultaneous subspace removal (CRA) vs sequential (INLP):

  INLP removes directions one at a time.  When direction k+1 is found,
  it is learned from Z already projected by directions 1..k.  Sequential
  projection distorts the remaining subspace because:
    - direction k+1 is no longer orthogonal to directions 1..k in Z
    - each round adds rounding error that accumulates

  CRA joint removal:
    1. Collect all k directions {v_1, ..., v_k}
    2. Gram-Schmidt orthonormalize → {u_1, ..., u_k}  (orthonormal basis)
    3. Project Z onto the complement of span{u_1,...,u_k} in one shot:
       Z' = Z - lam * (Z V^T) V   where V = [u_1; ...; u_k]
    4. Result: removes entire subspace simultaneously, no ordering artefacts.

  Geometry preserved by joint removal:
    - Angles between directions outside the erased subspace are unchanged
    - INLP's sequential approach modifies those angles, breaking geometry
"""

import numpy as np
from typing import List


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


# ── joint causal subspace removal (CRA's key differentiator) ──────────────────

def gram_schmidt(directions: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Gram-Schmidt orthonormalization of row vectors in directions [k, d].
    Returns [k', d] orthonormal basis (rows with near-zero norm discarded).
    """
    basis = []
    for v in directions:
        for u in basis:
            v = v - np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > tol:
            basis.append(v / norm)
    return np.stack(basis) if basis else np.zeros((0, directions.shape[1]))


def joint_causal_removal(
    Z: np.ndarray,
    directions: List[np.ndarray],
    lam: float = 1.0,
) -> np.ndarray:
    """
    Simultaneously remove the subspace spanned by `directions` from Z.

    Unlike INLP (sequential, order-dependent), this performs a single
    orthogonal projection onto the complement of span(directions),
    preserving geometry outside the erased subspace exactly.

    Parameters
    ----------
    Z          : [n, d]  representations
    directions : list of [d] unit-norm vectors (need not be orthogonal)
    lam        : erasure strength in [0, 1]; 1.0 = full removal

    Returns
    -------
    Z_int : [n, d]  representations with subspace removed
    """
    if not directions:
        return Z.copy()
    V_raw = np.stack(directions)              # [k, d]
    V = gram_schmidt(V_raw)                   # [k', d] orthonormal
    if V.shape[0] == 0:
        return Z.copy()
    proj = Z @ V.T @ V                        # [n, d]  projection onto subspace
    return Z - lam * proj


def geometry_preservation_score(
    Z_orig: np.ndarray,
    Z_joint: np.ndarray,
    Z_sequential: np.ndarray,
    n_pairs: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compare how well joint vs sequential removal preserves pairwise geometry
    (cosine similarities) among samples.

    Lower distortion = better geometry preservation.
    This quantifies CRA's advantage over INLP.
    """
    rng = np.random.RandomState(seed)
    n = min(len(Z_orig), 300)
    idx = rng.choice(len(Z_orig), n, replace=False)
    A, B, C = Z_orig[idx], Z_joint[idx], Z_sequential[idx]

    # Pairwise cosine similarities in original space
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    cos_orig  = An @ An.T
    cos_joint = Bn @ Bn.T
    cos_seq   = Cn @ Cn.T

    mask = np.triu(np.ones_like(cos_orig, dtype=bool), k=1)
    orig_v  = cos_orig[mask]
    joint_v = cos_joint[mask]
    seq_v   = cos_seq[mask]

    distort_joint = float(np.mean((orig_v - joint_v) ** 2))
    distort_seq   = float(np.mean((orig_v - seq_v)   ** 2))

    return {
        "distortion_joint_removal":      distort_joint,
        "distortion_sequential_removal": distort_seq,
        "geometry_advantage":            distort_seq - distort_joint,  # >0 = CRA better
        "relative_improvement_pct":      100.0 * (distort_seq - distort_joint) / (distort_seq + 1e-12),
    }
