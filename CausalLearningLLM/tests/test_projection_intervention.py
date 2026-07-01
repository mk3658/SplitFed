"""Tests for projection interventions (mathematical correctness)."""

import pytest, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.interventions import remove_direction, amplify_direction, remove_subspace, normalize


def test_remove_direction_orthogonality():
    """z' = z - (z·v)v should be orthogonal to v."""
    rng = np.random.RandomState(0)
    Z = rng.randn(50, 32)
    v = normalize(rng.randn(32))
    Z_prime = remove_direction(Z, v, lam=1.0)
    projections = Z_prime @ v
    np.testing.assert_allclose(projections, 0.0, atol=1e-5)


def test_remove_direction_norm():
    """Random direction must be unit norm."""
    from src.probing.directions import get_random_direction
    v = get_random_direction(64, seed=0)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_subspace_projection_idempotent():
    """P_V^2 = P_V (idempotent)."""
    rng = np.random.RandomState(1)
    V = rng.randn(3, 16)
    Q, _ = np.linalg.qr(V.T)
    P = Q @ Q.T  # [d, d]
    np.testing.assert_allclose(P @ P, P, atol=1e-6)


def test_amplify_increases_component():
    """Amplification should increase the component along v."""
    rng = np.random.RandomState(2)
    Z = rng.randn(20, 16)
    v = normalize(rng.randn(16))
    lam = 2.0
    Z_amp = amplify_direction(Z, v, lam=lam)
    proj_before = np.abs(Z @ v).mean()
    proj_after = np.abs(Z_amp @ v).mean()
    assert proj_after > proj_before


def test_full_removal_zero_component():
    """With lam=1.0, the direction component should be approximately zero."""
    rng = np.random.RandomState(3)
    Z = rng.randn(10, 8)
    v = normalize(rng.randn(8))
    Z_prime = remove_direction(Z, v, lam=1.0)
    np.testing.assert_allclose(Z_prime @ v, 0.0, atol=1e-5)
