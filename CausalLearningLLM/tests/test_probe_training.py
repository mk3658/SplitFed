"""Tests for probe training."""

import pytest, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.probes import LogisticRegressionProbe


def _make_linearly_separable(n=100, d=16, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    X = rng.randn(n, d) + y[:, None] * 3.0
    return X, y


def test_probe_above_chance():
    X, y = _make_linearly_separable(100, 16)
    probe = LogisticRegressionProbe(C=1.0, max_iter=500)
    probe.fit(X[:70], y[:70])
    metrics = probe.score(X[70:], y[70:])
    assert metrics["accuracy"] > 0.6, "Probe should beat chance on linearly separable data"


def test_probe_direction_unit_norm():
    X, y = _make_linearly_separable()
    probe = LogisticRegressionProbe()
    probe.fit(X, y)
    v = probe.get_direction()
    assert v is not None
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_probe_predict_proba_sums_to_one():
    X, y = _make_linearly_separable(50, 8)
    probe = LogisticRegressionProbe()
    probe.fit(X, y)
    proba = probe.predict_proba(X[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_probe_metrics_keys():
    X, y = _make_linearly_separable(60, 8)
    probe = LogisticRegressionProbe()
    probe.fit(X[:40], y[:40])
    metrics = probe.score(X[40:], y[40:])
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "cross_entropy"):
        assert key in metrics
