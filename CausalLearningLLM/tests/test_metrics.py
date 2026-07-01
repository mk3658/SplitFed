"""Tests for causal effect metrics."""

import pytest, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.effects import compute_pus, compute_rus, compute_cra_score, counterfactual_robustness


def test_pus_positive_when_leak_reduces_more_than_utility():
    assert compute_pus(delta_leak=0.2, delta_task=0.05, beta=1.0) > 0


def test_rus_negative_when_utility_loss_exceeds_gain():
    assert compute_rus(delta_spur=0.01, delta_task=0.5, eta=1.0) < 0


def test_cra_score_increases_with_leak_reduction():
    s1 = compute_cra_score(0.1, 0.3, 0.2, 0.05)
    s2 = compute_cra_score(0.1, 0.5, 0.2, 0.05)
    assert s2 > s1


def test_counterfactual_robustness_consistent():
    rng = np.random.RandomState(0)
    n, c = 50, 3
    p_orig = rng.dirichlet([1]*c, n)
    p_cf = p_orig.copy()  # same predictions
    labels = rng.randint(0, c, n)
    res = counterfactual_robustness(p_orig, p_cf, labels)
    assert res["prediction_consistency"] == 1.0
    assert res["spurious_reliance"] == 0.0
