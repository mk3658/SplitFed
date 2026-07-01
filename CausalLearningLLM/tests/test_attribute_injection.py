"""Tests for synthetic attribute injection."""

import pytest, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_clinical import SyntheticClinicalDataset
from src.data.attributes import (
    inject_sensitive_attributes,
    inject_spurious_attributes,
    compute_empirical_correlation,
)

DUMMY_CONFIG = {"experiment": {"seed": 42}, "dataset": {}}


def _get_samples(n=60):
    ds = SyntheticClinicalDataset(DUMMY_CONFIG, n_samples=n)
    ds.load()
    return ds.samples


def test_sensitive_injection_keys():
    samples = _get_samples(20)
    rng = np.random.RandomState(0)
    samples = inject_sensitive_attributes(samples, ["age_group", "gender"], gamma=0.0, rng=rng)
    for s in samples:
        assert "age_group" in s.sensitive_attributes
        assert "gender" in s.sensitive_attributes


def test_independent_attribute_low_correlation():
    samples = _get_samples(60)
    rng = np.random.RandomState(1)
    samples = inject_sensitive_attributes(samples, ["age_group"], gamma=0.0, mode="independent", rng=rng)
    attr_vals = [s.sensitive_attributes.get("age_group", "unknown") for s in samples]
    labels = [s.target_label for s in samples]
    v, p = compute_empirical_correlation(attr_vals, labels)
    assert v < 0.5, f"Expected low correlation for independent mode, got {v}"


def test_correlated_attribute_higher_correlation():
    samples = _get_samples(80)
    rng = np.random.RandomState(2)
    samples = inject_sensitive_attributes(samples, ["age_group"], gamma=0.9, mode="correlated", rng=rng)
    attr_vals = [s.sensitive_attributes.get("age_group", "unknown") for s in samples]
    labels = [s.target_label for s in samples]
    v, p = compute_empirical_correlation(attr_vals, labels)
    # With gamma=0.9, we expect above-chance correlation
    assert v > 0.1, f"Expected some correlation for correlated mode, got {v}"
