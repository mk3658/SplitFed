"""Tests for counterfactual generation."""

import pytest, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_clinical import SyntheticClinicalDataset
from src.data.attributes import inject_sensitive_attributes, generate_counterfactuals

DUMMY_CONFIG = {"experiment": {"seed": 42}, "dataset": {}}


def test_counterfactual_label_preservation():
    ds = SyntheticClinicalDataset(DUMMY_CONFIG, n_samples=20)
    ds.load()
    rng = np.random.RandomState(0)
    samples = inject_sensitive_attributes(ds.samples, ["age_group"], gamma=0.0, rng=rng)
    samples = generate_counterfactuals(samples, cf_type="sensitive")
    for s in samples:
        if s.counterfactual_versions:
            for cf in s.counterfactual_versions:
                assert cf["target_label"] == s.target_label


def test_counterfactual_text_changes():
    ds = SyntheticClinicalDataset(DUMMY_CONFIG, n_samples=20)
    ds.load()
    rng = np.random.RandomState(0)
    samples = inject_sensitive_attributes(ds.samples, ["age_group"], gamma=0.0, rng=rng)
    samples = generate_counterfactuals(samples, cf_type="sensitive")
    has_cf = any(s.counterfactual_versions for s in samples)
    assert has_cf, "Expected at least some samples to have counterfactuals"
