"""Tests for dataset loading."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_clinical import SyntheticClinicalDataset
from src.data.base import ClinicalSample


DUMMY_CONFIG = {"experiment": {"seed": 42}, "dataset": {}}


def test_synthetic_load():
    ds = SyntheticClinicalDataset(DUMMY_CONFIG, n_samples=20)
    ds.load()
    assert len(ds) == 20
    assert all(isinstance(s, ClinicalSample) for s in ds.samples)


def test_synthetic_labels_binary():
    ds = SyntheticClinicalDataset(DUMMY_CONFIG, n_samples=50)
    ds.load()
    labels = set(s.target_label for s in ds.samples)
    assert labels <= {0, 1}


def test_synthetic_sample_fields():
    ds = SyntheticClinicalDataset(DUMMY_CONFIG, n_samples=5)
    ds.load()
    s = ds.samples[0]
    assert s.sample_id.startswith("syn_")
    assert len(s.input_text) > 0
    assert s.answer_choices is not None
