"""Tests for representation extraction shapes (no model needed — uses mock)."""

import pytest, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.representations.extraction import get_sensitive_labels, get_spurious_labels


def _fake_layer_data(n=30, d=16):
    return {
        0: {
            "Z": np.random.randn(n, d),
            "labels": np.random.randint(0, 2, n),
            "sensitive": [{"age_group": "young" if i % 2 == 0 else "older"} for i in range(n)],
            "spurious": [{"hospital_type": "urban" if i % 3 == 0 else "rural"} for i in range(n)],
            "sample_ids": [f"s_{i}" for i in range(n)],
            "layer": 0, "pooling": "mean", "n_samples": n, "hidden_size": d,
        }
    }


def test_get_sensitive_labels_shape():
    ld = _fake_layer_data(30, 16)
    s = get_sensitive_labels(ld, "age_group", 0)
    assert s is not None
    assert len(s) == 30


def test_get_sensitive_labels_binary():
    ld = _fake_layer_data(30, 16)
    s = get_sensitive_labels(ld, "age_group", 0)
    assert set(s.tolist()) == {0, 1}


def test_get_spurious_labels_ternary():
    ld = _fake_layer_data(30, 16)
    c = get_spurious_labels(ld, "hospital_type", 0)
    assert c is not None
    assert len(np.unique(c)) >= 2
