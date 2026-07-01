"""Tests for representation taxonomy classification."""

import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.cra import CRAFramework
import numpy as np


DUMMY_CONFIG = {
    "experiment": {"seed": 42},
    "dataset": {},
    "cra": {
        "taxonomy_thresholds": {"clinical": 0.6, "sensitive": 0.55, "spurious": 0.55, "utility_loss": 0.1},
        "intervention_strengths": [1.0],
        "direction_types": ["random"],
        "layer_selection": [0],
        "beta_privacy": 1.0,
        "eta_robustness": 1.0,
        "gamma_task": 0.5,
    },
    "probes": {"regularization": 1.0, "max_iter": 100, "test_size": 0.3, "random_state": 42},
}


def _make_probe_results(layer, clin_auroc, sens_auroc, spur_auroc):
    from src.probing.probes import LogisticRegressionProbe
    rng = np.random.RandomState(42)
    # Create minimal probe stubs with known metric values
    class FakeProbe:
        label_encoder = type("LE", (), {"classes_": np.array([0, 1]), "transform": lambda self, y: y})()
        def predict(self, X): return np.zeros(len(X), dtype=int)
        def predict_proba(self, X):
            n = len(X)
            return np.hstack([np.full((n, 1), 0.5), np.full((n, 1), 0.5)])
        def get_direction(self): return rng.randn(8) / (np.linalg.norm(rng.randn(8)) + 1e-12)

    return {
        layer: {
            "probes": {"clinical": FakeProbe(), "sensitive": FakeProbe(), "spurious": FakeProbe()},
            "metrics": {
                "clinical": {"auroc": clin_auroc},
                "sensitive": {"auroc": sens_auroc},
                "spurious": {"auroc": spur_auroc},
            },
        }
    }


def _make_layer_data(layer):
    rng = np.random.RandomState(0)
    return {
        layer: {
            "Z": rng.randn(20, 8),
            "labels": np.array([0]*10 + [1]*10),
            "sensitive": [{"age_group": "young"}]*20,
            "spurious": [{"hospital_type": "urban"}]*20,
        }
    }


def test_taxonomy_clinically_useful():
    layer = 0
    ld = _make_layer_data(layer)
    pr = _make_probe_results(layer, clin_auroc=0.75, sens_auroc=0.5, spur_auroc=0.5)
    cra = CRAFramework(ld, pr, DUMMY_CONFIG)
    label = cra.classify_taxonomy(layer)
    assert label == "clinically_useful"


def test_taxonomy_privacy_sensitive():
    layer = 0
    ld = _make_layer_data(layer)
    pr = _make_probe_results(layer, clin_auroc=0.5, sens_auroc=0.75, spur_auroc=0.5)
    cra = CRAFramework(ld, pr, DUMMY_CONFIG)
    label = cra.classify_taxonomy(layer)
    assert label == "privacy_sensitive"


def test_taxonomy_nuisance():
    layer = 0
    ld = _make_layer_data(layer)
    pr = _make_probe_results(layer, clin_auroc=0.5, sens_auroc=0.5, spur_auroc=0.5)
    cra = CRAFramework(ld, pr, DUMMY_CONFIG)
    label = cra.classify_taxonomy(layer)
    assert label == "nuisance"
