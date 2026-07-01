"""Smoke tests for visualization functions."""

import pytest, sys, os, tempfile
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_framework_diagram_smoke(tmp_dir):
    from src.visualization.framework_diagram import plot_framework_diagram
    plot_framework_diagram(tmp_dir, dpi=72)
    figs = [f for f in os.listdir(os.path.join(tmp_dir, "figures", "png")) if f.endswith(".png")]
    assert len(figs) > 0


def test_layer_information_smoke(tmp_dir):
    from src.visualization.layerwise import plot_layer_information
    df = pd.DataFrame({
        "layer": [0, 1, 2],
        "clinical_auroc": [0.7, 0.75, 0.8],
        "sensitive_auroc": [0.6, 0.65, 0.62],
        "spurious_auroc": [0.55, 0.58, 0.6],
    })
    plot_layer_information(df, tmp_dir, dpi=72)
    figs = [f for f in os.listdir(os.path.join(tmp_dir, "figures", "png")) if "layer" in f]
    assert len(figs) > 0


def test_pareto_smoke(tmp_dir):
    from src.visualization.pareto import plot_privacy_utility_pareto
    df = pd.DataFrame({
        "direction_type": ["cra", "cra", "random", "random"],
        "lambda": [0.5, 1.0, 0.5, 1.0],
        "delta_leak": [0.1, 0.2, 0.05, 0.08],
        "delta_task_loss": [0.01, 0.03, 0.005, 0.01],
        "privacy_utility_score": [0.09, 0.17, 0.045, 0.07],
    })
    plot_privacy_utility_pareto(df, tmp_dir, dpi=72)
