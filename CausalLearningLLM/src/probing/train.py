"""Probe training pipeline for clinical, sensitive, and spurious tasks."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.probing.probes import LogisticRegressionProbe
from src.representations.extraction import get_sensitive_labels, get_spurious_labels

logger = logging.getLogger(__name__)


def train_probes(
    layer_data: Dict,
    layers: List[int],
    sensitive_attr: str,
    spurious_attr: str,
    config: dict,
    output_dir: Optional[str] = None,
) -> Dict:
    """Train clinical / sensitive / spurious probes for each layer.

    Returns
    -------
    dict
        ``{layer: {"probes": {...}, "metrics": {...}}}``
    """
    results: Dict = {}
    all_metrics = []

    probe_cfg = config.get("probes", {})
    test_size = probe_cfg.get("test_size", 0.3)
    rs = probe_cfg.get("random_state", 42)
    C = probe_cfg.get("regularization", 1.0)
    max_iter = probe_cfg.get("max_iter", 500)

    for l in layers:
        if l not in layer_data:
            continue

        data = layer_data[l]
        Z = data["Z"]
        y = data["labels"]
        s_labels = get_sensitive_labels(layer_data, sensitive_attr, l)
        c_labels = get_spurious_labels(layer_data, spurious_attr, l)

        layer_result: Dict = {"layer": l, "probes": {}, "metrics": {}}

        for task, lbl in [("clinical", y), ("sensitive", s_labels), ("spurious", c_labels)]:
            if lbl is None:
                continue
            n_classes = len(np.unique(lbl))
            if n_classes < 2:
                logger.warning("Layer %d task %s: only 1 class, skipping.", l, task)
                continue

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Z, lbl, test_size=test_size, random_state=rs,
                    stratify=lbl if n_classes < 20 else None,
                )
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(Z, lbl, test_size=test_size, random_state=rs)

            probe = LogisticRegressionProbe(C=C, max_iter=max_iter, random_state=rs)
            probe.fit(X_tr, y_tr)
            metrics = probe.score(X_te, y_te)

            layer_result["probes"][task] = probe
            layer_result["metrics"][task] = metrics
            all_metrics.append({"layer": l, "task": task, **metrics})

            logger.info("Layer %d | %-10s | acc=%.3f auroc=%.3f", l, task, metrics["accuracy"], metrics["auroc"])

            if output_dir:
                probe.save(os.path.join(output_dir, f"layer_{l}_{task}_probe.pkl"))

        results[l] = layer_result

    if output_dir and all_metrics:
        pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "probe_metrics.csv"), index=False)

    return results


def extract_all_directions(probe_results: Dict) -> Dict:
    """Return {layer: {task: direction_vector}} from probe results."""
    directions: Dict = {}
    for l, res in probe_results.items():
        directions[l] = {}
        for task, probe in res.get("probes", {}).items():
            v = probe.get_direction()
            if v is not None:
                directions[l][task] = v
    return directions
