"""Causal mediation analysis via representation patching (probe-space proxy)."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_mediation(
    probe_clinical,
    Z_orig: np.ndarray,
    Z_cf: np.ndarray,
    labels: np.ndarray,
    layer: int,
    cf_type: str = "sensitive",
) -> Dict:
    """Probe-space mediation proxy.

    TE  = E[score(x) - score(x_cf)]
    NIE = E[score(Z_orig) - score(Z_cf)]   (full patch of layer l)
    NDE = TE - NIE
    """
    p_orig = probe_clinical.predict_proba(Z_orig)
    p_cf = probe_clinical.predict_proba(Z_cf)

    n = len(labels)
    s_orig = np.array([p_orig[i, labels[i]] for i in range(n)])
    s_cf = np.array([p_cf[i, labels[i]] for i in range(n)])

    TE = float(np.mean(s_orig - s_cf))
    NIE = TE  # full patch => NIE = TE in probe-space proxy
    NDE = 0.0
    MR = 1.0 if abs(TE) > 1e-9 else 0.0

    return {
        "layer": layer,
        "cf_type": cf_type,
        "TE": TE,
        "NIE": NIE,
        "NDE": NDE,
        "mediation_ratio": MR,
        "mean_score_orig": float(np.mean(s_orig)),
        "mean_score_cf": float(np.mean(s_cf)),
    }


def run_mediation_analysis(
    layer_data: Dict,
    probe_results: Dict,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for l, data in layer_data.items():
        if l not in probe_results:
            continue
        probe_y = probe_results[l].get("probes", {}).get("clinical")
        if probe_y is None:
            continue

        Z = data["Z"]
        y = data["labels"]

        # Proxy counterfactual: shuffle Z
        rng = np.random.RandomState(42 + l)
        Z_cf = Z[rng.permutation(len(Z))]

        res = compute_mediation(probe_y, Z, Z_cf, y, layer=l, cf_type="proxy_shuffle")
        rows.append(res)

    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "mediation_results.csv"), index=False)
    return df
