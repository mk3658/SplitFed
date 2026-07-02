"""
Causal Mediation Analysis for Representation Attribution (CRA).

Mediation decomposition (Pearl 2001):

  TE  = Total Effect = full sensitive leakage captured by INLP (k rounds).
        Uses multi-round INLP as the upper bound of linear S dependence.

  NIE = Natural Indirect Effect = leakage reduction from erasing primary
        direction v at strength lam.  Effect routed through v (single vector).

  NDE = TE - NIE  (at lam=1) = leakage reduction from INLP rounds 2..k.
        Nonlinear/higher-order S information NOT captured by primary v.
        NDE > 0  => v is incomplete mediator; multi-direction removal needed.
        NDE = 0  => v is sufficient mediator (LRH validated at this layer).

  PM  = NIE / TE  = proportion of total S effect mediated through primary v.

This replaces the previous trivial formulation where NIE = TE always
(NDE = 0 by construction from a single-direction full erasure).
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.probing.metrics import probe_auroc as _probe_auroc_canonical
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


_probe_auroc = _probe_auroc_canonical  # delegated to src/probing/metrics.py


def _project_out(Z: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
    v_unit = v / (np.linalg.norm(v) + 1e-12)
    return Z - lam * np.outer(Z @ v_unit, v_unit)


def _inlp_total_effect(Z: np.ndarray, s_labels: np.ndarray,
                        probe_sensitive, n_rounds: int = 5) -> float:
    """AUROC reduction from k-round INLP = Total Effect upper bound."""
    d = Z.shape[1]
    Z_curr = Z.copy()
    auroc_base = _probe_auroc(probe_sensitive, Z, s_labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_rounds):
            clf = LogisticRegression(C=1.0, max_iter=2000,
                                     class_weight="balanced", random_state=42)
            try:
                clf.fit(Z_curr, s_labels)
            except Exception:
                break
            W = clf.coef_
            v = W[0] if W.shape[0] == 1 else np.linalg.svd(W, full_matrices=False)[2][0]
            v = v / (np.linalg.norm(v) + 1e-12)
            Z_curr = Z_curr - np.outer(Z_curr @ v, v)
    auroc_after = _probe_auroc(probe_sensitive, Z_curr, s_labels)
    return auroc_base - auroc_after, auroc_base, Z_curr


def compute_mediation(
    probe_clinical,
    probe_sensitive,
    Z: np.ndarray,
    v_direction: np.ndarray,
    y_clinical: np.ndarray,
    s_labels: np.ndarray,
    layer: int,
    lambda_values: List[float] = None,
    direction_type: str = "correlational",
    n_inlp_rounds: int = 5,
) -> List[Dict]:
    if lambda_values is None:
        lambda_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    auroc_clin_base = _probe_auroc(probe_clinical, Z, y_clinical)
    auroc_sens_base = _probe_auroc(probe_sensitive, Z, s_labels)

    # Total Effect: multi-round INLP captures full linear S dependence
    TE, _, Z_inlp = _inlp_total_effect(Z, s_labels, probe_sensitive, n_inlp_rounds)
    auroc_clin_inlp = _probe_auroc(probe_clinical, Z_inlp, y_clinical)
    TE_clin = auroc_clin_base - auroc_clin_inlp

    rows = []
    for lam in lambda_values:
        Z_int = _project_out(Z, v_direction, lam)

        auroc_clin_int = _probe_auroc(probe_clinical,  Z_int, y_clinical)
        auroc_sens_int = _probe_auroc(probe_sensitive, Z_int, s_labels)

        NIE        = auroc_sens_base - auroc_sens_int
        delta_clin = auroc_clin_base - auroc_clin_int

        # NDE at lam=1: residual S effect not captured by primary direction
        NDE = float(TE - NIE) if abs(lam - 1.0) < 0.01 else 0.0
        PM  = float(NIE / (abs(TE) + 1e-9)) if abs(TE) > 1e-6 else 0.0
        PM_clin = float(delta_clin / (abs(TE_clin) + 1e-9)) if abs(TE_clin) > 1e-6 else 0.0

        rows.append({
            "layer":          layer,
            "lambda":         lam,
            "direction_type": direction_type,
            "TE":             float(TE),
            "NIE":            float(NIE),
            "NDE":            NDE,
            "proportion_mediated_privacy":  PM,
            "proportion_mediated_clinical": PM_clin,
            "auroc_sensitive_base": float(auroc_sens_base),
            "auroc_sensitive_int":  float(auroc_sens_int),
            "auroc_clinical_base":  float(auroc_clin_base),
            "auroc_clinical_int":   float(auroc_clin_int),
            "delta_leak":  float(NIE),
            "delta_clin":  float(delta_clin),
            "PUS":         float(NIE - delta_clin),
        })

    return rows


def run_mediation_analysis(
    layer_data: Dict,
    probe_results: Dict,
    lambda_values: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
    causal_directions: Optional[Dict] = None,
) -> pd.DataFrame:
    if lambda_values is None:
        lambda_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    all_rows: list = []

    for l, data in sorted(layer_data.items()):
        if l not in probe_results:
            continue
        probes = probe_results[l].get("probes", {})
        probe_clin = probes.get("clinical")
        probe_sens = probes.get("sensitive")
        if probe_clin is None or probe_sens is None:
            continue

        Z    = data["Z"]
        meta = data.get("meta", {})

        y_clin = np.array(meta.get("labels", data.get("labels", [])))
        if len(y_clin) == 0:
            continue

        sens_raw = meta.get("sensitive", [])
        if sens_raw and isinstance(sens_raw[0], dict):
            key = list(sens_raw[0].keys())[0]
            raw_vals = [d.get(key, "") for d in sens_raw]
        else:
            raw_vals = list(sens_raw) if sens_raw else list(y_clin)
        s_enc = LabelEncoder().fit_transform(raw_vals)

        # Correlational direction from probe weights
        W = probe_sens.clf.coef_
        if W.shape[0] == 1:
            v_corr = W[0] / (np.linalg.norm(W[0]) + 1e-12)
        else:
            _, _, Vt = np.linalg.svd(W, full_matrices=False)
            v_corr = Vt[0]

        rows_corr = compute_mediation(
            probe_clin, probe_sens, Z, v_corr,
            y_clin, s_enc, layer=l,
            lambda_values=lambda_values,
            direction_type="correlational",
        )
        all_rows.extend(rows_corr)

        if causal_directions and l in causal_directions:
            rows_caus = compute_mediation(
                probe_clin, probe_sens, Z, causal_directions[l],
                y_clin, s_enc, layer=l,
                lambda_values=lambda_values,
                direction_type="causal",
            )
            all_rows.extend(rows_caus)

        lam1 = [r for r in rows_corr if abs(r["lambda"] - 1.0) < 0.01]
        if lam1:
            r = lam1[0]
            logger.info(
                "Layer %d | TE=%.4f NIE=%.4f NDE=%.4f PM=%.3f [%s]",
                l, r["TE"], r["NIE"], r["NDE"],
                r["proportion_mediated_privacy"],
                "causal+corr" if (causal_directions and l in causal_directions) else "corr",
            )

    df = pd.DataFrame(all_rows)
    if output_dir and not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "mediation_results.csv"), index=False)

    return df
