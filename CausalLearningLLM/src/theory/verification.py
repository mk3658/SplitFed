"""Empirical verification of CRA theoretical claims."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.causal.effects import sensitive_leakage, spurious_decodability, task_loss_probe
from src.causal.interventions import remove_direction
from src.causal.counterfactual_effects import compute_counterfactual_robustness
from src.representations.geometry import cosine_similarity

logger = logging.getLogger(__name__)


def verify_claim1_leakage_reduction(
    layer_data: Dict,
    probe_results: Dict,
    sensitive_attr: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Claim 1: Sensitive direction intervention reduces sensitive leakage."""
    from src.representations.extraction import get_sensitive_labels

    rows = []
    for l, data in layer_data.items():
        if l not in probe_results:
            continue
        probes = probe_results[l].get("probes", {})
        probe_s = probes.get("sensitive")
        if probe_s is None:
            continue

        Z = data["Z"]
        s_labels = get_sensitive_labels(layer_data, sensitive_attr, l)
        if s_labels is None:
            continue

        v = probe_s.get_direction()
        if v is None:
            continue

        leak_before = sensitive_leakage(probe_s, Z, s_labels)["auroc"]
        Z_prime = remove_direction(Z, v, lam=1.0)
        leak_after = sensitive_leakage(probe_s, Z_prime, s_labels)["auroc"]

        rows.append({
            "layer": l,
            "leak_before": leak_before,
            "leak_after": leak_after,
            "reduction": leak_before - leak_after,
            "claim_supported": leak_after <= leak_before,
        })

    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "theorem_claim1_leakage.csv"), index=False)
    return df


def verify_claim2_orthogonality_utility(
    layer_data: Dict,
    probe_results: Dict,
    sensitive_attr: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Claim 2: Utility preservation depends on orthogonality between v_y and v_s."""
    from src.representations.extraction import get_sensitive_labels

    rows = []
    for l, data in layer_data.items():
        if l not in probe_results:
            continue
        probes = probe_results[l].get("probes", {})
        probe_y = probes.get("clinical")
        probe_s = probes.get("sensitive")
        if probe_y is None or probe_s is None:
            continue

        Z = data["Z"]
        y = data["labels"]
        v_y = probe_y.get_direction()
        v_s = probe_s.get_direction()
        if v_y is None or v_s is None:
            continue

        cos_ys = abs(cosine_similarity(v_y, v_s))
        Z_prime = remove_direction(Z, v_s, lam=1.0)
        loss_before = task_loss_probe(probe_y, Z, y)
        loss_after = task_loss_probe(probe_y, Z_prime, y)
        delta_task = loss_after - loss_before

        rows.append({
            "layer": l,
            "cos_y_s": cos_ys,
            "delta_task_loss": delta_task,
            "loss_before": loss_before,
            "loss_after": loss_after,
        })

    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "theorem_claim2_orthogonality_utility.csv"), index=False)
    return df


def verify_claim3_spurious_robustness(
    layer_data: Dict,
    probe_results: Dict,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Claim 3: Spurious direction removal improves counterfactual robustness."""
    rows = []
    for l, data in layer_data.items():
        if l not in probe_results:
            continue
        probes = probe_results[l].get("probes", {})
        probe_y = probes.get("clinical")
        probe_c = probes.get("spurious")
        if probe_y is None or probe_c is None:
            continue

        Z = data["Z"]
        y = data["labels"]
        v_c = probe_c.get_direction()
        if v_c is None:
            continue

        # Proxy counterfactual
        rng = np.random.RandomState(42 + l)
        Z_cf = Z[rng.permutation(len(Z))]
        labels_enc = probe_y.label_encoder.transform(y)

        rob_before = compute_counterfactual_robustness(probe_y, Z, Z_cf, labels_enc)
        Z_prime = remove_direction(Z, v_c, lam=1.0)
        Z_cf_prime = remove_direction(Z_cf, v_c, lam=1.0)
        rob_after = compute_counterfactual_robustness(probe_y, Z_prime, Z_cf_prime, labels_enc)

        rows.append({
            "layer": l,
            "robustness_before": rob_before["prediction_consistency"],
            "robustness_after": rob_after["prediction_consistency"],
            "delta_robust": rob_after["prediction_consistency"] - rob_before["prediction_consistency"],
            "claim_supported": rob_after["prediction_consistency"] >= rob_before["prediction_consistency"],
        })

    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "theorem_claim3_spurious_robustness.csv"), index=False)
    return df


def verify_claim4_random_projection(
    layer_data: Dict,
    probe_results: Dict,
    n_random: int = 200,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Claim 4: Random directions have small expected alignment with sensitive subspace."""
    rows = []
    rng = np.random.RandomState(0)

    for l, data in layer_data.items():
        if l not in probe_results:
            continue
        probe_s = probe_results[l].get("probes", {}).get("sensitive")
        if probe_s is None:
            continue

        d = data["Z"].shape[1]
        v_s = probe_s.get_direction()
        if v_s is None:
            continue

        projections = []
        for _ in range(n_random):
            r = rng.randn(d)
            r /= np.linalg.norm(r) + 1e-12
            projections.append(float((r @ v_s) ** 2))

        k = 1  # sensitive subspace dimension
        theoretical_mean = k / d

        rows.append({
            "layer": l,
            "empirical_mean_sq_proj": float(np.mean(projections)),
            "theoretical_mean": theoretical_mean,
            "k": k,
            "d": d,
            "ratio": float(np.mean(projections)) / (theoretical_mean + 1e-12),
        })

    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "theorem_claim4_random_projection.csv"), index=False)
    return df


def verify_claim5_tradeoff(
    intervention_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Claim 5: CRA achieves better PUS/RUS than random/correlation-only baselines."""
    if intervention_df.empty:
        return pd.DataFrame()

    summary = (
        intervention_df.groupby("direction_type")[["privacy_utility_score", "robustness_utility_score"]]
        .mean()
        .reset_index()
    )
    summary["is_cra"] = summary["direction_type"].isin(["clinical_probe", "sensitive_probe", "spurious_probe"])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary.to_csv(os.path.join(output_dir, "theorem_claim5_tradeoff.csv"), index=False)
    return summary


def verify_claim6_conditional_leakage(
    layer_data: Dict,
    probe_results: Dict,
    sensitive_attr: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Claim 6: Conditional leakage reveals residual sensitive info beyond Y."""
    from src.representations.extraction import get_sensitive_labels
    from src.privacy.leakage import conditional_leakage, estimate_leakage

    rows = []
    for l, data in layer_data.items():
        if l not in probe_results:
            continue
        probe_s = probe_results[l].get("probes", {}).get("sensitive")
        if probe_s is None:
            continue

        Z = data["Z"]
        y = data["labels"]
        s_labels = get_sensitive_labels(layer_data, sensitive_attr, l)
        if s_labels is None:
            continue

        marg = estimate_leakage(probe_s, Z, s_labels)["auroc"]
        cond = conditional_leakage(probe_s, Z, s_labels, y)["conditional_leakage_auroc"]

        rows.append({
            "layer": l,
            "marginal_leakage": marg,
            "conditional_leakage": cond,
            "residual": cond - (1.0 / len(np.unique(s_labels))),
        })

    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "theorem_claim6_conditional_leakage.csv"), index=False)
    return df


def run_all_verifications(
    layer_data: Dict,
    probe_results: Dict,
    intervention_df: pd.DataFrame,
    sensitive_attr: str,
    output_dir: str,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    return {
        "claim1": verify_claim1_leakage_reduction(layer_data, probe_results, sensitive_attr, output_dir),
        "claim2": verify_claim2_orthogonality_utility(layer_data, probe_results, sensitive_attr, output_dir),
        "claim3": verify_claim3_spurious_robustness(layer_data, probe_results, output_dir),
        "claim4": verify_claim4_random_projection(layer_data, probe_results, output_dir=output_dir),
        "claim5": verify_claim5_tradeoff(intervention_df, output_dir),
        "claim6": verify_claim6_conditional_leakage(layer_data, probe_results, sensitive_attr, output_dir),
    }
