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


def verify_causal_claims(
    layer_data: Dict,
    probe_results: Dict,
    causal_directions: Optional[Dict] = None,  # {layer: v_causal ndarray}
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Verify causal claims with PM, NDE, HSIC, causal-direction separation.

    For each layer computes:
      - PM (proportion mediated) via run_mediation_analysis
      - NDE_nonzero: whether natural direct effect is meaningfully non-zero
      - cos_causal_corr: cosine between causal and correlational directions
      - hsic_reduction_pct: % HSIC reduction after direction removal
      - conditional_leakage_before / _after

    Parameters
    ----------
    layer_data : Dict
        Mapping layer -> {"Z": ndarray, "labels": array, ...}
    probe_results : Dict
        Mapping layer -> {"probes": {"sensitive": probe, ...}}
    causal_directions : Optional[Dict]
        Mapping layer -> v_causal (ndarray).  If None, cos_causal_corr is NaN.
    output_dir : Optional[str]
        If provided, saves causal_claims_verification.csv there.

    Returns
    -------
    pd.DataFrame with columns:
        layer, PM, NDE_nonzero, cos_causal_corr,
        hsic_reduction_pct, conditional_leakage_before, conditional_leakage_after
    """
    # Lazy imports so the function only errors if the sub-modules are missing,
    # not when verification.py is first imported.
    try:
        from src.causal.mediation import run_mediation_analysis
    except ImportError:
        run_mediation_analysis = None
        logger.warning("src.causal.mediation not found; PM/NDE will be NaN")

    try:
        from src.causal.independence import hsic_delta
    except ImportError:
        hsic_delta = None
        logger.warning("src.causal.independence not found; HSIC metrics will be NaN")

    try:
        from src.privacy.leakage import conditional_leakage
    except ImportError:
        conditional_leakage = None
        logger.warning("src.privacy.leakage not found; leakage metrics will be NaN")

    try:
        from src.representations.extraction import get_sensitive_labels
    except ImportError:
        get_sensitive_labels = None
        logger.warning("src.representations.extraction not found; some metrics unavailable")

    rows = []

    for l, data in layer_data.items():
        if l not in probe_results:
            continue

        probes = probe_results[l].get("probes", {})
        probe_s = probes.get("sensitive")
        if probe_s is None:
            continue

        Z = data["Z"]
        y = data.get("labels")

        # Sensitive labels
        s_labels = None
        if get_sensitive_labels is not None:
            s_labels = get_sensitive_labels(layer_data, "sensitive", l)

        v_corr = probe_s.get_direction()

        # ------------------------------------------------------------------
        # PM and NDE via mediation analysis
        # ------------------------------------------------------------------
        pm = float("nan")
        nde_nonzero = False
        if run_mediation_analysis is not None and y is not None and s_labels is not None:
            try:
                med_result = run_mediation_analysis(Z, s_labels, y)
                pm = float(med_result.get("PM", float("nan")))
                nde = float(med_result.get("NDE", 0.0))
                nde_nonzero = abs(nde) > 0.05
            except Exception as exc:
                logger.warning("Layer %s: run_mediation_analysis failed: %s", l, exc)

        # ------------------------------------------------------------------
        # Cosine between causal direction and correlational direction
        # ------------------------------------------------------------------
        cos_causal_corr = float("nan")
        if causal_directions is not None and l in causal_directions and v_corr is not None:
            v_causal = causal_directions[l]
            try:
                cos_causal_corr = float(cosine_similarity(v_causal, v_corr))
                cos_causal_corr = abs(cos_causal_corr)
            except Exception as exc:
                logger.warning("Layer %s: cosine_similarity failed: %s", l, exc)

        # ------------------------------------------------------------------
        # HSIC before / after direction removal
        # ------------------------------------------------------------------
        hsic_reduction_pct = float("nan")
        if hsic_delta is not None and s_labels is not None and v_corr is not None:
            try:
                hsic_before, hsic_after = hsic_delta(Z, s_labels, v_corr)
                if abs(hsic_before) > 1e-12:
                    hsic_reduction_pct = 100.0 * (hsic_before - hsic_after) / abs(hsic_before)
                else:
                    hsic_reduction_pct = 0.0
            except Exception as exc:
                logger.warning("Layer %s: hsic_delta failed: %s", l, exc)

        # ------------------------------------------------------------------
        # Conditional leakage before / after removing sensitive direction
        # ------------------------------------------------------------------
        cond_leak_before = float("nan")
        cond_leak_after = float("nan")
        if (
            conditional_leakage is not None
            and y is not None
            and s_labels is not None
            and v_corr is not None
        ):
            try:
                result_before = conditional_leakage(probe_s, Z, s_labels, y)
                cond_leak_before = float(
                    result_before.get("conditional_leakage_auroc", float("nan"))
                )
                Z_prime = remove_direction(Z, v_corr, lam=1.0)
                result_after = conditional_leakage(probe_s, Z_prime, s_labels, y)
                cond_leak_after = float(
                    result_after.get("conditional_leakage_auroc", float("nan"))
                )
            except Exception as exc:
                logger.warning("Layer %s: conditional_leakage failed: %s", l, exc)

        rows.append({
            "layer": l,
            "PM": pm,
            "NDE_nonzero": nde_nonzero,
            "cos_causal_corr": cos_causal_corr,
            "hsic_reduction_pct": hsic_reduction_pct,
            "conditional_leakage_before": cond_leak_before,
            "conditional_leakage_after": cond_leak_after,
        })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Log which claims are empirically supported
    # ------------------------------------------------------------------
    if not df.empty:
        n = len(df)
        supported = {
            "PM > 0 (mediation exists)": int((df["PM"] > 0.0).sum()),
            "NDE nonzero": int(df["NDE_nonzero"].sum()),
            "HSIC reduction > 10%": int((df["hsic_reduction_pct"] > 10.0).sum()),
            "Cond. leakage reduced": int(
                (df["conditional_leakage_after"] < df["conditional_leakage_before"]).sum()
            ),
        }
        logger.info("Causal claims verification summary (%d layers):", n)
        for claim, count in supported.items():
            logger.info("  %-35s  %d / %d layers (%.0f%%)", claim, count, n, 100 * count / n)

        if not df["cos_causal_corr"].isna().all():
            mean_sep = df["cos_causal_corr"].mean()
            logger.info(
                "  %-35s  %.4f (lower = more distinct)",
                "Mean cos(v_causal, v_corr)", mean_sep,
            )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "causal_claims_verification.csv")
        df.to_csv(path, index=False)
        logger.info("Saved causal claims verification → %s", path)

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
