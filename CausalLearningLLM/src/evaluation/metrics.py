"""Consolidated evaluation metric computation for CRA experiments."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.causal.effects import (
    sensitive_leakage,
    spurious_decodability,
    task_loss_probe,
    compute_pus,
    compute_rus,
)
from src.representations.extraction import get_sensitive_labels, get_spurious_labels

logger = logging.getLogger(__name__)


def compute_layer_information_profile(
    layer_data: Dict,
    probe_results: Dict,
    sensitive_attr: str,
    spurious_attr: str,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """For each layer compute Clin, Leak, Spur metrics."""
    import os

    rows = []
    for l, data in layer_data.items():
        if l not in probe_results:
            continue
        metrics = probe_results[l].get("metrics", {})
        Z = data["Z"]
        y = data["labels"]

        clin = metrics.get("clinical", {})
        sens = metrics.get("sensitive", {})
        spur = metrics.get("spurious", {})

        rows.append({
            "layer": l,
            "clinical_accuracy": clin.get("accuracy", float("nan")),
            "clinical_auroc": clin.get("auroc", float("nan")),
            "sensitive_accuracy": sens.get("accuracy", float("nan")),
            "sensitive_auroc": sens.get("auroc", float("nan")),
            "spurious_accuracy": spur.get("accuracy", float("nan")),
            "spurious_auroc": spur.get("auroc", float("nan")),
            "clinical_ce": clin.get("cross_entropy", float("nan")),
            "sensitive_ce": sens.get("cross_entropy", float("nan")),
            "spurious_ce": spur.get("cross_entropy", float("nan")),
        })

    df = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "layer_information.csv"), index=False)
    return df


def compute_privacy_utility_curve(
    intervention_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    import os

    if intervention_df.empty or "delta_leak" not in intervention_df.columns:
        return pd.DataFrame()

    df = intervention_df[["direction_type", "lambda", "delta_leak", "delta_task_loss", "privacy_utility_score"]].dropna()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "privacy_utility_curve.csv"), index=False)
    return df


def compute_robustness_spuriousness_curve(
    intervention_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    import os

    if intervention_df.empty or "delta_spur" not in intervention_df.columns:
        return pd.DataFrame()

    df = intervention_df[["direction_type", "lambda", "delta_spur", "delta_task_loss", "robustness_utility_score"]].dropna()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "robustness_spuriousness_curve.csv"), index=False)
    return df
