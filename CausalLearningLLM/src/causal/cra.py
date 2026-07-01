"""
CRA: Causal Representation Attribution — core orchestration class.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.causal.effects import (
    clinical_effect_probe,
    compute_cra_score,
    compute_pus,
    compute_rus,
    sensitive_leakage,
    spurious_decodability,
    task_loss_probe,
)
from src.causal.interventions import amplify_direction, remove_direction, scale_direction
from src.probing.directions import get_pca_direction, get_random_direction, get_mean_difference_direction
from src.representations.extraction import get_sensitive_labels, get_spurious_labels
from src.representations.geometry import cosine_similarity

logger = logging.getLogger(__name__)


class CRAFramework:
    """Orchestrates direction computation, interventions, and effect estimation."""

    def __init__(
        self,
        layer_data: Dict,
        probe_results: Dict,
        config: dict,
        sensitive_attr: str = "age_group",
        spurious_attr: str = "hospital_type",
    ) -> None:
        self.layer_data = layer_data
        self.probe_results = probe_results
        self.config = config
        self.sensitive_attr = sensitive_attr
        self.spurious_attr = spurious_attr

        cra = config.get("cra", {})
        self.beta = cra.get("beta_privacy", 1.0)
        self.eta = cra.get("eta_robustness", 1.0)
        self.gamma = cra.get("gamma_task", 0.5)
        self._dir_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Direction computation
    # ------------------------------------------------------------------

    def get_directions(self, layer: int) -> Dict[str, np.ndarray]:
        if layer in self._dir_cache:
            return self._dir_cache[layer]

        if layer not in self.layer_data:
            return {}

        Z = self.layer_data[layer]["Z"]
        y = self.layer_data[layer]["labels"]
        d = Z.shape[1]
        dirs: Dict[str, np.ndarray] = {}

        # From probes
        if layer in self.probe_results:
            for task, probe in self.probe_results[layer].get("probes", {}).items():
                v = probe.get_direction()
                if v is not None:
                    dirs[f"{task}_probe"] = v

        dirs["mean_difference"] = get_mean_difference_direction(Z, y)
        dirs["pca"] = get_pca_direction(Z)
        dirs["random"] = get_random_direction(d, seed=42 + layer)

        self._dir_cache[layer] = dirs
        return dirs

    # ------------------------------------------------------------------
    # Single intervention experiment
    # ------------------------------------------------------------------

    def run_intervention_experiment(
        self,
        layer: int,
        direction_type: str,
        intervention_type: str = "remove",
        lam: float = 1.0,
    ) -> Optional[Dict]:
        if layer not in self.layer_data or layer not in self.probe_results:
            return None

        data = self.layer_data[layer]
        Z = data["Z"]
        y = data["labels"]

        s_labels = get_sensitive_labels(self.layer_data, self.sensitive_attr, layer)
        c_labels = get_spurious_labels(self.layer_data, self.spurious_attr, layer)

        dirs = self.get_directions(layer)
        if direction_type not in dirs:
            return None
        v = dirs[direction_type]

        if intervention_type == "remove":
            Z_prime = remove_direction(Z, v, lam)
        elif intervention_type == "amplify":
            Z_prime = amplify_direction(Z, v, lam)
        elif intervention_type == "scale_zero":
            Z_prime = scale_direction(Z, v, 0.0)
        else:
            Z_prime = remove_direction(Z, v, lam)

        probes = self.probe_results[layer].get("probes", {})
        probe_y = probes.get("clinical")
        probe_s = probes.get("sensitive")
        probe_c = probes.get("spurious")

        if probe_y is None:
            return None

        ce_y = clinical_effect_probe(probe_y, Z, Z_prime, y)
        loss_orig = task_loss_probe(probe_y, Z, y)
        loss_int = task_loss_probe(probe_y, Z_prime, y)
        delta_task = loss_int - loss_orig

        m_orig = probe_y.score(Z, y)
        m_int = probe_y.score(Z_prime, y)

        leak_orig = leak_int = delta_leak = None
        if probe_s is not None and s_labels is not None:
            try:
                lo = sensitive_leakage(probe_s, Z, s_labels)
                li = sensitive_leakage(probe_s, Z_prime, s_labels)
                leak_orig = lo["auroc"]
                leak_int = li["auroc"]
                delta_leak = leak_orig - leak_int
            except Exception:
                pass

        spur_orig = spur_int = delta_spur = None
        if probe_c is not None and c_labels is not None:
            try:
                so = spurious_decodability(probe_c, Z, c_labels)
                si = spurious_decodability(probe_c, Z_prime, c_labels)
                spur_orig = so["auroc"]
                spur_int = si["auroc"]
                delta_spur = spur_orig - spur_int
            except Exception:
                pass

        pus = compute_pus(delta_leak or 0.0, delta_task, self.beta)
        rus = compute_rus(delta_spur or 0.0, delta_task, self.eta)
        cra_score = compute_cra_score(
            ce_y, delta_leak or 0.0, delta_spur or 0.0, delta_task, gamma=self.gamma
        )

        return {
            "layer": layer,
            "direction_type": direction_type,
            "intervention_type": intervention_type,
            "lambda": lam,
            "clinical_effect_ce_y": ce_y,
            "pre_task_loss": loss_orig,
            "post_task_loss": loss_int,
            "delta_task_loss": delta_task,
            "pre_clinical_acc": m_orig["accuracy"],
            "post_clinical_acc": m_int["accuracy"],
            "pre_clinical_auroc": m_orig["auroc"],
            "post_clinical_auroc": m_int["auroc"],
            "pre_sensitive_leakage": leak_orig,
            "post_sensitive_leakage": leak_int,
            "delta_leak": delta_leak,
            "pre_spurious_score": spur_orig,
            "post_spurious_score": spur_int,
            "delta_spur": delta_spur,
            "privacy_utility_score": pus,
            "robustness_utility_score": rus,
            "cra_score": cra_score,
        }

    # ------------------------------------------------------------------
    # Full sweep
    # ------------------------------------------------------------------

    def run_full_intervention_study(self, output_dir: Optional[str] = None) -> pd.DataFrame:
        cra = self.config.get("cra", {})
        layers = cra.get("layer_selection", list(self.layer_data.keys()))
        direction_types = cra.get("direction_types", ["clinical_probe", "sensitive_probe", "random"])
        lambdas = cra.get("intervention_strengths", [0.5, 1.0, 2.0])

        rows = []
        for l in layers:
            for d_type in direction_types:
                for lam in lambdas:
                    res = self.run_intervention_experiment(l, d_type, "remove", lam)
                    if res:
                        rows.append(res)

        df = pd.DataFrame(rows)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, "cra_intervention_results.csv"), index=False)
        return df

    # ------------------------------------------------------------------
    # Direction geometry
    # ------------------------------------------------------------------

    def compute_direction_geometry(self, output_dir: Optional[str] = None) -> pd.DataFrame:
        rows = []
        for l in self.layer_data:
            dirs = self.get_directions(l)
            row: Dict = {"layer": l}
            for n1 in ["clinical_probe", "sensitive_probe", "spurious_probe"]:
                for n2 in ["clinical_probe", "sensitive_probe", "spurious_probe"]:
                    if n1 in dirs and n2 in dirs:
                        row[f"cos_{n1[:3]}_{n2[:3]}"] = cosine_similarity(dirs[n1], dirs[n2])
            row["cos_y_s"] = row.get("cos_cli_sen", float("nan"))
            row["cos_y_c"] = row.get("cos_cli_spu", float("nan"))
            row["cos_s_c"] = row.get("cos_sen_spu", float("nan"))
            rows.append(row)

        df = pd.DataFrame(rows)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, "direction_geometry.csv"), index=False)
        return df

    # ------------------------------------------------------------------
    # Taxonomy
    # ------------------------------------------------------------------

    def classify_taxonomy(self, layer: int) -> str:
        thr = self.config.get("cra", {}).get("taxonomy_thresholds", {})
        tc = thr.get("clinical", 0.6)
        ts = thr.get("sensitive", 0.55)
        tsp = thr.get("spurious", 0.55)

        if layer not in self.probe_results:
            return "nuisance"

        metrics = self.probe_results[layer].get("metrics", {})
        clin = metrics.get("clinical", {}).get("auroc", 0.5)
        sens = metrics.get("sensitive", {}).get("auroc", 0.5)
        spur = metrics.get("spurious", {}).get("auroc", 0.5)

        ic, is_, isp = clin >= tc, sens >= ts, spur >= tsp

        if ic and is_ and isp:
            return "highly_entangled"
        if ic and is_:
            return "mixed_clinical_sensitive"
        if ic and isp:
            return "mixed_clinical_spurious"
        if is_ and isp:
            return "mixed_sensitive_spurious"
        if ic:
            return "clinically_useful"
        if is_:
            return "privacy_sensitive"
        if isp:
            return "spurious"
        return "nuisance"

    def build_taxonomy_table(self, output_dir: Optional[str] = None) -> pd.DataFrame:
        rows = []
        for l in self.layer_data:
            dirs = self.get_directions(l)
            metrics = self.probe_results.get(l, {}).get("metrics", {})
            clin = metrics.get("clinical", {}).get("auroc", 0.5)
            sens = metrics.get("sensitive", {}).get("auroc", 0.5)
            spur = metrics.get("spurious", {}).get("auroc", 0.5)
            label = self.classify_taxonomy(l)

            cos_ys = cosine_similarity(dirs["clinical_probe"], dirs["sensitive_probe"]) if "clinical_probe" in dirs and "sensitive_probe" in dirs else float("nan")
            cos_yc = cosine_similarity(dirs["clinical_probe"], dirs["spurious_probe"]) if "clinical_probe" in dirs and "spurious_probe" in dirs else float("nan")
            cos_sc = cosine_similarity(dirs["sensitive_probe"], dirs["spurious_probe"]) if "sensitive_probe" in dirs and "spurious_probe" in dirs else float("nan")

            for d_type in dirs:
                rows.append({
                    "layer": l, "direction_type": d_type,
                    "clinical_score": clin, "sensitive_score": sens, "spurious_score": spur,
                    "cos_y_s": cos_ys, "cos_y_c": cos_yc, "cos_s_c": cos_sc,
                    "taxonomy_label": label,
                })

        df = pd.DataFrame(rows)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, "representation_taxonomy.csv"), index=False)
        return df
