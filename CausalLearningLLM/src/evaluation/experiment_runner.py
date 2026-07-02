"""High-level experiment runner that sequences all CRA experiments."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full CRA experimental pipeline."""

    def __init__(self, config: dict, output_dir: str) -> None:
        self.config = config
        self.output_dir = output_dir
        self.results: Dict = {}
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        layer_data: Dict,
        probe_results: Dict,
        sensitive_attr: str,
        spurious_attr: str,
    ) -> Dict:
        from src.causal.cra import CRAFramework
        from src.evaluation.metrics import (
            compute_layer_information_profile,
            compute_privacy_utility_curve,
            compute_robustness_spuriousness_curve,
        )
        from src.mediation.mediation import run_mediation_analysis
        from src.theory.verification import run_all_verifications

        metrics_dir = os.path.join(self.output_dir, "metrics")
        interventions_dir = os.path.join(self.output_dir, "interventions")
        mediation_dir = os.path.join(self.output_dir, "mediation")
        stats_dir = os.path.join(self.output_dir, "statistics")

        t0 = time.time()

        logger.info("=== Exp 1: Layer information profile ===")
        layer_info = compute_layer_information_profile(
            layer_data, probe_results, sensitive_attr, spurious_attr, metrics_dir
        )
        self.results["layer_information"] = layer_info

        logger.info("=== Exp 2-3: CRA interventions and taxonomy ===")
        cra = CRAFramework(layer_data, probe_results, self.config, sensitive_attr, spurious_attr)
        intervention_df = cra.run_full_intervention_study(interventions_dir)
        self.results["interventions"] = intervention_df

        geo_df = cra.compute_direction_geometry(metrics_dir)
        self.results["direction_geometry"] = geo_df

        taxonomy_df = cra.build_taxonomy_table(metrics_dir)
        self.results["taxonomy"] = taxonomy_df

        logger.info("=== Exp 4: Privacy-utility and robustness curves ===")
        pu_df = compute_privacy_utility_curve(intervention_df, metrics_dir)
        rs_df = compute_robustness_spuriousness_curve(intervention_df, metrics_dir)
        self.results["privacy_utility"] = pu_df
        self.results["robustness_spuriousness"] = rs_df

        logger.info("=== Exp 5: Mediation analysis ===")
        med_df = run_mediation_analysis(layer_data, probe_results, output_dir=mediation_dir)
        self.results["mediation"] = med_df

        logger.info("=== Exp 6: Theorem verification ===")
        theorem_results = run_all_verifications(
            layer_data, probe_results, intervention_df,
            sensitive_attr, stats_dir
        )
        self.results["theorems"] = theorem_results

        elapsed = time.time() - t0
        summary = {"runtime_seconds": elapsed, "n_layers": len(layer_data)}
        with open(os.path.join(self.output_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("All experiments complete in %.1fs", elapsed)
        return self.results
