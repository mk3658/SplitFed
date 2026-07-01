#!/usr/bin/env python3
"""
End-to-end debug pipeline.

Usage:
    python scripts/run_all_debug.py --config configs/debug.yaml
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="CRA debug pipeline")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config.get("experiment", {})

    output_dir = args.output_dir or exp_cfg.get("output_dir", "outputs/debug")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("cra_debug", log_file=os.path.join(output_dir, "debug.log"))
    logger.info("=" * 60)
    logger.info("CRA Debug Pipeline")
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    seed = exp_cfg.get("seed", 42)
    set_seed(seed, deterministic=exp_cfg.get("deterministic", True))

    # ------------------------------------------------------------------ #
    # Step 1: Load / generate dataset
    # ------------------------------------------------------------------ #
    logger.info("[1/10] Generating synthetic dataset …")
    from src.data.synthetic_clinical import SyntheticClinicalDataset
    from src.data.attributes import inject_sensitive_attributes, inject_spurious_attributes, generate_counterfactuals
    from src.data.splits import assign_splits

    ds_cfg = config.get("dataset", {})
    n_samples = exp_cfg.get("num_samples", 50)

    dataset = SyntheticClinicalDataset(config, n_samples=n_samples, seed=seed)
    dataset.load()
    samples = dataset.samples

    # ------------------------------------------------------------------ #
    # Step 2: Inject attributes
    # ------------------------------------------------------------------ #
    logger.info("[2/10] Injecting synthetic attributes …")
    import numpy as np
    rng = np.random.RandomState(seed)
    gamma = ds_cfg.get("correlation_strength", 0.8)
    sensitive_attrs = ds_cfg.get("sensitive_attributes", ["age_group", "gender"])
    spurious_attrs = ds_cfg.get("spurious_attributes", ["hospital_type", "note_style"])

    samples = inject_sensitive_attributes(samples, sensitive_attrs, gamma=0.0, mode="independent", rng=rng)
    samples = inject_spurious_attributes(samples, spurious_attrs, gamma=gamma, mode="spurious_train", rng=rng)
    samples = assign_splits(samples, ratios=ds_cfg.get("train_val_test_split", [0.6, 0.2, 0.2]), seed=seed)

    # ------------------------------------------------------------------ #
    # Step 3: Generate counterfactuals
    # ------------------------------------------------------------------ #
    if ds_cfg.get("counterfactual_generation", True):
        logger.info("[3/10] Generating counterfactuals …")
        samples = generate_counterfactuals(samples, cf_type="sensitive")
        samples = generate_counterfactuals(samples, cf_type="spurious")

    logger.info("Samples: %d  |  Splits: train=%d val=%d test=%d",
                len(samples),
                sum(s.split == "train" for s in samples),
                sum(s.split == "val" for s in samples),
                sum(s.split == "test" for s in samples))

    # ------------------------------------------------------------------ #
    # Step 4: Load model
    # ------------------------------------------------------------------ #
    logger.info("[4/10] Loading model …")
    from src.models.hf_model import ClinicalLLMWrapper

    model_cfg = config.get("model", {})
    model = ClinicalLLMWrapper(
        model_name=model_cfg.get("name", "distilgpt2"),
        device=exp_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "float32"),
        max_length=model_cfg.get("max_length", 128),
    )
    model.load()
    logger.info("Model: %s | Layers: %d | Hidden: %d", model.model_name, model.n_layers, model.hidden_size)

    # ------------------------------------------------------------------ #
    # Step 5: Extract representations
    # ------------------------------------------------------------------ #
    logger.info("[5/10] Extracting representations …")
    from src.representations.extraction import extract_representations

    layers = model_cfg.get("selected_layers", [0, 2, 5])
    layers = [l for l in layers if l < model.n_layers]
    pooling = model_cfg.get("pooling", "mean")
    batch_size = exp_cfg.get("batch_size", 4)

    rep_dir = os.path.join(output_dir, "representations")
    layer_data = extract_representations(
        model, samples, layers=layers, pooling=pooling,
        batch_size=batch_size, output_dir=rep_dir,
    )
    logger.info("Extracted layers: %s | Z shape: %s", list(layer_data.keys()),
                layer_data[layers[0]]["Z"].shape if layer_data else "N/A")

    # ------------------------------------------------------------------ #
    # Step 6: Train probes
    # ------------------------------------------------------------------ #
    logger.info("[6/10] Training probes …")
    from src.probing.train import train_probes

    probe_dir = os.path.join(output_dir, "probes")
    sensitive_attr = sensitive_attrs[0]
    spurious_attr = spurious_attrs[0]

    probe_results = train_probes(
        layer_data, layers,
        sensitive_attr=sensitive_attr,
        spurious_attr=spurious_attr,
        config=config,
        output_dir=probe_dir,
    )

    # ------------------------------------------------------------------ #
    # Step 7: CRA interventions
    # ------------------------------------------------------------------ #
    logger.info("[7/10] Running CRA interventions …")
    from src.causal.cra import CRAFramework

    cra = CRAFramework(layer_data, probe_results, config, sensitive_attr, spurious_attr)
    intervention_df = cra.run_full_intervention_study(os.path.join(output_dir, "interventions"))
    logger.info("Intervention results: %d rows", len(intervention_df))

    # ------------------------------------------------------------------ #
    # Step 8: Metrics
    # ------------------------------------------------------------------ #
    logger.info("[8/10] Computing metrics …")
    from src.evaluation.metrics import (
        compute_layer_information_profile,
        compute_privacy_utility_curve,
        compute_robustness_spuriousness_curve,
    )
    metrics_dir = os.path.join(output_dir, "metrics")

    layer_info = compute_layer_information_profile(layer_data, probe_results, sensitive_attr, spurious_attr, metrics_dir)
    geo_df = cra.compute_direction_geometry(metrics_dir)
    taxonomy_df = cra.build_taxonomy_table(metrics_dir)
    pu_df = compute_privacy_utility_curve(intervention_df, metrics_dir)
    rs_df = compute_robustness_spuriousness_curve(intervention_df, metrics_dir)

    # ------------------------------------------------------------------ #
    # Step 9: Theorem verification
    # ------------------------------------------------------------------ #
    logger.info("[9/10] Verifying theoretical claims …")
    from src.theory.verification import run_all_verifications
    stats_dir = os.path.join(output_dir, "statistics")
    theorem_results = run_all_verifications(
        layer_data, probe_results, intervention_df, sensitive_attr, stats_dir
    )

    # ------------------------------------------------------------------ #
    # Step 10: Figures and tables
    # ------------------------------------------------------------------ #
    logger.info("[10/10] Generating figures and tables …")
    from src.visualization.framework_diagram import plot_framework_diagram
    from src.visualization.layerwise import plot_layer_information
    from src.visualization.heatmaps import plot_direction_geometry_heatmap, plot_causal_effect_heatmap
    from src.visualization.pareto import plot_privacy_utility_pareto, plot_robustness_spuriousness_pareto
    from src.visualization.theorem_plots import plot_theorem_verification
    from src.visualization.tables import generate_main_results_table, generate_layer_table, generate_taxonomy_table

    fig_cfg = config.get("visualization", {})
    dpi = fig_cfg.get("figure_dpi", 150)

    plot_framework_diagram(output_dir, dpi=dpi)
    plot_layer_information(layer_info, output_dir, dpi=dpi)
    plot_direction_geometry_heatmap(geo_df, output_dir, dpi=dpi)
    plot_causal_effect_heatmap(intervention_df, output_dir, dpi=dpi)
    plot_privacy_utility_pareto(pu_df if not pu_df.empty else intervention_df, output_dir, dpi=dpi)
    plot_robustness_spuriousness_pareto(rs_df if not rs_df.empty else intervention_df, output_dir, dpi=dpi)
    plot_theorem_verification(theorem_results, output_dir, dpi=dpi)

    generate_main_results_table(intervention_df, output_dir)
    generate_layer_table(layer_info, output_dir)
    generate_taxonomy_table(taxonomy_df, output_dir)

    logger.info("=" * 60)
    logger.info("Debug pipeline complete!")
    logger.info("Outputs saved to: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
