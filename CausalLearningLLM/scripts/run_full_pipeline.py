#!/usr/bin/env python3
"""
Full CRA pipeline.

Usage:
    python scripts/run_full_pipeline.py --config configs/pubmedqa.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pubmedqa.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config.get("experiment", {})
    output_dir = args.output_dir or exp_cfg.get("output_dir", "outputs/run")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("cra_full", log_file=os.path.join(output_dir, "run.log"))
    set_seed(exp_cfg.get("seed", 42), deterministic=exp_cfg.get("deterministic", True))

    ds_cfg = config.get("dataset", {})
    dataset_name = ds_cfg.get("name", "synthetic")

    # Load dataset
    logger.info("Loading dataset: %s", dataset_name)
    if dataset_name == "pubmedqa":
        from src.data.pubmedqa import PubMedQADataset
        ds = PubMedQADataset(config, max_samples=ds_cfg.get("max_samples"))
    elif dataset_name == "medqa":
        from src.data.medqa import MedQADataset
        ds = MedQADataset(config, max_samples=ds_cfg.get("max_samples"))
    elif dataset_name == "medmcqa":
        from src.data.medmcqa import MedMCQADataset
        ds = MedMCQADataset(config, max_samples=ds_cfg.get("max_samples"))
    elif dataset_name == "mimic_placeholder":
        from src.data.mimic_placeholder import MIMICPlaceholderDataset
        ds = MIMICPlaceholderDataset(config)
    else:
        from src.data.synthetic_clinical import SyntheticClinicalDataset
        ds = SyntheticClinicalDataset(config, n_samples=ds_cfg.get("max_samples", 200))
    ds.load()
    samples = ds.samples

    # Inject attributes
    import numpy as np
    from src.data.attributes import inject_sensitive_attributes, inject_spurious_attributes, generate_counterfactuals
    from src.data.splits import assign_splits

    rng = np.random.RandomState(exp_cfg.get("seed", 42))
    sensitive_attrs = ds_cfg.get("sensitive_attributes", ["age_group"])
    spurious_attrs = ds_cfg.get("spurious_attributes", ["hospital_type"])
    gamma = ds_cfg.get("correlation_strength", 0.75)

    if ds_cfg.get("use_synthetic_attributes", True):
        samples = inject_sensitive_attributes(samples, sensitive_attrs, gamma=0.0, mode="independent", rng=rng)
        samples = inject_spurious_attributes(samples, spurious_attrs, gamma=gamma, mode="spurious_train", rng=rng)

    if ds_cfg.get("counterfactual_generation", True):
        samples = generate_counterfactuals(samples, cf_type="sensitive")
        samples = generate_counterfactuals(samples, cf_type="spurious")

    samples = assign_splits(samples, ratios=ds_cfg.get("train_val_test_split", [0.6, 0.2, 0.2]))

    # Model
    from src.models.hf_model import ClinicalLLMWrapper
    model_cfg = config.get("model", {})
    model = ClinicalLLMWrapper(
        model_name=model_cfg.get("name", "distilgpt2"),
        device=exp_cfg.get("device", "auto"),
        dtype=model_cfg.get("dtype", "float32"),
        max_length=model_cfg.get("max_length", 256),
    ).load()

    # Extract representations
    from src.representations.extraction import extract_representations
    layers = model_cfg.get("selected_layers", list(range(model.n_layers)))
    layers = [l for l in layers if l < model.n_layers]
    layer_data = extract_representations(
        model, samples, layers=layers,
        pooling=model_cfg.get("pooling", "mean"),
        batch_size=exp_cfg.get("batch_size", 8),
        output_dir=os.path.join(output_dir, "representations"),
    )

    # Train probes
    from src.probing.train import train_probes
    sensitive_attr = sensitive_attrs[0]
    spurious_attr = spurious_attrs[0]
    probe_results = train_probes(
        layer_data, layers, sensitive_attr, spurious_attr,
        config=config, output_dir=os.path.join(output_dir, "probes"),
    )

    # Run all experiments
    from src.evaluation.experiment_runner import ExperimentRunner
    runner = ExperimentRunner(config, output_dir)
    results = runner.run(layer_data, probe_results, sensitive_attr, spurious_attr)

    # CRA framework intervention study (item 6)
    try:
        from src.causal.cra import CRAFramework
        cra_out = os.path.join(output_dir, "cra")
        cra = CRAFramework(layer_data, probe_results, config, sensitive_attr, spurious_attr)
        cra.run_full_intervention_study(cra_out)
        logger.info("CRA framework outputs: %s", cra_out)
    except Exception as e:
        logger.warning("CRAFramework skipped: %s", e)

    # Generate figures
    logger.info("Generating figures …")
    from src.visualization.framework_diagram import plot_framework_diagram
    from src.visualization.layerwise import plot_layer_information
    from src.visualization.heatmaps import plot_direction_geometry_heatmap, plot_causal_effect_heatmap
    from src.visualization.pareto import plot_privacy_utility_pareto, plot_robustness_spuriousness_pareto
    from src.visualization.mediation_plots import plot_mediation_heatmap
    from src.visualization.theorem_plots import plot_theorem_verification
    from src.visualization.tables import generate_main_results_table, generate_layer_table, generate_taxonomy_table

    dpi = config.get("visualization", {}).get("figure_dpi", 300)
    plot_framework_diagram(output_dir, dpi=dpi)
    plot_layer_information(results.get("layer_information", __import__("pandas").DataFrame()), output_dir, dpi=dpi)
    plot_direction_geometry_heatmap(results.get("direction_geometry", __import__("pandas").DataFrame()), output_dir, dpi=dpi)
    plot_causal_effect_heatmap(results.get("interventions", __import__("pandas").DataFrame()), output_dir, dpi=dpi)
    plot_privacy_utility_pareto(results.get("privacy_utility", __import__("pandas").DataFrame()), output_dir, dpi=dpi)
    plot_robustness_spuriousness_pareto(results.get("robustness_spuriousness", __import__("pandas").DataFrame()), output_dir, dpi=dpi)
    plot_mediation_heatmap(results.get("mediation", __import__("pandas").DataFrame()), output_dir, dpi=dpi)
    plot_theorem_verification(results.get("theorems", {}), output_dir, dpi=dpi)

    generate_main_results_table(results.get("interventions", __import__("pandas").DataFrame()), output_dir)
    generate_layer_table(results.get("layer_information", __import__("pandas").DataFrame()), output_dir)
    generate_taxonomy_table(results.get("taxonomy", __import__("pandas").DataFrame()), output_dir)

    logger.info("Pipeline complete. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
