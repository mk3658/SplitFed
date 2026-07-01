#!/usr/bin/env python3
"""Generate all publication figures from saved outputs."""

import argparse, os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()

    from src.utils.config import load_config
    config = load_config(args.config)
    output_dir = config.get("experiment", {}).get("output_dir", "outputs/debug")
    dpi = config.get("visualization", {}).get("figure_dpi", 150)

    metrics = os.path.join(output_dir, "metrics")
    interventions = os.path.join(output_dir, "interventions")
    med = os.path.join(output_dir, "mediation")
    stats = os.path.join(output_dir, "statistics")

    from src.visualization.framework_diagram import plot_framework_diagram
    from src.visualization.layerwise import plot_layer_information
    from src.visualization.heatmaps import plot_direction_geometry_heatmap, plot_causal_effect_heatmap
    from src.visualization.pareto import plot_privacy_utility_pareto, plot_robustness_spuriousness_pareto
    from src.visualization.mediation_plots import plot_mediation_heatmap
    from src.visualization.ablations import plot_ablation_summary

    plot_framework_diagram(output_dir, dpi=dpi)
    plot_layer_information(_load(os.path.join(metrics, "layer_information.csv")), output_dir, dpi=dpi)
    plot_direction_geometry_heatmap(_load(os.path.join(metrics, "direction_geometry.csv")), output_dir, dpi=dpi)

    int_df = _load(os.path.join(interventions, "cra_intervention_results.csv"))
    plot_causal_effect_heatmap(int_df, output_dir, dpi=dpi)
    plot_privacy_utility_pareto(_load(os.path.join(metrics, "privacy_utility_curve.csv")) or int_df, output_dir, dpi=dpi)
    plot_robustness_spuriousness_pareto(_load(os.path.join(metrics, "robustness_spuriousness_curve.csv")) or int_df, output_dir, dpi=dpi)
    plot_mediation_heatmap(_load(os.path.join(med, "mediation_results.csv")), output_dir, dpi=dpi)
    plot_ablation_summary(int_df, output_dir, dpi=dpi)

    print(f"Figures saved to {output_dir}/figures/")


if __name__ == "__main__":
    main()
