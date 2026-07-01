#!/usr/bin/env python3
"""Generate LaTeX tables from saved outputs."""

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

    from src.visualization.tables import (
        generate_main_results_table, generate_layer_table, generate_taxonomy_table
    )

    int_df = _load(os.path.join(output_dir, "interventions", "cra_intervention_results.csv"))
    layer_df = _load(os.path.join(output_dir, "metrics", "layer_information.csv"))
    tax_df = _load(os.path.join(output_dir, "metrics", "representation_taxonomy.csv"))

    generate_main_results_table(int_df, output_dir)
    generate_layer_table(layer_df, output_dir)
    generate_taxonomy_table(tax_df, output_dir)

    print(f"Tables saved to {output_dir}/tables/")


if __name__ == "__main__":
    main()
