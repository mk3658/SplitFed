#!/usr/bin/env python3
"""Run theorem verification experiments."""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.utils.logging import setup_logger
    config = load_config(args.config)
    exp_cfg = config.get("experiment", {})
    output_dir = exp_cfg.get("output_dir", "outputs/debug")
    logger = setup_logger("verify_theorems", os.path.join(output_dir, "theorems.log"))

    from src.representations.extraction import load_representations
    from src.probing.train import train_probes
    from src.causal.cra import CRAFramework
    from src.theory.verification import run_all_verifications
    import pandas as pd

    layer_data = load_representations(os.path.join(output_dir, "representations"))
    if not layer_data:
        logger.error("No representations found. Run extract_representations.py first.")
        sys.exit(1)

    ds_cfg = config.get("dataset", {})
    sensitive_attr = ds_cfg.get("sensitive_attributes", ["age_group"])[0]
    spurious_attr = ds_cfg.get("spurious_attributes", ["hospital_type"])[0]

    probe_results = train_probes(
        layer_data, list(layer_data.keys()),
        sensitive_attr=sensitive_attr, spurious_attr=spurious_attr,
        config=config,
    )

    cra = CRAFramework(layer_data, probe_results, config, sensitive_attr, spurious_attr)
    intervention_df = cra.run_full_intervention_study()

    stats_dir = os.path.join(output_dir, "statistics")
    results = run_all_verifications(layer_data, probe_results, intervention_df, sensitive_attr, stats_dir)

    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            logger.info("%s:\n%s", name, df.to_string(index=False))


if __name__ == "__main__":
    main()
