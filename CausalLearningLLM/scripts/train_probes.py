#!/usr/bin/env python3
"""Train probes for Y, S, C from cached representations."""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config.get("experiment", {})
    output_dir = exp_cfg.get("output_dir", "outputs/debug")
    logger = setup_logger("train_probes", os.path.join(output_dir, "probes.log"))
    set_seed(exp_cfg.get("seed", 42))

    from src.representations.extraction import load_representations
    from src.probing.train import train_probes

    rep_dir = os.path.join(output_dir, "representations")
    layer_data = load_representations(rep_dir)
    if not layer_data:
        logger.error("No representations found at %s. Run extract_representations.py first.", rep_dir)
        sys.exit(1)

    ds_cfg = config.get("dataset", {})
    sensitive_attr = ds_cfg.get("sensitive_attributes", ["age_group"])[0]
    spurious_attr = ds_cfg.get("spurious_attributes", ["hospital_type"])[0]

    probe_results = train_probes(
        layer_data, list(layer_data.keys()),
        sensitive_attr=sensitive_attr,
        spurious_attr=spurious_attr,
        config=config,
        output_dir=os.path.join(output_dir, "probes"),
    )
    logger.info("Probes trained for %d layers.", len(probe_results))


if __name__ == "__main__":
    main()
