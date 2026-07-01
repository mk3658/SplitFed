#!/usr/bin/env python3
"""Extract and cache representations."""

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
    logger = setup_logger("extract_repr", os.path.join(output_dir, "extract.log"))
    set_seed(exp_cfg.get("seed", 42))

    from src.data.synthetic_clinical import SyntheticClinicalDataset
    from src.models.hf_model import ClinicalLLMWrapper
    from src.representations.extraction import extract_representations

    ds = SyntheticClinicalDataset(config, n_samples=exp_cfg.get("num_samples", 50))
    ds.load()

    model_cfg = config.get("model", {})
    model = ClinicalLLMWrapper(
        model_name=model_cfg.get("name", "distilgpt2"),
        max_length=model_cfg.get("max_length", 128),
    ).load()

    layers = model_cfg.get("selected_layers", [0, 2, 5])
    layers = [l for l in layers if l < model.n_layers]

    layer_data = extract_representations(
        model, ds.samples, layers=layers,
        pooling=model_cfg.get("pooling", "mean"),
        batch_size=exp_cfg.get("batch_size", 4),
        output_dir=os.path.join(output_dir, "representations"),
    )
    logger.info("Done. Layers: %s", list(layer_data.keys()))


if __name__ == "__main__":
    main()
