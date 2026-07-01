"""Extract and cache hidden representations from a ClinicalLLMWrapper."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_representations(
    model_wrapper,
    samples: List,
    layers: List[int],
    pooling: str = "mean",
    batch_size: int = 8,
    output_dir: Optional[str] = None,
) -> Dict[int, Dict]:
    """Extract pooled hidden states for all samples and selected layers.

    Returns
    -------
    dict
        ``{layer: {"Z": np.ndarray [N, d], "labels": ..., ...}}``
    """
    texts = [s.input_text for s in samples]
    labels = [s.target_label for s in samples]
    sensitive = [{**s.sensitive_attributes} for s in samples]
    spurious = [{**s.spurious_attributes} for s in samples]
    ids = [s.sample_id for s in samples]

    layer_reps: Dict[int, List[np.ndarray]] = {l: [] for l in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting representations"):
        batch = texts[i : i + batch_size]
        hidden = model_wrapper.get_hidden_states(batch, layers=layers, pooling=pooling)
        for l in layers:
            if l in hidden:
                layer_reps[l].append(hidden[l])

    result: Dict[int, Dict] = {}
    for l in layers:
        if not layer_reps[l]:
            continue
        Z = np.vstack(layer_reps[l])
        result[l] = {
            "Z": Z,
            "labels": np.array(labels),
            "sensitive": sensitive,
            "spurious": spurious,
            "sample_ids": ids,
            "layer": l,
            "pooling": pooling,
            "n_samples": len(samples),
            "hidden_size": Z.shape[1],
        }

    if output_dir:
        save_representations(result, output_dir)

    return result


def save_representations(layer_data: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for l, data in layer_data.items():
        np.save(os.path.join(output_dir, f"layer_{l}_Z.npy"), data["Z"])
        meta = {k: v for k, v in data.items() if k != "Z"}
        if "labels" in meta and hasattr(meta["labels"], "tolist"):
            meta["labels"] = meta["labels"].tolist()
        with open(os.path.join(output_dir, f"layer_{l}_meta.json"), "w") as f:
            json.dump(meta, f)
    logger.info("Saved representations to %s", output_dir)


def load_representations(
    output_dir: str,
    layers: Optional[List[int]] = None,
) -> Dict[int, Dict]:
    import glob

    result: Dict[int, Dict] = {}
    if not os.path.exists(output_dir):
        return result

    for fpath in sorted(glob.glob(os.path.join(output_dir, "layer_*_Z.npy"))):
        l = int(os.path.basename(fpath).split("_")[1])
        if layers and l not in layers:
            continue
        Z = np.load(fpath)
        meta_path = fpath.replace("_Z.npy", "_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["labels"] = np.array(meta["labels"])
        meta["Z"] = Z
        result[l] = meta

    return result


def get_sensitive_labels(
    layer_data: Dict,
    attr_name: str,
    layer: int,
) -> Optional[np.ndarray]:
    if layer not in layer_data:
        return None
    vals = [s.get(attr_name, "unknown") for s in layer_data[layer]["sensitive"]]
    unique = sorted(set(vals))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[v] for v in vals])


def get_spurious_labels(
    layer_data: Dict,
    attr_name: str,
    layer: int,
) -> Optional[np.ndarray]:
    if layer not in layer_data:
        return None
    vals = [s.get(attr_name, "unknown") for s in layer_data[layer]["spurious"]]
    unique = sorted(set(vals))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[v] for v in vals])
