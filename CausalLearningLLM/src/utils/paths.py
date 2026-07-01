"""Path management utilities for CRA experiments."""

import os
from datetime import datetime
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* (and any missing parents) if it does not yet exist.

    Returns the resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_output_dir(config, timestamped: bool = True) -> Path:
    """Return (and create) the experiment output directory.

    If *timestamped* is True a sub-directory with the current date-time is
    appended so that repeated runs do not overwrite each other.

    Parameters
    ----------
    config:
        A ``Config`` object with an ``experiment.output_dir`` field.
    timestamped:
        Whether to append a ``YYYYMMDD_HHMMSS`` sub-directory.
    """
    base = Path(config.experiment.output_dir)
    if timestamped:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base / stamp
    else:
        out_dir = base
    return ensure_dir(out_dir)


def get_representations_path(config, layer: int, split: str) -> Path:
    """Return path for saved hidden-state representations.

    E.g. ``outputs/<name>/representations/layer_03_train.npy``
    """
    base = Path(config.experiment.output_dir) / "representations"
    ensure_dir(base)
    filename = f"layer_{layer:02d}_{split}.npy"
    return base / filename


def get_probe_path(config, task: str, layer: int) -> Path:
    """Return path for a saved probe checkpoint.

    E.g. ``outputs/<name>/probes/clinical_layer_03.pkl``
    """
    base = Path(config.experiment.output_dir) / "probes"
    ensure_dir(base)
    filename = f"{task}_layer_{layer:02d}.pkl"
    return base / filename


def get_figure_path(config, name: str, fmt: str = "pdf") -> Path:
    """Return path for a saved figure.

    Parameters
    ----------
    config:
        Config object.
    name:
        Figure base name (without extension), e.g. ``"layer_profile"``.
    fmt:
        File format: ``"pdf"``, ``"svg"``, or ``"png"``.
    """
    fmt = fmt.lower().lstrip(".")
    base = Path(config.experiment.output_dir) / "figures" / fmt
    ensure_dir(base)
    return base / f"{name}.{fmt}"


def get_table_path(config, name: str) -> Path:
    """Return path for a saved CSV table.

    E.g. ``outputs/<name>/tables/cra_scores.csv``
    """
    base = Path(config.experiment.output_dir) / "tables"
    ensure_dir(base)
    return base / f"{name}.csv"


def get_metrics_path(config, name: str) -> Path:
    """Return path for a saved JSON metrics file.

    E.g. ``outputs/<name>/metrics/probe_results.json``
    """
    base = Path(config.experiment.output_dir) / "metrics"
    ensure_dir(base)
    return base / f"{name}.json"
