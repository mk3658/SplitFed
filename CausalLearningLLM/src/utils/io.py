"""I/O helpers for saving and loading data artefacts."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialise *data* to a JSON file at *path*.

    The parent directory is created automatically if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)


def load_json(path: Union[str, Path]) -> Any:
    """Load and return the contents of a JSON file."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for types not supported by the default encoder."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(df: Any, path: Union[str, Path], index: bool = False) -> None:
    """Save a pandas DataFrame to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def load_csv(path: Union[str, Path]) -> Any:
    """Load a CSV file and return a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(Path(path))


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def save_yaml(data: Any, path: Union[str, Path]) -> None:
    """Serialise *data* to a YAML file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)


def load_yaml(path: Union[str, Path]) -> Any:
    """Load and return the contents of a YAML file."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
