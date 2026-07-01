"""Configuration loading and management for CRA experiments."""

import os
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Dot-access config wrapper
# ---------------------------------------------------------------------------

class Config:
    """Nested dot-access wrapper around a plain dict."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [Config(v) if isinstance(v, dict) else v for v in value])
            else:
                setattr(self, key, value)

    # Allow dict-style access for compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _load_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a deep copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Required fields validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: List[str] = [
    "experiment.name",
    "experiment.seed",
    "dataset.name",
    "model.name",
]


def validate_config(data: Dict[str, Any]) -> None:
    """Raise ValueError if required top-level/nested keys are missing."""
    for field_path in _REQUIRED_FIELDS:
        parts = field_path.split(".")
        node: Any = data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(
                    f"Required config field '{field_path}' is missing."
                )
            node = node[part]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> Config:
    """Load a YAML config file and return a dot-access Config object.

    Parameters
    ----------
    path:
        Path to the primary YAML config file.
    overrides:
        Optional dict of values to merge on top of the loaded config.
    validate:
        Whether to check for required fields.
    """
    data = _load_yaml_file(path)

    if overrides:
        data = _merge_dicts(data, overrides)

    if validate:
        validate_config(data)

    return Config(data)


def load_and_merge_configs(
    base_path: Union[str, Path],
    *override_paths: Union[str, Path],
    extra_overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> Config:
    """Load a base config, then merge one or more override YAML files on top.

    Parameters
    ----------
    base_path:
        Path to the base YAML config.
    *override_paths:
        Paths to additional YAML files to merge in order.
    extra_overrides:
        Optional dict merged last (highest priority).
    validate:
        Whether to validate the final merged config.
    """
    data = _load_yaml_file(base_path)

    for p in override_paths:
        override_data = _load_yaml_file(p)
        data = _merge_dicts(data, override_data)

    if extra_overrides:
        data = _merge_dicts(data, extra_overrides)

    if validate:
        validate_config(data)

    return Config(data)


def config_from_dict(data: Dict[str, Any], validate: bool = False) -> Config:
    """Create a Config directly from a plain dict (useful for tests)."""
    if validate:
        validate_config(data)
    return Config(data)
