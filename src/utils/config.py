"""Utility for loading YAML configuration files."""

import yaml
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML file and return the resulting configuration dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
