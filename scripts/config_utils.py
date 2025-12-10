"""
config_utils.py
----------------
Utility functions for configuration loading and directory handling.

Inputs:
    - Paths to YAML config files.
    - Directory paths that may or may not exist.

Outputs:
    - Python dicts containing config parameters.
    - Created directories (ensured via mkdir).
"""

from pathlib import Path
from typing import Dict

import yaml


def load_config(path: str | Path) -> Dict:
    """Load a YAML config file into a dictionary."""
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: Path) -> Path:
    """Create directory (recursively) if it does not exist, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path