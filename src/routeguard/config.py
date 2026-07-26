"""Load the project configuration from config/config.yaml.

Everything tunable (paths, split date, model settings) lives in that YAML file,
so we never hard-code magic numbers in the rest of the code. Call `load_config()`
anywhere you need a setting.
"""

from pathlib import Path

import yaml

# Project root = two levels up from this file (src/routeguard/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    """Read config/config.yaml and return it as a plain dictionary."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_dir() -> Path:
    """Absolute path to the data/raw folder (where the Olist CSVs live)."""
    cfg = load_config()
    return PROJECT_ROOT / cfg["paths"]["raw_dir"]
