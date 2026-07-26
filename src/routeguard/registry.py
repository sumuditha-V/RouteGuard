"""Load a saved model version from models/registry/.

One place that knows how to load model_v1 (model + calibrator + metadata), used by
explainability, serving (FastAPI), and the dashboard - so they all agree.
"""

import json
from functools import lru_cache

import joblib

from .config import PROJECT_ROOT, load_config


@lru_cache(maxsize=4)
def load_model(version: str = "model_v1") -> dict:
    """Return {model, calibrator, meta} for a saved version. Cached so repeated
    calls (e.g. per API request) don't re-read from disk."""
    cfg = load_config()
    vdir = PROJECT_ROOT / cfg["paths"]["model_registry"] / version
    if not (vdir / "model.pkl").exists():
        raise FileNotFoundError(
            f"{version} not found in {vdir}. Build it first: "
            f"python -m routeguard.evaluation"
        )
    with open(vdir / "metrics.json") as f:
        meta = json.load(f)
    return {
        "model": joblib.load(vdir / "model.pkl"),
        "calibrator": joblib.load(vdir / "calibrator.pkl"),
        "meta": meta,
    }


def predict_proba(features, version: str = "model_v1"):
    """Calibrated delay probability for a feature DataFrame (columns = meta['features'])."""
    bundle = load_model(version)
    raw = bundle["model"].predict_proba(features)[:, 1]
    return bundle["calibrator"].predict(raw)
