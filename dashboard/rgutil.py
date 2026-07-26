"""Shared helpers for the Streamlit dashboard.

Puts src/ on the path (so `import routeguard` works when running `streamlit run
dashboard/Home.py`) and provides cached data loaders so slow work (feature build,
rolling backtest, global SHAP) runs once per session.
"""

import pathlib
import sys

# make the routeguard package importable regardless of where streamlit is launched
_ROOT = next(a for a in pathlib.Path(__file__).resolve().parents
             if (a / "src" / "routeguard").exists())
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from routeguard.config import load_config  # noqa: E402
from routeguard.features import FEATURE_COLUMNS  # noqa: E402
from routeguard.registry import load_model, predict_proba  # noqa: E402


@st.cache_resource
def meta() -> dict:
    return load_model()["meta"]


@st.cache_data(show_spinner="Loading orders + features (first time only)…")
def sample_pool() -> pd.DataFrame:
    from routeguard.data import build_dataset
    from routeguard.features import build_features

    df = build_features(build_dataset()).tail(4000).copy()
    df["_p"] = predict_proba(df[FEATURE_COLUMNS])
    return df


@st.cache_data(show_spinner="Running rolling backtest…")
def backtest() -> pd.DataFrame:
    from routeguard.modeling import rolling_backtest

    return rolling_backtest()


@st.cache_data(show_spinner="Computing global feature importance…")
def global_importance() -> pd.DataFrame:
    from routeguard.explainability import global_importance as gi

    return gi()


def allowed_actions() -> list:
    return load_config()["agent"]["allowed_actions"]
