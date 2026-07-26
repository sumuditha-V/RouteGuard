"""About — stack, dataset, and honest caveats."""

import streamlit as st

import rgutil  # noqa: F401

st.set_page_config(page_title="About · RouteGuard", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About RouteGuard")

st.markdown(
    """
### What it is
An end-to-end delivery-delay prediction system. A machine-learning model estimates the
probability a delivery will be late **before dispatch**; SHAP explains the estimate; and a
LangGraph agent turns that into a business recommendation — **without the LLM ever
predicting the probability itself**.

### Stack
- **ML:** scikit-learn, LightGBM (baselines: Logistic Regression, Decision Tree, Random Forest)
- **Tuning:** Optuna · **Explainability:** SHAP · **Calibration:** isotonic
- **Agent:** LangGraph + Google Gemini (`gemini-3.5-flash`); provider is swappable to Claude
- **Serving:** FastAPI · **Dashboard:** Streamlit · **Storage:** SQLite (SQLAlchemy)
- **Dataset:** Brazilian E-Commerce (Olist), ~96k delivered orders

### Design highlights
- **Leakage-safe features** + a **time-based / rolling-backtest** evaluation (the honest way
  to measure a time-series model). Proven by unit tests.
- A **deterministic critique guardrail** that rejects any recommendation contradicting the
  SHAP explanation — the LLM proposes, code validates. Proven by unit tests.

### Honest caveats
- **No real carrier data.** Olist has no carrier identity, so the `choose_alternative_carrier`
  action is illustrative and backed by seller history as a proxy.
- **Geolocation is by zip-code prefix** (median centroid), so distances are approximate.
- The delay signal is genuinely **non-stationary** — monthly late rates swing from ~1% to ~21%,
  which caps achievable accuracy. The project measures this honestly rather than hiding it.
"""
)

m = rgutil.meta()
st.caption(f"Loaded model: {m['version']} ({m['model']}), trained {m['created_utc'][:10]}")
