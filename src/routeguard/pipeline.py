"""End-to-end inference pipeline: order -> probability -> SHAP -> agent (M7/M8 glue).

This is the single entry point the FastAPI service (M8) and Streamlit dashboard (M9)
call. It enforces the core flow: the ML model produces the probability, SHAP explains
it, and only then does the LLM agent turn that into a recommendation.
"""

import pandas as pd

from .agent import run_agent
from .config import load_config
from .explainability import explain_order
from .features import FEATURE_COLUMNS
from .registry import load_model, predict_proba

# order columns worth showing a human (kept small + readable)
_ORDER_SUMMARY_COLS = [
    "distance_km", "delivery_window_days", "total_freight", "max_weight_g",
    "seller_state", "customer_state", "product_category_name_english",
]


def analyze_order(order_row: pd.DataFrame, llm=None, version: str = "model_v1") -> dict:
    """Run the full pipeline for one order (a 1-row DataFrame with FEATURE_COLUMNS
    plus, ideally, the summary columns). `llm` is injected for tests; None = real Claude.

    Returns probability, prediction, SHAP drivers, and the agent's final response.
    """
    cfg = load_config()
    meta = load_model(version)["meta"]
    threshold = meta["threshold"]

    proba = float(predict_proba(order_row[FEATURE_COLUMNS], version)[0])
    prediction = int(proba >= threshold)
    drivers = explain_order(order_row, version=version)["top_drivers"]

    order_details = {c: (None if c not in order_row or pd.isna(order_row[c].iloc[0])
                         else _clean(order_row[c].iloc[0]))
                     for c in _ORDER_SUMMARY_COLS}

    state = {
        "order": order_details,
        "probability": proba,
        "prediction": prediction,
        "threshold": threshold,
        "shap_drivers": drivers,
        "allowed_actions": cfg["agent"]["allowed_actions"],
    }
    agent_final = run_agent(state, llm=llm)

    return {
        "probability": round(proba, 4),
        "prediction": "late" if prediction else "on_time",
        "threshold": threshold,
        "shap_drivers": drivers,
        "agent": agent_final,
    }


def _clean(v):
    """Make numpy scalars JSON-friendly."""
    try:
        return round(float(v), 3) if hasattr(v, "__float__") and not isinstance(v, bool) else v
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    # Demo: pick a high-risk recent order and run the whole pipeline.
    # predict + SHAP work with no API key; the agent step needs GEMINI_API_KEY.
    from .data import build_dataset
    from .features import build_features

    df = build_features(build_dataset()).tail(3000).copy()
    df["p"] = predict_proba(df[FEATURE_COLUMNS])
    order = df.sort_values("p", ascending=False).iloc[[0]]

    print(f"Selected order — model probability {order['p'].iloc[0]:.3f}\n")
    try:
        result = analyze_order(order)
        import json
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        # most likely a missing/invalid GEMINI_API_KEY
        print(f"Agent step failed ({type(e).__name__}: {e}).")
        print("Add your key to .env (GEMINI_API_KEY=...) to run the agent.")
