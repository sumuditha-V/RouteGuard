"""SHAP explanations for model_v1 (M6).

This is the bridge between the ML model and the LLM agent (M7). SHAP tells us WHICH
features pushed a given order toward "late" (positive SHAP) or "on time" (negative).

We explain the RAW LightGBM model. Calibration is a monotonic post-transform of the
score, so it doesn't change which features matter or their direction - it only
rescales the final probability. So SHAP on the raw model correctly explains the
ranking that the calibrated probability preserves.

Outputs are STRUCTURED (lists of dicts), not images: the agent consumes them as
read-only facts, and the dashboard renders them as waterfall/force plots.
"""

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from .data import build_dataset
from .features import FEATURE_COLUMNS, build_features
from .registry import load_model


@lru_cache(maxsize=1)
def _explainer(version: str = "model_v1"):
    """TreeExplainer for the saved model (cached - building it isn't free)."""
    model = load_model(version)["model"]
    return shap.TreeExplainer(model)


def _positive_class_shap(explainer, X: pd.DataFrame) -> np.ndarray:
    """Return SHAP values for the 'late' class as a (n_rows, n_features) array,
    robust to the different shapes SHAP returns for binary classifiers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = explainer.shap_values(X)
    if isinstance(vals, list):          # binary API: [class0, class1]
        return np.asarray(vals[1])
    vals = np.asarray(vals)
    if vals.ndim == 3:                  # (n, features, classes)
        return vals[:, :, 1]
    return vals                         # already (n, features)


def global_importance(sample_size: int = 2000, version: str = "model_v1") -> pd.DataFrame:
    """Overall feature importance = mean(|SHAP|) over a data sample.
    Returns a DataFrame [feature, mean_abs_shap] sorted descending."""
    df = build_features(build_dataset())
    X = df[FEATURE_COLUMNS]
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
    shap_vals = _positive_class_shap(_explainer(version), X)
    imp = np.abs(shap_vals).mean(axis=0)
    return (pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


def explain_order(order_features: pd.DataFrame, top_n: int = 6,
                  version: str = "model_v1") -> dict:
    """Local explanation for ONE order.

    order_features: a 1-row DataFrame whose columns include FEATURE_COLUMNS.
    Returns a dict with the top drivers (feature, value, shap, direction) - the
    exact structure the LLM agent receives as read-only facts.
    """
    X = order_features[FEATURE_COLUMNS].iloc[[0]]
    explainer = _explainer(version)
    shap_row = _positive_class_shap(explainer, X)[0]
    base = explainer.expected_value
    base = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) else float(base)

    drivers = []
    for feat, val, sv in zip(FEATURE_COLUMNS, X.iloc[0].values, shap_row):
        drivers.append({
            "feature": feat,
            "value": None if pd.isna(val) else round(float(val), 4),
            "shap": round(float(sv), 4),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
        })
    drivers.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return {
        "base_value": round(float(base), 4),
        "top_drivers": drivers[:top_n],
        "all_drivers": drivers,
    }


if __name__ == "__main__":
    # sanity check: global importance + explain one late-looking order
    print("=" * 55)
    print("GLOBAL FEATURE IMPORTANCE (mean |SHAP|)")
    print("=" * 55)
    print(global_importance().head(10).to_string(index=False))

    df = build_features(build_dataset())
    # pick an order the model scores as high risk to show a meaningful explanation
    from .registry import predict_proba
    df = df.tail(5000).copy()
    df["p"] = predict_proba(df[FEATURE_COLUMNS])
    order = df.sort_values("p", ascending=False).iloc[[0]]
    print("\n" + "=" * 55)
    print(f"LOCAL EXPLANATION - order prob={order['p'].iloc[0]:.3f} "
          f"actual_late={int(order['late_delivery'].iloc[0])}")
    print("=" * 55)
    exp = explain_order(order)
    print(f"base_value: {exp['base_value']}")
    for d in exp["top_drivers"]:
        print(f"  {d['feature']:28s} value={str(d['value']):>10s}  "
              f"shap={d['shap']:+.3f}  {d['direction']}")
