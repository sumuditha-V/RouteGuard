"""Model training + evaluation (M3).

Two evaluation views:

1. rolling_backtest()  -> PRIMARY, fair evaluation. Train on all months before a
   test month, score that month, roll forward. Averaged over many months this is
   stable and honest for a time-series problem.

2. time_split()        -> a single final holdout (train before cutoff, test after).
   Kept for reference, but note: pooling several months with very different late
   rates makes this view misleading (see docs/NOTES.md, M3).

Models compared: Logistic Regression, Decision Tree, Random Forest, LightGBM.
Ranking metric = PR-AUC (average precision), right for an imbalanced target.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from .config import load_config
from .data import build_dataset
from .features import FEATURE_COLUMNS, build_features


def time_split(df: pd.DataFrame):
    """Split into (train, test) by purchase time using the configured cutoff."""
    cfg = load_config()
    time_col = cfg["split"]["time_column"]
    cutoff = pd.Timestamp(cfg["split"]["train_cutoff_date"])
    return df[df[time_col] < cutoff], df[df[time_col] >= cutoff]


def build_models() -> dict:
    """Return {name: estimator}. Impute-sensitive models are wrapped in a Pipeline
    so imputer/scaler are fit on TRAIN only. LightGBM handles NaN natively."""
    cfg = load_config()
    seed = cfg["model"]["random_seed"]
    spw = cfg["imbalance"]["scale_pos_weight"]
    gbm_params = cfg["model"]["lightgbm_params"]

    return {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "decision_tree": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                           random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=12,
                                           class_weight="balanced", n_jobs=-1,
                                           random_state=seed)),
        ]),
        "lightgbm": LGBMClassifier(
            scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1,
            **gbm_params,
        ),
    }


def rolling_backtest() -> pd.DataFrame:
    """Primary evaluation: for each configured test month, train on all earlier
    months and score that month. Return per-model average PR-AUC / ROC-AUC."""
    cfg = load_config()
    target = cfg["data"]["target_column"]
    time_col = cfg["split"]["time_column"]
    months = cfg["backtest"]["test_months"]

    df = build_features(build_dataset())
    df["_ym"] = df[time_col].dt.to_period("M").astype(str)

    results = {name: {"pr": [], "roc": []} for name in build_models()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in months:
            train, test = df[df["_ym"] < m], df[df["_ym"] == m]
            if len(test) < 200 or test[target].nunique() < 2:
                continue
            Xtr, ytr = train[FEATURE_COLUMNS], train[target]
            Xte, yte = test[FEATURE_COLUMNS], test[target]
            for name, model in build_models().items():
                model.fit(Xtr, ytr)
                p = model.predict_proba(Xte)[:, 1]
                results[name]["pr"].append(average_precision_score(yte, p))
                results[name]["roc"].append(roc_auc_score(yte, p))

    rows = [
        {"model": name,
         "avg_pr_auc": round(np.mean(r["pr"]), 4),
         "avg_roc_auc": round(np.mean(r["roc"]), 4),
         "n_months": len(r["pr"])}
        for name, r in results.items()
    ]
    return pd.DataFrame(rows).sort_values("avg_pr_auc", ascending=False)


if __name__ == "__main__":
    res = rolling_backtest()
    print("=" * 60)
    print("ROLLING BACKTEST - primary evaluation (ranked by PR-AUC)")
    print("=" * 60)
    print(res.to_string(index=False))
