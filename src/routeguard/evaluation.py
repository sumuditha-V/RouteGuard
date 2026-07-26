"""Threshold selection, calibration, and final model export (M5).

Why calibration matters here: LightGBM is trained with scale_pos_weight (to fight
the 8% imbalance), which inflates its raw probabilities. On top of that, monthly
base rates drift (M3 finding). Isotonic calibration maps the raw score back onto a
real probability, so "0.78" actually means ~78% - essential because the M7 LLM
agent narrates that number to users.

Honest evaluation: we get out-of-sample (OOS) predictions from the rolling
backtest (each month predicted by a model trained only on earlier months). We then
split those OOS months in time: earlier months = "dev" (fit calibrator + pick
threshold), later months = "eval" (report unbiased metrics).

The shipped artifact (model_v1) = LightGBM retrained on ALL data + an isotonic
calibrator + the chosen threshold, saved to models/registry/.
"""

import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import PROJECT_ROOT, load_config
from .data import build_dataset
from .features import FEATURE_COLUMNS, build_features


def _make_lgbm():
    cfg = load_config()
    return LGBMClassifier(
        scale_pos_weight=cfg["imbalance"]["scale_pos_weight"],
        random_state=cfg["model"]["random_seed"], n_jobs=-1, verbose=-1,
        **cfg["model"]["lightgbm_params"],
    )


def oos_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-backtest OOS predictions for LightGBM: for each backtest month,
    train on earlier months and predict that month. Returns [ym, y_true, raw_proba]."""
    cfg = load_config()
    target = cfg["data"]["target_column"]
    months = cfg["backtest"]["test_months"]
    df = df.copy()
    df["_ym"] = df[cfg["split"]["time_column"]].dt.to_period("M").astype(str)

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in months:
            train, test = df[df["_ym"] < m], df[df["_ym"] == m]
            if len(test) < 200 or test[target].nunique() < 2:
                continue
            model = _make_lgbm()
            model.fit(train[FEATURE_COLUMNS], train[target])
            proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            out.append(pd.DataFrame({"ym": m, "y_true": test[target].values,
                                     "raw_proba": proba}))
    return pd.concat(out, ignore_index=True)


def pick_threshold(y_true, proba, cost_fn, cost_fp):
    """Sweep thresholds and return the one that minimizes business cost
    (cost_fn * false_negatives + cost_fp * false_positives)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        preds = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return round(float(best_t), 3), float(best_cost)


def reliability_curve(y_true, proba, n_bins=10):
    """Return (mean_predicted, observed_freq) per probability bin, for plotting."""
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins,
                                            strategy="quantile")
    return mean_pred.tolist(), frac_pos.tolist()


def _metrics_at(y_true, proba, threshold):
    preds = (proba >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_m5() -> dict:
    cfg = load_config()
    c_fn = cfg["evaluation"]["cost_false_negative"]
    c_fp = cfg["evaluation"]["cost_false_positive"]

    print("generating out-of-sample predictions (rolling backtest)...")
    df = build_features(build_dataset())
    oos = oos_predictions(df)

    # time-split the OOS months: earlier = dev (fit), later = eval (report)
    months = sorted(oos["ym"].unique())
    split = months[len(months) // 2]
    dev, ev = oos[oos["ym"] < split], oos[oos["ym"] >= split]
    print(f"dev months: {months[:len(months)//2]}  |  eval months: {months[len(months)//2:]}")

    # 1) fit isotonic calibrator on dev raw probabilities
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(dev["raw_proba"], dev["y_true"])
    ev_cal = iso.predict(ev["raw_proba"])
    dev_cal = iso.predict(dev["raw_proba"])

    # 2) choose the cost-minimizing threshold on calibrated dev predictions
    threshold, _ = pick_threshold(dev["y_true"], dev_cal, c_fn, c_fp)

    # 3) report on the untouched eval set
    brier_raw = round(brier_score_loss(ev["y_true"], ev["raw_proba"]), 4)
    brier_cal = round(brier_score_loss(ev["y_true"], ev_cal), 4)
    metrics_default = _metrics_at(ev["y_true"], ev_cal, 0.5)
    metrics_cost = _metrics_at(ev["y_true"], ev_cal, threshold)

    print("\n" + "=" * 60)
    print("M5 EVALUATION (on held-out eval months)")
    print("=" * 60)
    print(f"Brier score  raw={brier_raw}  ->  calibrated={brier_cal}  (lower is better)")
    print(f"Cost-optimal threshold (FN={c_fn}x, FP={c_fp}x): {threshold}")
    print(f"\n@0.50 threshold : recall={metrics_default['recall']}  "
          f"precision={metrics_default['precision']}  cm={metrics_default['confusion_matrix']}")
    print(f"@{threshold} threshold : recall={metrics_cost['recall']}  "
          f"precision={metrics_cost['precision']}  cm={metrics_cost['confusion_matrix']}")

    # ---- build & save the SHIPPED artifact (model_v1) ----
    print("\ntraining final model on ALL data + fitting final calibrator...")
    target = cfg["data"]["target_column"]
    final_model = _make_lgbm()
    final_model.fit(df[FEATURE_COLUMNS], df[target])
    # final calibrator uses ALL oos predictions (best available calibration data)
    final_iso = IsotonicRegression(out_of_bounds="clip")
    final_iso.fit(oos["raw_proba"], oos["y_true"])
    final_threshold, _ = pick_threshold(
        oos["y_true"], final_iso.predict(oos["raw_proba"]), c_fn, c_fp)

    version = cfg.get("serving", {}).get("model_version", "model_v1") \
        if isinstance(cfg.get("serving"), dict) else "model_v1"
    vdir = PROJECT_ROOT / cfg["paths"]["model_registry"] / "model_v1"
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, vdir / "model.pkl")
    joblib.dump(final_iso, vdir / "calibrator.pkl")

    mean_pred, obs = reliability_curve(ev["y_true"], ev_cal)
    meta = {
        "version": "model_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm",
        "features": FEATURE_COLUMNS,
        "threshold": final_threshold,
        "eval_metrics": metrics_cost,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "reliability_curve": {"mean_predicted": mean_pred, "observed": obs},
        "cost_fn": c_fn, "cost_fp": c_fp,
    }
    with open(vdir / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved model_v1 -> {vdir}  (threshold={final_threshold})")
    return meta


if __name__ == "__main__":
    run_m5()
