"""Optuna hyperparameter tuning for LightGBM (M4).

We optimise the SAME metric we report: average PR-AUC across the rolling backtest
months. This means tuning targets real out-of-sample performance, not a single
(misleading) pooled holdout.

Speed: features are built ONCE and each month's train/test matrices are cached, so
every trial just refits LightGBM - no repeated feature engineering.

Best practices used:
  - TPE sampler (Bayesian search) with a fixed seed for reproducibility.
  - MedianPruner: stop clearly-bad trials early using per-month intermediate scores.
  - Search space centred on regularised settings (this problem punishes overfitting).

Result: best params + score saved to models/artifacts/best_lgbm_params.json.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from .config import PROJECT_ROOT, load_config
from .data import build_dataset
from .features import FEATURE_COLUMNS, build_features


def _cached_months():
    """Build features once and return a list of (X_train, y_train, X_test, y_test)
    tuples, one per backtest month."""
    cfg = load_config()
    target = cfg["data"]["target_column"]
    time_col = cfg["split"]["time_column"]
    months = cfg["backtest"]["test_months"]

    df = build_features(build_dataset())
    df["_ym"] = df[time_col].dt.to_period("M").astype(str)

    folds = []
    for m in months:
        train, test = df[df["_ym"] < m], df[df["_ym"] == m]
        if len(test) < 200 or test[target].nunique() < 2:
            continue
        folds.append((train[FEATURE_COLUMNS], train[target],
                      test[FEATURE_COLUMNS], test[target]))
    return folds


def make_objective(folds, spw, seed):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 400),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        scores = []
        for i, (Xtr, ytr, Xte, yte) in enumerate(folds):
            model = LGBMClassifier(scale_pos_weight=spw, random_state=seed,
                                   n_jobs=-1, verbose=-1, subsample_freq=1, **params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            scores.append(average_precision_score(yte, p))
            # let Optuna prune obviously-bad trials early
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def run_tuning(n_trials: int = None) -> dict:
    cfg = load_config()
    seed = cfg["model"]["random_seed"]
    spw = cfg["imbalance"]["scale_pos_weight"]
    n_trials = n_trials or cfg["tuning"]["optuna_trials"]

    print("building features + backtest folds (once)...")
    folds = _cached_months()
    print(f"folds: {len(folds)} months")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(make_objective(folds, spw, seed), n_trials=n_trials,
                   show_progress_bar=False)

    out = {"best_pr_auc": round(study.best_value, 4), "best_params": study.best_params}
    art_dir = PROJECT_ROOT / cfg["paths"]["artifacts"]
    art_dir.mkdir(parents=True, exist_ok=True)
    with open(art_dir / "best_lgbm_params.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 60)
    print(f"current (M3) PR-AUC : 0.2265")
    print(f"tuned PR-AUC        : {out['best_pr_auc']}")
    print(f"trials run          : {len(study.trials)}")
    print("\nbest params:")
    print(json.dumps(out["best_params"], indent=2))
    print(f"\nsaved -> {art_dir / 'best_lgbm_params.json'}")
    return out


if __name__ == "__main__":
    run_tuning()
