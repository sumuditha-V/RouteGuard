"""FastAPI service exposing the RouteGuard pipeline (M8).

Endpoints:
  GET  /health              liveness
  GET  /model/info          current model version, threshold, metrics
  GET  /metrics             evaluation metrics + reliability curve (for the dashboard)
  GET  /sample              a real order record to try (risk=high|low|random)
  POST /predict             record -> probability, prediction, SHAP  (no agent, fast)
  POST /recommend           record -> full pipeline (predict+SHAP+agent), logged to DB
  GET  /predictions         recent logged results
  GET  /predictions/{id}    one logged result

Run:  PYTHONPATH=src uvicorn routeguard.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import db
from .explainability import explain_order
from .features import FEATURE_COLUMNS
from .pipeline import _ORDER_SUMMARY_COLS, analyze_order
from .registry import load_model, predict_proba


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()      # create tables if needed
    load_model()      # warm the model cache
    yield


app = FastAPI(title="RouteGuard API",
              description="Delivery delay prediction & dispatch agent",
              version="1.0.0", lifespan=lifespan)


class OrderIn(BaseModel):
    """A single order as a flat record of column -> value. Missing feature
    columns are treated as NaN (LightGBM handles them)."""
    record: dict[str, Any]


@lru_cache(maxsize=1)
def _sample_pool() -> pd.DataFrame:
    """Build a pool of real orders (with features) to serve from /sample. Built once."""
    from .data import build_dataset
    from .features import build_features

    df = build_features(build_dataset()).tail(4000).copy()
    df["_p"] = predict_proba(df[FEATURE_COLUMNS])
    return df


def _row_from_record(record: dict) -> pd.DataFrame:
    row = pd.DataFrame([record])
    for c in FEATURE_COLUMNS:
        if c not in row.columns:
            row[c] = np.nan
    return row


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    meta = load_model()["meta"]
    return {"version": meta["version"], "model": meta["model"],
            "threshold": meta["threshold"], "trained_utc": meta["created_utc"],
            "features": meta["features"]}


@app.get("/metrics")
def metrics():
    meta = load_model()["meta"]
    return {"eval_metrics": meta["eval_metrics"],
            "brier_raw": meta["brier_raw"],
            "brier_calibrated": meta["brier_calibrated"],
            "reliability_curve": meta["reliability_curve"],
            "threshold": meta["threshold"]}


@app.get("/sample")
def sample(risk: str = "random"):
    """Return a real order record to try. risk = high | low | random."""
    pool = _sample_pool()
    if risk == "high":
        row = pool.sort_values("_p", ascending=False).iloc[[0]]
    elif risk == "low":
        row = pool.sort_values("_p", ascending=True).iloc[[0]]
    else:
        row = pool.sample(1)
    cols = FEATURE_COLUMNS + [c for c in _ORDER_SUMMARY_COLS if c not in FEATURE_COLUMNS]
    rec = row[cols].iloc[0].to_dict()
    rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
    return {"record": rec}


@app.post("/predict")
def predict(order: OrderIn):
    """Fast path: probability + prediction + SHAP, no LLM agent."""
    row = _row_from_record(order.record)
    meta = load_model()["meta"]
    proba = float(predict_proba(row[FEATURE_COLUMNS])[0])
    pred = "late" if proba >= meta["threshold"] else "on_time"
    drivers = explain_order(row)["top_drivers"]
    return {"probability": round(proba, 4), "prediction": pred,
            "threshold": meta["threshold"], "shap_drivers": drivers}


@app.post("/recommend")
def recommend(order: OrderIn):
    """Full pipeline: predict + SHAP + LangGraph agent. Needs GEMINI_API_KEY.
    Result is persisted to the database."""
    row = _row_from_record(order.record)
    try:
        result = analyze_order(row)
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"Agent step failed: {type(e).__name__}: {e}")
    meta = load_model()["meta"]
    saved = db.save_prediction(
        model_version=meta["version"], probability=result["probability"],
        prediction=result["prediction"], threshold=result["threshold"],
        order=result["order"], shap_drivers=result["shap_drivers"],
        agent=result["agent"])
    result["id"] = saved["id"]
    return result


@app.get("/predictions")
def predictions(limit: int = 50):
    return {"predictions": db.list_predictions(limit)}


@app.get("/predictions/{pred_id}")
def prediction_detail(pred_id: int):
    row = db.get_prediction(pred_id)
    if row is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    return row
