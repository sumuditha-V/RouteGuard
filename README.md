# RouteGuard — Delivery Delay Prediction & Dispatch Agent

Predict, **before dispatch**, whether an e-commerce order is likely to arrive late —
then explain *why* and recommend the best dispatch action.

> **Core principle:** the **ML model** predicts the delay probability.
> The **LLM never predicts** — it only *explains* the ML prediction and *recommends* actions.

## What it does
```
Order → Features → ML Model → Delay Probability → SHAP → LangGraph Agent → Recommendation → Dashboard
```

## Tech stack (trimmed for clarity)
- **ML:** scikit-learn, LightGBM (baselines: Logistic Regression, Decision Tree, Random Forest)
- **Imbalance:** `scale_pos_weight`
- **Explainability:** SHAP (TreeExplainer)
- **Agent:** LangGraph + Claude (multi-node: coordinator → explanation → recommendation → critique)
- **Serving:** FastAPI
- **Dashboard:** Streamlit
- **Storage:** SQLite (SQLAlchemy)
- **Dataset:** Brazilian E-Commerce (Olist)

## Project status
🚧 In development — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full plan
and [docs/NOTES.md](docs/NOTES.md) for the running decision log.

## Non-negotiables (the headline features)
1. **Leakage-safe features** + **time-based split** (train on past, test on future).
2. **Critique node** that rejects LLM recommendations contradicting the SHAP explanation.

## Setup (placeholder — filled in during M0/M1)
```bash
python -m venv .venv
# activate, then:
pip install -r requirements.txt
```

## Docs
- [Architecture & plan](docs/ARCHITECTURE.md)
- [Data card (dataset provenance + caveats)](docs/DATA_CARD.md)
- [Decision notes](docs/NOTES.md)
