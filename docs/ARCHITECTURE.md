# RouteGuard — Architecture & Implementation Plan

This is the reference plan. The trimmed scope (below) is what we're actually building.

## Core principle
The **ML model predicts** the delay probability. The **LLM never predicts** — it only
explains the ML prediction and recommends dispatch actions. Probability flows ML → agent,
never the reverse. The agent treats probability / prediction / SHAP / order details as
**read-only facts**.

## System flow
```
Order Data
  → Feature Engineering (leakage-safe)
  → ML Model (LightGBM)
  → Delay Probability
  → SHAP Explanation
  → LangGraph Agent (coordinator → explanation → recommendation → critique)
  → Dispatch Recommendation
  → Dashboard
```

## Trimmed scope (decided)
| Area | Decision |
|---|---|
| Models | LightGBM shipped; LR / Decision Tree / Random Forest as baselines. No CatBoost, no stacking. |
| Imbalance | `scale_pos_weight` only. No SMOTE. |
| Database | SQLite only (SQLAlchemy). No Postgres/Alembic for now. |
| Experiment tracking | `metrics.json` per model version. MLflow optional at the end. |
| Carrier feature | Dropped as a real feature (Olist has none). "choose_alternative_carrier" kept as an illustrative agent action only. See DATA_CARD. |

## Non-negotiables
1. Leakage-safe historical aggregates (expanding window + `shift`) and time-based split.
2. Critique node that rejects recommendations contradicting SHAP / order context.

## Folder structure (grows as needed)
```
config/     # YAML config — all tunables
data/       # raw → interim → processed (raw is immutable)
src/routeguard/
  ingestion/       # load + join Olist tables
  features/        # leakage-safe transforms (shared train + serve)
  training/        # model zoo + Optuna
  evaluation/      # metrics, threshold, calibration, plots
  explainability/  # SHAP (global + local)
  agents/          # LangGraph nodes, graph, guardrails
  serving/         # FastAPI + Pydantic schemas
  db/              # SQLAlchemy models
  utils/           # config, logging, seeding
prompts/    # versioned prompt templates
dashboard/  # Streamlit (presentation only)
models/     # versioned artifacts (registry/)
tests/      # leakage tests + agent-guardrail tests
docs/       # this file, DATA_CARD, NOTES
```

## Feature engineering (leakage rules)
- Sort globally by `order_purchase_timestamp` first.
- Historical aggregates (seller / region / category late rates) use **expanding windows with
  `shift(1)`** so the current row never sees itself or the future.
- Fit all transformers on **train only**.
- Exclude post-dispatch columns (carrier handoff date, actual delivery date, etc.).
- Only use features known at/before purchase time (the pre-dispatch moment).
- A unit test asserts the leakage invariant.

## Data split — time-based (why)
Train on past, test on future. Random split leaks the future via seasonality/concept drift
and via the historical aggregates. Time split is the only honest estimate of production
performance. Expect metrics to drop vs. a random split — that drop is the truth.

## Evaluation
- Primary ranking metric: **PR-AUC** (imbalanced). Confirm with ROC-AUC.
- Report precision, recall, F1, confusion matrix, Brier score.
- **Threshold** chosen by business cost: `cost_FN * FN + cost_FP * FP`, minimized on validation.
- Calibrate probabilities (isotonic/Platt) so "78% chance" means what it says.

## Agent workflow (LangGraph)
State (read-only): order_details, delay_probability, prediction, shap_top_features, threshold.
Nodes:
1. Coordinator — validate inputs, route.
2. Explanation Worker — turn SHAP + order into a plain business narrative.
3. Recommendation Worker — pick 1–3 actions from the allowed enum, each tied to a SHAP driver.
4. Critique / Validator — reject if recommendation contradicts SHAP / context, restates the
   probability, or is out of the allowed set. reject → retry (max N) → fallback manual_review.
5. Final Response — explanation · reason · action(s) · confidence · business impact. Persist to DB.

## API (FastAPI)
`GET /health` · `GET /model/info` · `POST /predict` · `POST /explain` ·
`POST /recommend` (runs agent) · `GET /predictions` · `GET /predictions/{id}` · `GET /metrics`

## Database (SQLite)
Tables: predictions, explanations, agent_outputs, model_registry. Full audit trail so
"why did we recommend X for order Y under model v1?" is answerable.

## Deployment
Docker + Hugging Face Space (SQLite) for the public demo. Render/Railway optional later.

## Milestones
M0 scaffold · M1 data+EDA · M2 features · M3 split+baselines · M4 LightGBM+Optuna ·
M5 imbalance+eval+threshold · M6 SHAP · M7 agent · M8 FastAPI · M9 dashboard ·
M10 polish · M11 deploy · M12 docs.

## Build strategy
Get one thin slice running end-to-end early (5 features → untuned LightGBM → SHAP → agent →
one Streamlit page), then improve. Don't perfect features before the pipe connects.
