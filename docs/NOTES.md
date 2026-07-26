# Decision Notes

One line per decision. Future-you and reviewers read this. Newest at the bottom.

## Scope decisions (M0)
- **Dataset:** Olist (Brazilian E-Commerce) — standard, real, multi-table, recognizable.
- **Shipped model:** LightGBM only. Baselines: Logistic Regression, Decision Tree, Random Forest.
  → Dropped CatBoost + stacking to keep the project understandable and TreeSHAP clean.
- **Imbalance:** `scale_pos_weight` only. → Dropped SMOTE (avoids synthetic-data confusion,
  keeps probabilities calibratable).
- **Database:** SQLite only. → Dropped Postgres/Alembic for now (same SQLAlchemy code; README
  notes it's swappable).
- **Experiment tracking:** `metrics.json` per model version. → MLflow optional at the end.
- **Carrier:** dropped as a real feature (Olist has none). Kept `choose_alternative_carrier`
  as an illustrative-only agent action, labeled in the UI. See DATA_CARD.
- **Split:** time-based (train on past, test on future). Never random.
- **Build strategy:** thin end-to-end slice first, then improve.

## M1 — Data + EDA findings
- **96,470** usable delivered orders (of 99,441 total).
- **Late rate = 8.11%** → confirms imbalance; `scale_pos_weight ≈ 11.3`.
- Date range: 2016-09 → 2018-10 (usable volume really 2017-01 → 2018-08).
- **Late rate is very volatile month-to-month** (e.g. Mar 2018 = 21%, Jun 2018 = 1.3%).
  → strong real-world justification for time-based split (random split would leak the future).
- Missingness negligible (14 missing approved_at, 1 missing carrier date).
- **Split date confirmed: `2018-06-01`** (train before, test = Jun/Jul/Aug 2018 ≈ 80/20).

## M2 — Feature engineering
- Structure choice: flat modules in `src/routeguard/` (config.py, data.py, features.py)
  instead of deep nested folders — easier to navigate / understand.
- 20 features built: distance (Haversine), same_state, delivery_window_days, freight_ratio,
  total_freight/price, n_items, n_sellers, max_weight_g, calendar (dow/month/hour/weekend),
  is_holiday_season, and leakage-safe hist late rates + counts (seller / cust_state / category).
- **Holiday handling simplified:** used an `is_holiday_season` flag (Nov/Dec) instead of a full
  holiday-calendar dependency — keeps it understandable. Calendar features cover the rest.
- **Leakage-safe rates:** expanding mean with `shift(1)`, sorted by purchase time; cold-start
  filled with global prior-so-far. Documented simplification: "prior" = earlier purchase time
  (not earlier *known* outcome); residual leakage small, accepted for clarity.
- Missing values (distance 478, weight 16) left as NaN on purpose — LightGBM handles them.
- **3 leakage unit tests pass** (future can't leak into past). Headline correctness guarantee.

## M3 — Time split, baselines & a key evaluation finding
- **Winner: LightGBM** — rolling-backtest avg PR-AUC 0.227, ROC 0.715 (beats LR 0.201/0.704,
  RF 0.165/0.661, DT 0.144/0.603). Confirms the plan to ship LightGBM.
- **Important finding — evaluation methodology matters more than the model here:**
  A single pooled holdout (train <2018-06, test Jun–Aug) gave LightGBM ROC ~0.50 (looked broken).
  Root cause = monthly base rates drift hugely (Jun 1.4% vs Mar 21%); pooling months with
  different late rates breaks *cross-month* ranking even though *within-month* ranking is fine
  (per-month ROC ~0.70). Diagnosed via single-feature train-vs-test AUC (hist-rate features
  invert out-of-sample) + rolling backtest.
  → **Primary evaluation = rolling-origin backtest** (train on all earlier months, test next
    month, roll forward). Config: `backtest.test_months`.
- **LightGBM must be regularized** to generalize under this shift: num_leaves 15, max_depth 4,
  min_child_samples 200, subsample/colsample 0.8, reg_lambda 5, lr 0.03. Saved in config.
- **Calibration (M5) is now known to be important** — cross-month probability drift is exactly
  what calibration fixes, and it's what makes the LLM's stated "% chance" trustworthy.

## M4 — Optuna tuning
- 50 trials, TPE sampler + MedianPruner, scored on the rolling backtest (same metric we report).
- Features built once and folds cached for speed; each trial only refits LightGBM.
- **Result: PR-AUC 0.227 -> 0.229 (marginal ~1% gain).** Honest takeaway: the M3 regularized
  params were already near-optimal; the performance ceiling here is data non-stationarity,
  not hyperparameters. Adopted tuned params anyway (config `lightgbm_params`).
- Full-precision best params saved to models/artifacts/best_lgbm_params.json (gitignored).

## M5 — Threshold, calibration & final model
- **Calibration is a big win:** isotonic calibration cut Brier score 0.205 -> 0.058 on held-out
  eval months. Raw scale_pos_weight probabilities are distorted; calibration makes "0.78" mean
  ~78% (needed for the M7 LLM to state trustworthy percentages).
- **Cost-based threshold = 0.16-0.17** (FN costs 5x, FP costs 1x). Transforms usefulness:
  naive 0.50 catches ~2% of late orders; cost-optimal ~0.17 catches ~35% (15x more caught),
  accepting cheaper false alarms. This is the core "score -> real decision" story.
- Honest eval design: OOS predictions from rolling backtest, then time-split OOS into
  dev (fit calibrator + threshold) and eval (report) so nothing is fit on the eval months.
- **Shipped artifact model_v1** = LightGBM retrained on all data + isotonic calibrator +
  threshold, saved to models/registry/model_v1/ (model.pkl, calibrator.pkl, metrics.json).
  Regenerate with `python -m routeguard.evaluation`. Binaries gitignored; metrics.json tracked.

## M6 — SHAP explainability
- registry.py: single loader for model_v1 (model + calibrator + meta), lru_cached;
  predict_proba() helper returns calibrated probability.
- explainability.py: TreeExplainer on the raw LightGBM. global_importance() (mean|SHAP|)
  and explain_order() returning STRUCTURED drivers (feature, value, shap, direction).
- Explains the RAW model (calibration is monotonic, doesn't change feature ranking/direction).
- Top global drivers: delivery_window_days, purchase_month, cust_state/seller hist late rates,
  distance. Sensible + business-interpretable (tight window = #1 risk factor).
- Structured output is the contract feeding the M7 agent (read-only facts) and the dashboard
  (waterfall/force plots).

## M7 — LangGraph dispatch agent
- Graph: coordinator -> explanation -> recommendation -> critique -> (retry | finalize).
- **The LLM never predicts** — probability/prediction/SHAP are read-only facts in the prompt;
  prompts forbid restating/recalculating the probability.
- **Critique is DETERMINISTIC code, not an LLM** (design choice): the LLM proposes, code
  validates against SHAP. Rejects if action out of allowed set, no_action on a predicted-late
  order, or justifying_feature isn't a real risk driver / doesn't justify the action.
  ACTION_REQUIREMENTS maps action -> justifying features.
- LLM client is injected (Protocol) so tests use a FakeLLM — no API key needed for tests.
- Model: claude-sonnet-5, effort=low (Sonnet 5 rejects temperature -> 400). Structured output
  via output_config json_schema. Loads ANTHROPIC_API_KEY from .env (dotenv).
- **4 guardrail tests pass** (contradicting-SHAP, wrong-feature, no_action-on-late all rejected;
  valid recommendation approved). Second headline correctness guarantee.
- Prompts versioned in prompts/ (explanation.md, recommendation.md).

## Provider switch — Gemini
- Agent LLM switched from Claude to **Google Gemini** (`gemini-2.5-flash`) via `google-genai`.
- Clean swap thanks to the injected LLM Protocol: added `GeminiLLM`, kept `AnthropicLLM`,
  added `get_llm()` factory driven by `agent.provider` in config (gemini | anthropic).
- Gemini uses temperature (0.2) + `response_mime_type=application/json`; schema embedded in
  the prompt for structured nodes. Anthropic path uses effort + output_config (unchanged).
- Env var now `GEMINI_API_KEY`. All 7 tests still pass (FakeLLM, no key needed).

## M8 — FastAPI service + SQLite
- db.py: single `predictions` table (SQLAlchemy 2.0) with JSON columns; SQLite at project root,
  swappable to Postgres via connection string. init_db/save/list/get helpers.
- api.py: /health, /model/info, /metrics, /sample (real order to try), /predict (fast, no LLM),
  /recommend (full pipeline + persist), /predictions, /predictions/{id}. Modern lifespan startup.
- /sample serves real orders (high/low/random risk) so the dashboard can demo without hand-typing
  20 features. /predict works without an API key; /recommend needs GEMINI_API_KEY.
- test_api.py: TestClient smoke test (health, model info, metrics, sample->predict) — no key needed.
- Run: `PYTHONPATH=src uvicorn routeguard.api:app --reload` ; docs at /docs.

## (add new decisions below as you build)
