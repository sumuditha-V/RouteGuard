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

## (add new decisions below as you build)
