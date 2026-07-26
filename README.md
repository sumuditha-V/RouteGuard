# 🚚 RouteGuard — Delivery Delay Prediction & Dispatch Agent

Predict, **before dispatch**, whether an e-commerce order will arrive late — then
explain *why* and recommend the best dispatch action.

> **Core principle:** the **ML model** predicts the delay probability.
> The **LLM never predicts** — it only *explains* the model's prediction and
> *recommends* an action. A deterministic guardrail rejects any recommendation
> that contradicts the explanation.

```
Order → Features → ML model → Delay probability → SHAP → LangGraph agent → Recommendation → Dashboard
```

---

## Why this project is interesting

Most "predict + LLM" demos let the LLM make up numbers. RouteGuard is built around a
strict separation of duties, and two guarantees that are **proven by unit tests**:

1. **Leakage-safe, honestly-evaluated ML.** Historical-rate features use expanding
   windows so a row can never see its own future, and the model is judged with a
   **rolling-origin backtest** (train on the past, test on the next month, roll
   forward) — the only fair way to score a time-series model.
2. **A guardrail the LLM cannot talk its way past.** The critique step is
   deterministic code, not a prompt: it rejects any recommendation whose justifying
   feature isn't actually a SHAP risk driver, or that restates/ignores the
   probability.

## Results (real numbers from this repo)

| What | Value |
|---|---|
| Dataset | Olist, **96,470** delivered orders |
| Base late rate | **8.1%** (imbalanced) |
| Best model | **LightGBM** — rolling-backtest PR-AUC **0.23**, ROC-AUC **0.72** (beats LR / RF / DT) |
| Calibration | Brier **0.205 → 0.058** after isotonic calibration |
| Cost-based threshold | **0.16** (missed-late costs 5×) → catches **~15× more** late orders than a naïve 0.5 cutoff |

The delay signal is genuinely **non-stationary** (monthly late rate swings ~1% → 21%),
which caps achievable accuracy — the project measures this honestly instead of hiding it.

---

## Quickstart

```bash
# 1. create + activate a virtual environment, then:
pip install -r requirements.txt

# 2. add your Gemini API key (free at https://aistudio.google.com/apikey)
cp .env.example .env      # then edit .env and set GEMINI_API_KEY

# 3. run the dashboard
streamlit run dashboard/Home.py
```

Go to **Prediction** → pick an order → see the risk %, SHAP reasons, and the agent's
recommendation.

> The trained model (`models/registry/model_v1/`) is included, so prediction, SHAP,
> and the dashboard work out of the box. The **Prediction** page's order picker and
> the model-training steps below additionally need the raw Olist CSVs in `data/raw/`
> (download from Kaggle — see [docs/DATA_CARD.md](docs/DATA_CARD.md)).

### Rebuild the model from scratch (optional)

```bash
cd src
python -m routeguard.data          # smoke-test data loading
python -m routeguard.modeling      # rolling backtest: compare models
python -m routeguard.tuning        # Optuna hyperparameter search
python -m routeguard.evaluation    # calibration + threshold + save model_v1
python -m routeguard.explainability  # SHAP sanity check
python -m routeguard.pipeline      # full pipeline on one order (needs GEMINI_API_KEY)
```

### API

```bash
PYTHONPATH=src uvicorn routeguard.api:app --reload
# docs at http://127.0.0.1:8000/docs
```

### Tests

```bash
PYTHONPATH=src pytest        # 8 tests: leakage-safety + agent guardrail + API
```

---

## Architecture

| Layer | Choice |
|---|---|
| ML | scikit-learn, **LightGBM** (baselines: LogReg, Decision Tree, Random Forest) |
| Tuning | Optuna (TPE + pruning) |
| Explainability | SHAP (TreeExplainer) |
| Calibration | isotonic regression |
| Imbalance | `scale_pos_weight` |
| Agent | LangGraph + **Google Gemini** (`gemini-3.5-flash`); provider swappable to Claude |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit (multipage) |
| Storage | SQLite (SQLAlchemy) |
| Dataset | Brazilian E-Commerce (Olist) |

### Agent flow

```
coordinator → explanation → recommendation → critique → (retry | finalize)
                                   ↑____________(reject)__________|
```
The LLM writes the explanation and proposes an action from a fixed allowed set;
the deterministic **critique** validates it against the SHAP facts; `finalize`
returns the result (or falls back to `manual_review` if the guardrail never approves).

---

## Project structure

```
config/       config.yaml — every tunable, no magic numbers
data/         raw → interim → processed  (raw CSVs gitignored)
src/routeguard/
  config.py         load config
  data.py           load + join Olist, build the target
  features.py       20 leakage-safe features
  modeling.py       time split + model zoo + rolling backtest
  tuning.py         Optuna search
  evaluation.py     calibration + threshold + model export
  explainability.py SHAP (global + local)
  registry.py       load a saved model version
  agent.py          LangGraph agent + providers (Gemini/Anthropic) + guardrail
  pipeline.py       end-to-end: predict → SHAP → agent
  api.py            FastAPI service
  db.py             SQLite persistence
prompts/      versioned LLM prompt templates
dashboard/    Streamlit app (Home + 5 pages)
models/       versioned artifacts (model_v1)
tests/        leakage + guardrail + API tests
docs/         ARCHITECTURE, DATA_CARD, NOTES (decision log)
```

---

## Deployment

- **Docker (local):** `docker build -t routeguard . && docker run -p 8501:8501 --env-file .env routeguard`
- **Hugging Face Spaces / Streamlit Cloud:** point it at `dashboard/Home.py`; set
  `GEMINI_API_KEY` as a secret. The included `model_v1` covers prediction; bundle a
  sample dataset if you want the order picker to work without the raw CSVs.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full plan and
[docs/NOTES.md](docs/NOTES.md) for the decision log.

## Honest caveats

- **No real carrier data** in Olist — `choose_alternative_carrier` is illustrative,
  backed by seller history as a proxy ([docs/DATA_CARD.md](docs/DATA_CARD.md)).
- Geolocation is by zip-code prefix (median centroid), so distances are approximate.
