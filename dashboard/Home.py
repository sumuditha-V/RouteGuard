"""RouteGuard dashboard — home page.

Run:  streamlit run dashboard/Home.py
"""

import streamlit as st

import rgutil

st.set_page_config(page_title="RouteGuard", page_icon="🚚", layout="wide")

st.title("🚚 RouteGuard")
st.subheader("Delivery Delay Prediction & Dispatch Agent")

st.markdown(
    """
Predict, **before dispatch**, whether an order will arrive late — then explain
*why* and recommend the best dispatch action.

> **Core principle:** the **ML model** predicts the delay probability.
> The **LLM never predicts** — it only *explains* the prediction and *recommends* actions.
"""
)

m = rgutil.meta()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", m["model"])
c2.metric("Decision threshold", m["threshold"])
c3.metric("PR-AUC (eval)", m["eval_metrics"]["pr_auc"])
c4.metric("Brier (calibrated)", m["brier_calibrated"])

st.divider()
st.markdown(
    """
### How it flows
```
Order → Features → ML model → Delay probability → SHAP → LangGraph agent → Recommendation
```

### Pages
- **Prediction** — pick a real order, see its risk, SHAP reasons, and the agent's advice
- **Model Performance** — rolling backtest, metrics, calibration curve
- **SHAP Explanation** — which features drive delay risk overall
- **System Logs** — recommendations logged to the database
- **About** — stack, dataset, and honest caveats

👈 Use the sidebar to navigate.
"""
)
