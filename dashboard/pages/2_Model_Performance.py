"""Model Performance — rolling backtest, evaluation metrics, calibration curve."""

import pandas as pd
import streamlit as st

import rgutil

st.set_page_config(page_title="Model Performance · RouteGuard", page_icon="📊", layout="wide")
st.title("📊 Model Performance")

m = rgutil.meta()

# ---- model comparison (rolling backtest) ----
st.subheader("Model comparison — rolling backtest")
st.caption("Fair evaluation: train on all earlier months, test on the next, roll forward. "
           "Averaged over 10 months. A single pooled holdout is misleading here because "
           "monthly late rates swing a lot.")
bt = rgutil.backtest()
st.dataframe(bt, hide_index=True, use_container_width=True)
st.bar_chart(bt.set_index("model")["avg_pr_auc"], horizontal=True)

st.divider()

# ---- headline metrics at the chosen threshold ----
st.subheader(f"Metrics on held-out months (threshold {m['threshold']})")
em = m["eval_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC", em["pr_auc"])
c2.metric("ROC-AUC", em["roc_auc"])
c3.metric("Recall", em["recall"])
c4.metric("Precision", em["precision"])

cm = em["confusion_matrix"]
st.markdown("**Confusion matrix** (at the business threshold)")
cm_df = pd.DataFrame(
    [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
    index=["actual on-time", "actual late"],
    columns=["predicted on-time", "predicted late"],
)
st.dataframe(cm_df, use_container_width=True)

st.divider()

# ---- calibration ----
st.subheader("Calibration")
st.caption("Isotonic calibration makes the probabilities trustworthy — essential because "
           "the agent states them to users. Lower Brier score is better.")
c1, c2 = st.columns(2)
c1.metric("Brier — raw", m["brier_raw"])
c2.metric("Brier — calibrated", m["brier_calibrated"], help="lower is better")

rc = m["reliability_curve"]
rel = pd.DataFrame({"predicted probability": rc["mean_predicted"],
                    "observed frequency": rc["observed"]}).set_index("predicted probability")
rel["perfect calibration"] = rel.index
st.line_chart(rel)
st.caption("Closer to the diagonal = better calibrated.")
