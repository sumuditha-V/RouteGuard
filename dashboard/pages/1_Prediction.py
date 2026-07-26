"""Prediction page — pick a real order, see its risk, SHAP reasons, and the agent's advice."""

import pandas as pd
import streamlit as st

import rgutil
from routeguard import db
from routeguard.explainability import explain_order
from routeguard.features import FEATURE_COLUMNS
from routeguard.pipeline import _ORDER_SUMMARY_COLS
from routeguard.registry import predict_proba

st.set_page_config(page_title="Prediction · RouteGuard", page_icon="🔮", layout="wide")


def _clean(v):
    """Make numpy scalars JSON-friendly for the agent state + DB."""
    try:
        return round(float(v), 3) if hasattr(v, "__float__") else v
    except (TypeError, ValueError):
        return str(v)


st.title("🔮 Prediction")
st.caption("ML model predicts the probability. The agent only explains it and recommends an action.")

pool = rgutil.sample_pool()
m = rgutil.meta()
threshold = m["threshold"]

risk = st.radio("Choose an order to analyze", ["High risk", "Low risk", "Random"],
                horizontal=True)
if st.button("🎲 Pick an order", type="primary"):
    if risk == "High risk":
        cand = pool.sort_values("_p", ascending=False).head(50)
    elif risk == "Low risk":
        cand = pool.sort_values("_p").head(50)
    else:
        cand = pool
    st.session_state["order"] = cand.sample(1)
    st.session_state.pop("agent_result", None)

if "order" not in st.session_state:
    st.info("Pick an order to begin.")
    st.stop()

row = st.session_state["order"]
proba = float(predict_proba(row[FEATURE_COLUMNS])[0])
pred_late = proba >= threshold

# ---- prediction + order details ----
left, right = st.columns([1, 2])
with left:
    st.metric("Delay probability", f"{proba * 100:.1f}%")
    st.progress(min(int(proba * 100), 100))
    if pred_late:
        st.error("Predicted: **LATE**")
    else:
        st.success("Predicted: **ON TIME**")
    st.caption(f"business threshold = {threshold}")

with right:
    st.markdown("**Order details**")
    details = {c: row[c].iloc[0] for c in _ORDER_SUMMARY_COLS if c in row}
    st.dataframe(pd.DataFrame(details.items(), columns=["field", "value"]),
                 hide_index=True, use_container_width=True)

# ---- SHAP ----
st.subheader("Why — top SHAP drivers")
drivers = explain_order(row)["top_drivers"]
ddf = pd.DataFrame(drivers)
st.bar_chart(ddf.set_index("feature")["shap"], color="#e45756", horizontal=True)
st.dataframe(ddf[["feature", "value", "shap", "direction"]], hide_index=True,
             use_container_width=True)
st.caption("Positive SHAP = pushes toward LATE. Negative = pushes toward ON TIME.")

# ---- agent ----
st.subheader("🤖 AI dispatch agent")
if st.button("Ask the agent for a recommendation"):
    from routeguard.agent import run_agent

    state = {
        "order": {c: (None if pd.isna(row[c].iloc[0]) else _clean(row[c].iloc[0]))
                  for c in _ORDER_SUMMARY_COLS if c in row},
        "probability": proba,
        "prediction": int(pred_late),
        "threshold": threshold,
        "shap_drivers": drivers,
        "allowed_actions": rgutil.allowed_actions(),
    }
    try:
        with st.spinner("Gemini is analyzing…"):
            final = run_agent(state)
        db.init_db()
        db.save_prediction(model_version=m["version"], probability=proba,
                           prediction="late" if pred_late else "on_time",
                           threshold=threshold, order=state["order"],
                           shap_drivers=drivers, agent=final)
        st.session_state["agent_result"] = final
    except Exception as e:
        st.warning(f"The agent needs GEMINI_API_KEY in your .env file.\n\n"
                   f"({type(e).__name__}: {e})")

if st.session_state.get("agent_result"):
    final = st.session_state["agent_result"]
    st.markdown(f"**Recommended action:** `{final['recommended_action']}`")
    if final["guardrail"] == "approve":
        st.success("Guardrail: approved ✅")
    else:
        st.error("Guardrail: rejected → escalated to manual review")
    st.markdown(f"**Explanation:** {final['explanation']}")
    st.markdown(f"**Reasoning:** {final['reasoning']}")
    st.info(final["confidence_statement"])
