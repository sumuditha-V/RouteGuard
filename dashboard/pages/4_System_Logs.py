"""System Logs — recommendations logged to the database (from the Prediction page / API)."""

import pandas as pd
import streamlit as st

import rgutil  # noqa: F401  (ensures src is on the path)
from routeguard import db

st.set_page_config(page_title="Logs · RouteGuard", page_icon="🗂️", layout="wide")
st.title("🗂️ System Logs")
st.caption("Every agent recommendation is persisted to SQLite for auditability.")

db.init_db()
rows = db.list_predictions(100)

if not rows:
    st.info("No recommendations logged yet. Go to the Prediction page and click "
            "'Ask the agent' to create one.")
    st.stop()

table = pd.DataFrame([{
    "id": r["id"],
    "time (UTC)": r["created_at"][:19],
    "probability": r["probability"],
    "prediction": r["prediction"],
    "action": (r["agent"] or {}).get("recommended_action"),
    "guardrail": (r["agent"] or {}).get("guardrail"),
} for r in rows])
st.dataframe(table, hide_index=True, use_container_width=True)

st.subheader("Inspect one")
pred_id = st.selectbox("Prediction id", table["id"].tolist())
detail = db.get_prediction(int(pred_id))
if detail:
    st.json(detail)
