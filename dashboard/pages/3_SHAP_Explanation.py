"""SHAP Explanation — which features drive delay risk overall (global importance)."""

import streamlit as st

import rgutil

st.set_page_config(page_title="SHAP · RouteGuard", page_icon="🧭", layout="wide")
st.title("🧭 SHAP Explanation")
st.caption("Global feature importance = average absolute SHAP value across many orders. "
           "This is what the model relies on overall; the Prediction page shows per-order drivers.")

imp = rgutil.global_importance()
st.bar_chart(imp.set_index("feature")["mean_abs_shap"], horizontal=True)
st.dataframe(imp, hide_index=True, use_container_width=True)

st.info("The top driver is usually **delivery_window_days** — a tight promised window is the "
        "strongest signal of delay risk, which matches real dispatch intuition.")
