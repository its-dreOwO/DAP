import pickle

import pandas as pd
import shap
import streamlit as st

MODEL_PATH = "Data/filtered/xgb_model.pkl"

st.set_page_config(page_title="Supplier Delay Risk", layout="centered")
st.title("Supplier Lead-Time Risk Predictor")
st.caption("DAP391m — Group 8, FPT University HCMC")

with st.sidebar:
    st.header("Shipment Parameters")
    lead_time_days = st.number_input(
        "Lead Time (days)", min_value=1, max_value=120, value=14
    )
    customs_clearance_time = st.number_input(
        "Customs Clearance Time (days)", min_value=0, max_value=30, value=3
    )
    freight_cost = st.number_input(
        "Freight Cost (USD)", min_value=0, max_value=50000, value=1500
    )
    delay_hours_avg = st.number_input(
        "Carrier Avg Delay (hours)", min_value=0.0, max_value=200.0, value=12.0
    )
    warehouse_util = st.slider("Warehouse Utilisation (%)", 0, 100, 75)
    satisfaction_score = st.slider("Supplier Satisfaction Score", 1, 5, 3)
    predict_btn = st.button("Predict Delay Risk", type="primary")

if predict_btn:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        features = pd.DataFrame(
            [
                {
                    "lead_time_days": lead_time_days,
                    "customs_clearance_time_days": customs_clearance_time,
                    "freight_cost": freight_cost,
                    "delay_hours_avg": delay_hours_avg,
                    "warehouse_utilization_percent": warehouse_util,
                    "satisfaction_score": satisfaction_score,
                }
            ]
        )

        prob = model.predict_proba(features)[0][1]
        risk_label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
        color = "red" if prob >= 0.5 else "green"

        st.markdown(f"## Delay Probability: `{prob:.1%}`")
        st.markdown(f"**Risk Level:** :{color}[{risk_label}]")
        st.progress(float(prob))

        st.subheader("SHAP Explanation")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(features)
        fig, ax = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning(
            f"Model not found at `{MODEL_PATH}`. Run `src-code/05_modeling.py` first."
        )
