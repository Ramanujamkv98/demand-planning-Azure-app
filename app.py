# =========================================================
# STREAMLIT APP: Semiconductor Spare Parts Replenishment
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load("replenishment_model.pkl")

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def map_qualitative_inputs(criticality, cost, rating):
    # Convert dropdowns into numeric values (based on synthetic dataset scale)
    criticality_map = {"Low": 1, "Moderate": 3, "High": 5}
    cost_map = {"Low": 300, "Moderate": 1000, "High": 1800}
    rating_map = {"Low": 4, "Moderate": 7, "High": 9}

    return (
        criticality_map[criticality],
        cost_map[cost],
        rating_map[rating]
    )

def business_interpretation(pred):
    if pred < 400:
        return "🟢 Stable demand — maintain standard reorder cycle."
    elif pred < 700:
        return "🟡 Moderate consumption — consider supplier lead time."
    else:
        return "🔴 High replenishment requirement — prioritize procurement and buffer stock."

# ---------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------

st.set_page_config(page_title="Spare Parts Replenishment Predictor", page_icon="⚙️", layout="centered")

st.title("⚙️ Semiconductor Spare Parts Replenishment Assistant")
st.write("Estimate replenishment quantity based on supply chain inputs — no technical jargon required.")

st.divider()

# Input fields
item_id = st.selectbox("Select Item ID", [f"P{str(i).zfill(3)}" for i in range(1, 101)])
past_demand = st.slider("Past Demand (units)", 50, 1200, 600, step=50)
lead_time = st.slider("Supplier Lead Time (days)", 3, 45, 20)
seasonality = st.slider("Seasonality Index (1.0 = neutral)", 0.8, 1.3, 1.05, step=0.01)
uptime = st.slider("Average Equipment Uptime (%)", 70, 99, 92)
service_level = st.slider("Desired Service Level (%)", 90, 99, 95)
scrap_rate = st.slider("Scrap Rate (%)", 0.0, 0.15, 0.05, step=0.01)

st.subheader("Qualitative Inputs (Manager-Friendly)")
criticality = st.radio("Criticality Level", ["Low", "Moderate", "High"], horizontal=True)
cost = st.radio("Cost Range", ["Low", "Moderate", "High"], horizontal=True)
supplier_rating = st.radio("Supplier Rating", ["Low", "Moderate", "High"], horizontal=True)

# Convert qualitative inputs to numeric
criticality_score, unit_cost, rating = map_qualitative_inputs(criticality, cost, supplier_rating)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔍 Predict Replenishment Quantity"):
    input_data = pd.DataFrame({
        "past_demand": [past_demand],
        "lead_time_days": [lead_time],
        "supplier_rating": [rating],
        "criticality_score": [criticality_score],
        "seasonality_index": [seasonality],
        "unit_cost": [unit_cost],
        "avg_uptime_percent": [uptime],
        "service_level": [service_level / 100],
        "scrap_rate": [scrap_rate]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"📦 **Predicted Replenishment Quantity: {round(prediction, 2)} units**")

    st.info(business_interpretation(prediction))

st.divider()
st.caption("Developed by Ram • MSBA Candidate, Arizona State University")
