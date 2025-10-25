# =========================================================
# STREAMLIT APP: Supply Chain Replenishment Dashboard (Final + Confidence First)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.stats import norm  # for dynamic confidence levels

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
model = joblib.load("replenishment_model.pkl")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def map_qualitative_inputs(criticality, cost, rating):
    """Convert categorical manager inputs to numeric scales."""
    criticality_map = {"Low": 1, "Moderate": 3, "High": 5}
    cost_map = {"Low": 300, "Moderate": 1000, "High": 1800}
    rating_map = {"Low": 4, "Moderate": 7, "High": 9}
    return criticality_map[criticality], cost_map[cost], rating_map[rating]

def business_interpretation(pred):
    """Simple qualitative recommendation based on predicted demand."""
    if pred < 400:
        return "Low consumption — maintain standard reorder cycle."
    elif pred < 700:
        return "Moderate usage — monitor supplier lead time closely."
    else:
        return "High demand — prioritize procurement and buffer stock."

def generate_past_demand(item_id):
    """Generate synthetic 12-month demand pattern for visualization."""
    np.random.seed(hash(item_id) % (2**32))
    months = pd.date_range(end=pd.Timestamp.today(), periods=12, freq="M")
    base = np.random.randint(300, 800)
    variation = np.random.normal(0, 40, 12)
    data = pd.DataFrame({
        "Month": months,
        "Past Demand": np.maximum(base + variation, 0).round(0).astype(int)
    })
    return data

def forecast_demand(df):
    """Create a simple linear regression forecast for the next 3 months."""
    df = df.reset_index(drop=True)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Past Demand"]
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(df), len(df) + 3).reshape(-1, 1)
    forecast = np.maximum(model.predict(future_X), 0).round(0).astype(int)
    future_months = pd.date_range(df["Month"].iloc[-1] + pd.offsets.MonthBegin(), periods=3, freq="M")
    forecast_df = pd.DataFrame({"Month": future_months, "Forecasted Demand": forecast})
    return forecast_df

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Supply Chain Replenishment Dashboard", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    .main { background-color: #111418; }
    h1, h2, h3, h4, h5 { color: #29b5e8; font-weight: 700; }
    .stButton>button {
        background-color: #29b5e8; color: white; border: none; border-radius: 6px;
        padding: 0.5em 1.2em; font-weight: 600; font-size: 15px;
    }
    .stButton>button:hover { background-color: #1a8bbd; }
    </style>
""", unsafe_allow_html=True)

st.title("Supply Chain Replenishment Dashboard")
st.write("A predictive decision-support application for planning semiconductor spare parts replenishment.")

st.divider()

# ---------------------------------------------------------
# Input Section
# ---------------------------------------------------------
col1, col2 = st.columns(2)

# Realistic semiconductor spare part list
part_list = [
    "IC-Controller-XT150", "MOSFET-Power-Transistor", "Capacitor-47uF-16V",
    "Resistor-220ohm-SMD", "Microchip-ATMega328", "Diode-Schottky-SS14",
    "Crystal-Oscillator-16MHz", "Connector-USB-TypeC", "Transistor-BC547",
    "Voltage-Regulator-LM7805", "Sensor-Thermal-TMP36", "LED-Green-5mm",
    "Relay-5V-DC", "Display-LCD-16x2", "Switch-Tactile-SPST"
]

with col1:
    item_id = st.selectbox("Select Material / Component", part_list)
    past_demand = st.number_input("Recent Monthly Consumption (units)", 50, 2000, 600, step=50)
    lead_time = st.radio("Supplier Delivery Speed", ["Fast", "Average", "Slow"])
    service = st.radio("Target Service Level", ["90% - Basic", "95% - Standard", "99% - Premium"])

with col2:
    uptime = st.radio("Equipment Reliability", ["Low", "Moderate", "High"])
    seasonality = st.radio("Demand Pattern", ["Stable", "Slightly Seasonal", "Highly Seasonal"])
    scrap_rate = st.slider("Wastage / Scrap (%)", 0.0, 0.15, 0.05, step=0.01)

st.markdown("#### Business Inputs")
criticality = st.radio("Part Importance", ["Low", "Moderate", "High"], horizontal=True)
cost = st.radio("Cost Category", ["Low", "Moderate", "High"], horizontal=True)
supplier_rating = st.radio("Supplier Reliability", ["Low", "Moderate", "High"], horizontal=True)

# ---------------------------------------------------------
# Confidence Level Selection (Before Prediction)
# ---------------------------------------------------------
st.divider()
st.markdown("#### Confidence Level for Safety Stock Calculation")
confidence_level = st.slider("Set Confidence Level (%)", 80, 99, 95, step=1)
z_value = round(norm.ppf(confidence_level / 100), 2)
st.write(f"Selected Confidence Level: **{confidence_level}%** (Z = {z_value})")

# ---------------------------------------------------------
# Time-Series Demand Visualization
# ---------------------------------------------------------
past_data = generate_past_demand(item_id)
forecast_df = forecast_demand(past_data)
combined = past_data.merge(forecast_df, how="outer", on="Month")

fig = px.line(
    combined, x="Month", y=["Past Demand", "Forecasted Demand"],
    line_shape="spline", markers=True, color_discrete_sequence=["#29b5e8", "#ffaa00"]
)
fig.update_layout(
    title=f"Historical and Forecasted Demand for {item_id}",
    xaxis_title="Month", yaxis_title="Units",
    plot_bgcolor="#111418", paper_bgcolor="#111418",
    font=dict(color="white")
)
fig.update_yaxes(tickformat="d")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Convert Inputs and Run Model
# ---------------------------------------------------------
lead_time_map = {"Fast": 7, "Average": 15, "Slow": 30}
service_map = {"90% - Basic": 90, "95% - Standard": 95, "99% - Premium": 99}
seasonality_map = {"Stable": 1.0, "Slightly Seasonal": 1.1, "Highly Seasonal": 1.25}
uptime_map = {"Low": 80, "Moderate": 90, "High": 98}

criticality_score, unit_cost, rating = map_qualitative_inputs(criticality, cost, supplier_rating)

lead_time_days = lead_time_map[lead_time]
service_level = service_map[service]
seasonality_index = seasonality_map[seasonality]
avg_uptime_percent = uptime_map[uptime]

# ---------------------------------------------------------
# Prediction and KPI Calculations
# ---------------------------------------------------------
if st.button("Run Analysis and Predict Replenishment"):

    input_data = pd.DataFrame({
        "past_demand": [past_demand],
        "lead_time_days": [lead_time_days],
        "supplier_rating": [rating],
        "criticality_score": [criticality_score],
        "seasonality_index": [seasonality_index],
        "unit_cost": [unit_cost],
        "avg_uptime_percent": [avg_uptime_percent],
        "service_level": [service_level / 100],
        "scrap_rate": [scrap_rate]
    })

    predicted_qty = int(round(model.predict(input_data)[0]))

    # Safety Stock and Reorder Point using selected confidence
    demand_std = past_data["Past Demand"].std()
    safety_stock = int(round(z_value * demand_std * np.sqrt(lead_time_days / 30)))
    reorder_point = int(round((past_demand * (lead_time_days / 30)) + safety_stock))

    # KPI Summary
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Predicted Replenishment", f"{predicted_qty} units")
    kpi2.metric("Safety Stock", f"{safety_stock} units")
    kpi3.metric("Reorder Point", f"{reorder_point} units")

    st.markdown("#### Business Recommendation")
    st.write(business_interpretation(predicted_qty))

st.divider()
st.caption("Developed by Venkata Ramanujam Kandalam • MSBA Candidate, Arizona State University")
