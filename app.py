# =========================================================
# STREAMLIT APP: Supply Chain Replenishment Assistant with Time Series
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
model = joblib.load("replenishment_model.pkl")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def map_qualitative_inputs(criticality, cost, rating):
    criticality_map = {"Low": 1, "Moderate": 3, "High": 5}
    cost_map = {"Low": 300, "Moderate": 1000, "High": 1800}
    rating_map = {"Low": 4, "Moderate": 7, "High": 9}
    return criticality_map[criticality], cost_map[cost], rating_map[rating]

def business_interpretation(pred):
    if pred < 400:
        return "✅ Low consumption — maintain standard reorder cycle."
    elif pred < 700:
        return "🟡 Moderate usage — monitor supplier lead time closely."
    else:
        return "🔴 High demand — prioritize procurement and buffer stock."

# Generate synthetic past demand (for demonstration)
def generate_past_demand(item_id):
    np.random.seed(int(item_id[-2:]) * 3)  # same pattern for same item
    months = pd.date_range(end=pd.Timestamp.today(), periods=12, freq="M")
    base = np.random.randint(300, 800)
    variation = np.random.normal(0, 40, 12)
    data = pd.DataFrame({
        "Month": months.strftime("%b-%Y"),
        "Past Demand (units)": np.maximum(base + variation, 0).round(0)
    })
    return data

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Supply Chain Replenishment Assistant", page_icon="📦", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    .main { background-color: #111418; }
    h1, h2, h3, h4, h5 { color: #29b5e8; font-weight: 700; }
    .stButton>button {
        background-color: #29b5e8; color: white; border: none; border-radius: 8px;
        padding: 0.6em 1.2em; font-weight: bold; font-size: 16px;
    }
    .stButton>button:hover { background-color: #1a8bbd; }
    </style>
""", unsafe_allow_html=True)

st.title("🏭 Supply Chain Replenishment Assistant")
st.write("Make smarter replenishment decisions based on historical usage and operational insights — no technical jargon required.")

st.divider()

# ---------------------------------------------------------
# Manager-Friendly Inputs
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    item_id = st.selectbox("Select Material / Part ID", [f"P{str(i).zfill(3)}" for i in range(1, 101)])
    past_demand = st.number_input("Recent Monthly Consumption (units)", 50, 2000, 600, step=50)
    lead_time = st.radio("Supplier Delivery Speed", ["Fast", "Average", "Slow"])
    service = st.radio("Stock Availability Target", ["90% - Basic", "95% - Standard", "99% - Premium"])

with col2:
    uptime = st.radio("Equipment Reliability", ["Low", "Moderate", "High"])
    seasonality = st.radio("Demand Pattern", ["Stable", "Slightly Seasonal", "Highly Seasonal"])
    scrap_rate = st.slider("Wastage / Scrap %", 0.0, 0.15, 0.05, step=0.01)

st.markdown("### Manager Assessment")
criticality = st.radio("Part Importance", ["Low", "Moderate", "High"], horizontal=True)
cost = st.radio("Cost Sensitivity", ["Low", "Moderate", "High"], horizontal=True)
supplier_rating = st.radio("Supplier Reliability", ["Low", "Moderate", "High"], horizontal=True)

# ---------------------------------------------------------
# Time Series Demand Visualization
# ---------------------------------------------------------
st.markdown("### 📈 Historical Consumption Trend")
past_data = generate_past_demand(item_id)

fig = px.line(
    past_data,
    x="Month",
    y="Past Demand (units)",
    markers=True,
    line_shape="spline",
    color_discrete_sequence=["#29b5e8"]
)
fig.update_layout(
    title=f"Demand Trend for {item_id} (Last 12 Months)",
    xaxis_title="Month",
    yaxis_title="Units Consumed",
    plot_bgcolor="#111418",
    paper_bgcolor="#111418",
    font=dict(color="white")
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Convert Qualitative Inputs to Numeric
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
# Prediction Section
# ---------------------------------------------------------
if st.button("📊 Predict Replenishment Quantity"):
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

    prediction = model.predict(input_data)[0]

    st.success(f"📦 **Recommended Replenishment Quantity: {round(prediction, 2)} units**")
    st.info(business_interpretation(prediction))

st.divider()
st.caption("Developed by Ramanujam • MSBA Candidate, Arizona State University")
