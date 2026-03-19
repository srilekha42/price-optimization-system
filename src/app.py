import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Price Optimization", layout="wide")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("outputs/pricing_model.pkl")
features = model.feature_names_in_

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

cost = st.sidebar.slider("Cost Price (₹)", 50, 200, 100)
min_price = st.sidebar.slider("Min Price", 10, 100, 20)
max_price = st.sidebar.slider("Max Price", 200, 500, 400)

# -----------------------------
# Title
# -----------------------------
st.markdown("# 📊 Price Optimization Dashboard")
st.markdown("### Maximize profit using machine learning-based pricing strategy")

st.info("💡 Optimal price is calculated by maximizing profit = (price - cost) × predicted demand")

# -----------------------------
# Price Range
# -----------------------------
price_range = np.linspace(min_price, max_price, 60)

# -----------------------------
# Feature Preparation
# -----------------------------
mean_values = np.ones(len(features)) * 50  # fallback

input_df = pd.DataFrame(
    np.tile(mean_values, (len(price_range), 1)),
    columns=features
)

# Assign price features
input_df["unit_price"] = price_range

if "total_price" in features:
    input_df["total_price"] = price_range

if "lag_price" in features:
    input_df["lag_price"] = price_range

# -----------------------------
# Prediction
# -----------------------------
demands = model.predict(input_df)
demands = np.clip(demands, 0, 10000)

# -----------------------------
# Profit Calculation
# -----------------------------
profits = (price_range - cost) * demands

# -----------------------------
# Optimal Price
# -----------------------------
optimal_index = np.argmax(profits)
optimal_price = price_range[optimal_index]
optimal_profit = profits[optimal_index]
optimal_demand = demands[optimal_index]

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Optimal Price", f"₹{optimal_price:.2f}")
col2.metric("📦 Demand", f"{optimal_demand:.2f}")
col3.metric("📈 Profit", f"₹{optimal_profit:.2f}")

# -----------------------------
# Graphs
# -----------------------------
st.subheader("📈 Analysis")

colA, colB = st.columns(2)

# Profit Curve
fig1, ax1 = plt.subplots()
ax1.plot(price_range, profits, label="Profit Curve")
ax1.scatter(optimal_price, optimal_profit, color="red", s=100)
ax1.set_title("Price vs Profit")
ax1.set_xlabel("Price")
ax1.set_ylabel("Profit")
ax1.legend()

with colA:
    st.pyplot(fig1)

# Demand Curve
fig2, ax2 = plt.subplots()
ax2.plot(price_range, demands)
ax2.set_title("Price vs Demand")
ax2.set_xlabel("Price")
ax2.set_ylabel("Demand")

with colB:
    st.pyplot(fig2)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📊 Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_df.set_index("Feature"))
else:
    st.warning("Feature importance not available.")

# -----------------------------
# User Simulation
# -----------------------------
st.subheader("🎯 Try Your Own Price")

user_price = st.slider("Select Price", float(min_price), float(max_price), 100.0)

# ✅ FIXED SHAPE ISSUE HERE
user_input = pd.DataFrame([mean_values], columns=features)

user_input["unit_price"] = user_price

if "total_price" in features:
    user_input["total_price"] = user_price

if "lag_price" in features:
    user_input["lag_price"] = user_price

predicted_demand = model.predict(user_input)[0]
predicted_demand = max(predicted_demand, 0)

predicted_profit = (user_price - cost) * predicted_demand

st.success(f"📦 Predicted Demand: {predicted_demand:.2f} units")
st.success(f"📈 Expected Profit: ₹{predicted_profit:.2f}")