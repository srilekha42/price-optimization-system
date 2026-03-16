import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------

# Page Config

# -----------------------------

st.set_page_config(page_title="Price Optimization System", layout="wide")

# -----------------------------

# Load trained model

# -----------------------------

model = joblib.load("outputs/pricing_model.pkl")

st.title("📊 Price Optimization Dashboard")
st.write("Predict product demand and identify the optimal price that maximizes revenue.")

# -----------------------------

# Simulate price range

# -----------------------------

price_range = np.linspace(20, 400, 60)

# Get feature names used in training

features = model.feature_names_in_

# Create input dataframe

input_df = pd.DataFrame(np.zeros((len(price_range), len(features))), columns=features)

# Assign price-related features

input_df["unit_price"] = price_range

if "total_price" in features:
    input_df["total_price"] = price_range

if "lag_price" in features:
    input_df["lag_price"] = price_range

# -----------------------------

# Predict demand

# -----------------------------

demands = model.predict(input_df)

# Ensure demand is non-negative

demands = np.maximum(demands, 0)

# -----------------------------

# Revenue calculation

# -----------------------------

revenues = price_range * demands

# -----------------------------

# Find optimal price

# -----------------------------

optimal_index = np.argmax(revenues)
optimal_price = price_range[optimal_index]
optimal_revenue = revenues[optimal_index]
optimal_demand = demands[optimal_index]

# -----------------------------

# Show key metrics

# -----------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Optimal Price", f"₹{optimal_price:.2f}")
col2.metric("Expected Demand", f"{optimal_demand:.2f} units")
col3.metric("Expected Revenue", f"₹{optimal_revenue:.2f}")

# -----------------------------

# Plot revenue curve

# -----------------------------

fig, ax = plt.subplots()

ax.plot(price_range, revenues, label="Revenue Curve")

# Highlight optimal price

ax.scatter(optimal_price, optimal_revenue, color="red", s=100, label="Optimal Price")

ax.set_xlabel("Price")
ax.set_ylabel("Revenue")
ax.set_title("Price vs Revenue Curve")

ax.legend()

st.pyplot(fig)

# -----------------------------

# Price simulation tool

# -----------------------------

st.subheader("Try Your Own Price")

user_price = st.slider("Select a price", 20.0, 400.0, 100.0)

user_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)

user_input["unit_price"] = user_price

if "total_price" in features:
    user_input["total_price"] = user_price

if "lag_price" in features:
    user_input["lag_price"] = user_price

predicted_demand = model.predict(user_input)[0]
predicted_revenue = user_price * predicted_demand

st.write(f"Predicted Demand: {predicted_demand:.2f} units")
st.write(f"Predicted Revenue: ₹{predicted_revenue:.2f}")
