# =============================================================
# 📊 STREAMLIT APP — AI-Driven Cashflow Prediction
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor

# -----------------------------
# Load trained models
# -----------------------------
@st.cache_resource
def load_models():
    arima = joblib.load("arima_model.pkl")
    lstm = load_model("lstm_model.keras")
    xgb = XGBRegressor()
    xgb.load_model("xgb_model.json")
    scaler = joblib.load("scaler.pkl")
    return arima, lstm, xgb, scaler

arima_model, lstm_model, xgb_model, scaler = load_models()
st.success("✅ Models loaded successfully!")

# -----------------------------
# App layout
# -----------------------------
st.title("💰 AI-Driven Cashflow Forecasting App")
st.markdown("Predict next-day **net cashflow** and its direction (Positive/Negative) using hybrid AI models (ARIMA + LSTM + XGBoost).")

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Input Financial & Macro Parameters")

inflation_rate_cpi = st.sidebar.number_input("Inflation Rate (CPI) (%)", value=3.5)
interest_rate = st.sidebar.number_input("Interest Rate (%)", value=7.0)
fx_usd_inr = st.sidebar.number_input("USD/INR Exchange Rate", value=80.0)
commodity_price_oil = st.sidebar.number_input("Oil Price (USD/barrel)", value=90.0)
consumer_confidence_index = st.sidebar.number_input("Consumer Confidence Index", value=100.0)
employment_rate = st.sidebar.number_input("Employment Rate (%)", value=98.0)
cash_inflow = st.sidebar.number_input("Cash Inflow (₹)", value=12000.0)
cash_outflow = st.sidebar.number_input("Cash Outflow (₹)", value=7000.0)
lag_1 = st.sidebar.number_input("Previous Day Net Cashflow", value=11000.0)
lag_7 = st.sidebar.number_input("Cashflow (7 days ago)", value=10000.0)
rolling_avg_7 = st.sidebar.number_input("7-Day Rolling Average", value=9500.0)

# -----------------------------
# Make prediction
# -----------------------------
if st.button("🔮 Predict Cashflow"):
    # Prepare input as DataFrame
    features = np.array([[inflation_rate_cpi, interest_rate, fx_usd_inr,
                          commodity_price_oil, consumer_confidence_index,
                          employment_rate, cash_inflow, cash_outflow,
                          lag_1, lag_7, rolling_avg_7]])
    
    feature_df = pd.DataFrame(features, columns=[
        'inflation_rate_cpi','interest_rate','fx_usd_inr',
        'commodity_price_oil','consumer_confidence_index','employment_rate',
        'cash_inflow','cash_outflow','lag_1','lag_7','rolling_avg_7'
    ])

    # ARIMA prediction
    arima_pred = float(arima_model.forecast(steps=1)[0])

    # LSTM prediction
    X_scaled = scaler.transform(feature_df)
    X_scaled = X_scaled.reshape((1, 1, len(feature_df.columns)))
    lstm_pred = float(lstm_model.predict(X_scaled, verbose=0).ravel()[0])

    # XGBoost stacking
    meta_sample = pd.DataFrame({
        "arima_pred": [arima_pred],
        "lstm_pred": [lstm_pred],
        "inflation_rate_cpi": [inflation_rate_cpi],
        "interest_rate": [interest_rate],
        "fx_usd_inr": [fx_usd_inr],
        "commodity_price_oil": [commodity_price_oil],
        "consumer_confidence_index": [consumer_confidence_index],
        "employment_rate": [employment_rate]
    })

    xgb_pred = float(xgb_model.predict(meta_sample)[0])

    # Display results
    st.subheader("📈 Prediction Results")
    st.metric("Predicted Net Cashflow (₹)", f"{xgb_pred:,.2f}")
    direction = "🟢 Positive (Inflow > Outflow)" if xgb_pred > 0 else "🔴 Negative (Outflow > Inflow)"
    st.write(f"**Direction:** {direction}")

    # Show model details
    st.caption(f"ARIMA Output: {arima_pred:,.2f} | LSTM Output: {lstm_pred:,.2f} | Final XGBoost Stack: {xgb_pred:,.2f}")

    # Optional visualization
    st.bar_chart(pd.DataFrame({
        "ARIMA": [arima_pred],
        "LSTM": [lstm_pred],
        "XGBoost (Final)": [xgb_pred]
    }))
