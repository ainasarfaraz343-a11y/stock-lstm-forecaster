import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Asset Forecaster", layout="wide")
st.title("Enterprise Deep Learning Suite: Asset Trajectory")

st.sidebar.header("🕹️ Model Controls")
asset_ticker = st.sidebar.selectbox("Select Target Asset:", ["AAPL", "MSFT", "GOOGL"])
forecast_days_slider = st.sidebar.slider("Select Horizon (Days Ahead):", min_value=1, max_value=7, value=7)

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    raw_asset = yf.download(ticker, start='2018-01-01', end='2026-05-23')
    raw_macro = yf.download('SPY', start='2018-01-01', end='2026-05-23')
    processed_df = raw_asset[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
    processed_df['Market_Index_Proxy'] = raw_macro['Close']
    processed_df['MA14'] = processed_df['Close'].rolling(window=14).mean()
    processed_df['Historical_Volatility'] = processed_df['Close'].pct_change().rolling(window=10).std()
    processed_df.dropna(inplace=True)
    return processed_df

with st.spinner("Fetching Live Market Data..."):
    df = get_market_data(asset_ticker)

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_x.fit_transform(df.values)
scaled_target = scaler_y.fit_transform(df[['Close']].values)

@st.cache_resource
def load_saved_nn_weights():
    return load_model("deep_model.h5", compile=False)

net_engine = load_saved_nn_weights()

LOOKBACK_WINDOWS = 60
terminal_window = scaled_features[-LOOKBACK_WINDOWS:]
terminal_window = np.expand_dims(terminal_window, axis=0)

live_future_scaled = net_engine.predict(terminal_window)
live_future_unscaled = scaler_y.inverse_transform(live_future_scaled)
filtered_predictions = live_future_unscaled[0][:forecast_days_slider]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Dynamic Trajectory Chart")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(1, forecast_days_slider + 1), filtered_predictions, marker='o', color='darkmagenta', linewidth=2.5)
    ax.set_xticks(range(1, forecast_days_slider + 1))
    ax.set_xticklabels([f"Day {i}" for i in range(1, forecast_days_slider + 1)])
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

with col2:
    st.subheader("Calculated Prediction Metrics View")
    results_table = pd.DataFrame({
        "Sequence Target Index": [f"Day {i}" for i in range(1, forecast_days_slider + 1)],
        "Predicted Valuation ($)": [f"${price:.2f}" for price in filtered_predictions]
    })
    st.dataframe(results_table, use_container_width=True)
