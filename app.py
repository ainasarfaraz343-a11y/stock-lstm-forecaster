import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Stock Forecaster Pro", layout="wide")

st.title("Advanced Stock Price Prediction Dashboard")
st.write("This dashboard trains a live Bidirectional LSTM model on real-time multivariate technical data to project a 7-day future horizon path.")

# Sidebar Configuration Settings
st.sidebar.header("Model Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, MSFT):", value="AAPL").upper()
lookback_days = st.sidebar.slider("Historical Lookback Window (Days):", min_value=30, max_value=90, value=60)
epochs_count = st.sidebar.slider("Training Optimization Epochs:", min_value=5, max_value=25, value=15)

if st.sidebar.button("Train Model & Forecast Future Horizon"):
    with st.spinner(f"Sourcing live feeds and optimizing neural network weights for {ticker}... Please wait."):
        
        # 1. Fetch Asset Data and S&P 500 Macro Proxy simultaneously
        raw_asset = yf.download(ticker, start='2018-01-01')
        raw_macro = yf.download('SPY', start='2018-01-01')
        
        if raw_asset.empty or raw_macro.empty:
            st.error("Invalid Ticker Symbol or Data Fetch Failure! Please check the input framework.")
        else:
            # 2. Syncing Datasets and Technical Feature Engineering
            df_pro = raw_asset[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
            df_pro['Market_Close'] = raw_macro['Close']
            
            # 14-Day Simple Moving Average for trend velocity mapping
            df_pro['MA14'] = df_pro['Close'].rolling(window=14).mean()
            # 10-Day historical returns standard deviation for active volatility scoring
            df_pro['Volatility'] = df_pro['Close'].pct_change().rolling(window=10).std()
            df_pro.dropna(inplace=True)
            
            # Display live preview data panel
            st.subheader(f"Live Multivariate Technical Feed Preview ({ticker})")
            st.dataframe(df_pro.tail(5))
            
            # 3. Multi-Variate Scaling Layout Matrix
            dataset_pro = df_pro.values
            scaler_features = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler_features.fit_transform(dataset_pro)
            
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaled_target = scaler_target.fit_transform(df_pro[['Close']].values)
            
            FORECAST_DAYS = 7
            
            # 4. Building 3D Tensor Window Sequences
            X_pro, y_pro = [], []
            for i in range(lookback_days, len(scaled_features) - FORECAST_DAYS + 1):
                X_pro.append(scaled_features[i-lookback_days:i])
                y_pro.append(scaled_target[i:i+FORECAST_DAYS, 0])
                
            X_pro, y_pro = np.array(X_pro), np.array(y_pro)
            
            # 5. Compile and Synthesize Bidirectional LSTM Network Architecture
            live_network = Sequential([
                Bidirectional(LSTM(units=90, return_sequences=True), input_shape=(X_pro.shape[1], X_pro.shape[2])),
                Dropout(0.2),
                LSTM(units=60, return_sequences=False),
                Dropout(0.2),
                Dense(units=32, activation='relu'),
                Dense(units=FORECAST_DAYS)
            ])
            
            live_network.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Live Training execution step
            live_network.fit(X_pro, y_pro, epochs=epochs_count, batch_size=32, verbose=0)
            
            # 6. Extrapolating Future Horizon (Next 7 Trading Days)
            terminal_window = scaled_features[-lookback_days:]
            terminal_window = np.expand_dims(terminal_window, axis=0)
            
            future_prediction_scaled = live_network.predict(terminal_window)
            future_prices = scaler_target.inverse_transform(future_prediction_scaled)[0]
            
            # 7. Rendering Dynamic Web Metrics Dashboards
            st.success("Neural optimization cycle finalized successfully!")
            
            current_close = float(df_pro['Close'].iloc[-1])
            tomorrow_est = float(future_prices[0])
            price_delta = tomorrow_est - current_close
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Asset Close", f"${current_close:.2f}")
            m2.metric("Predicted Tomorrow", f"${tomorrow_est:.2f}")
            m3.metric("Expected Next-Day Movement", f"${price_delta:.2f}", delta=f"{price_delta:.2f}")
            
            # 8. Rendering Clean Trajectory Path Graphs
            st.subheader(f"Predicted 7-Day Asset Trajectory Future Horizon ({ticker})")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(range(1, 8), future_prices, marker='o', color='purple', linewidth=2.5, label='LSTM Predicted Path Vector')
            ax.set_xlabel("Days Ahead (Future Horizon Index Space)")
            ax.set_ylabel("Theoretical Valuations ($)")
            ax.set_xticks(range(1, 8))
            ax.set_xticklabels([f"Day {i}" for i in range(1, 8)])
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best')
            st.pyplot(fig)
use_container_width=True)
