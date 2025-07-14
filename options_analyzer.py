# 📊 Streamlit App to Analyze Options Greeks and Provide Buy Signals (Robust Version)

import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import time

st.set_page_config(page_title="Options Greek Signal Analyzer", layout="wide")
st.title("📈 Options Greeks Buy Signal Analyzer (Enhanced with Technicals)")

# Initialize session state for refresh
if 'refresh_key' not in st.session_state:
    st.session_state.refresh_key = 0

refresh_clicked = st.button("🔄 Refresh App")
if refresh_clicked:
    st.session_state.refresh_key += 1

ticker = st.text_input("Enter Ticker Symbol (e.g., IWM):", value="IWM", key=f"ticker_{st.session_state.refresh_key}")

# Retry logic for yfinance
def fetch_with_retry(func, max_attempts=3, delay=2):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                st.error(f"Failed to fetch data after {max_attempts} attempts: {e}")
                return None
            time.sleep(delay)

def get_expiries(ticker):
    def fetch():
        return yf.Ticker(ticker).options
    return fetch_with_retry(fetch) or []

expiries = get_expiries(ticker)
if not expiries:
    st.warning("No expiry dates available. Please check the ticker.")
    st.stop()

expiry = st.selectbox("Select Expiry Date:", expiries)

# Cache option chain data
@st.cache_data(show_spinner=False)
def get_option_chain(ticker, expiry, _refresh_key):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)
    return chain.calls, chain.puts

calls_df, puts_df = get_option_chain(ticker, expiry, st.session_state.refresh_key)

st.subheader("Top Signals Across All Strikes")

# Fetch stock data with retry
def fetch_stock_data(ticker):
    def fetch():
        return yf.download(ticker, period="2mo", interval="1d", auto_adjust=True)
    return fetch_with_retry(fetch)

data = fetch_stock_data(ticker)
if data is None or data.empty:
    st.error("No data available for this ticker. Please try a different symbol.")
    st.stop()

# Calculate technical indicators
try:
    # EMAs
    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    with np.errstate(divide='ignore', invalid='ignore'):
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = data['RSI'].fillna(50)
    
    # VWAP
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Fill missing values
    data = data.ffill().bfill()
except Exception as e:
    st.error(f"Error calculating technical indicators: {e}")
    data['EMA9'] = data['Close']
    data['EMA21'] = data['Close']
    data['RSI'] = 50
    data['VWAP'] = data['Close']
    data['MACD'] = 0
    data['Signal'] = 0

# Validate data for charting
if data[['Close', 'EMA9', 'EMA21', 'VWAP', 'MACD', 'Signal']].isna().all().any():
    st.error("Insufficient or invalid data for charting. Please check the ticker or data period.")
    st.stop()

# Get real-time price
def get_real_time_price(ticker):
    def fetch():
        return yf.Ticker(ticker).info.get('regularMarketPrice', None)
    return fetch_with_retry(fetch)

current_price = get_real_time_price(ticker)
if current_price:
    st.write(f"**Current Price:** ${current_price:.2f}")

# Extract latest values
latest = data.iloc[-1]
ema_condition = latest['EMA9'] > latest['EMA21']
vwap_condition = latest['Close'] > latest['VWAP']
rsi = float(latest['RSI'])
macd_bullish = latest['MACD'] > latest['Signal']

# Handle NaN values in Greek calculations
def safe_get_value(row, key, default=0):
    value = row.get(key, default)
    return float(value) if not pd.isna(value) and value is not None else default

# Enhanced scoring with market alignment
results = []
for df, option_type in [(calls_df, 'call'), (puts_df, 'put')]:
    for _, row in df.iterrows():
        score = 0
        
        # Extract Greeks
        delta = safe_get_value(row, 'delta')
        gamma = safe_get_value(row, 'gamma')
        theta = safe_get_value(row, 'theta')
        vega = safe_get_value(row, 'vega')
        vol = safe_get_value(row, 'volume', 0)
        oi = safe_get_value(row, 'openInterest', 0)

        # Scoring logic
        if option_type == 'call':
            if delta >= 0.6: score += 30
            if gamma >= 0.1: score += 30
            if theta <= 0.03: score += 20
            if vega >= 0.1: score += 20
            if rsi < 30: score += 10
            if macd_bullish: score += 15
        else:
            if delta <= -0.6: score += 30
            if gamma >= 0.1: score += 30
            if theta <= 0.03: score += 20
            if vega >= 0.1: score += 20
            if rsi > 70: score += 10
            if not macd_bullish: score += 15

        if ema_condition: score += 5
        if vwap_condition: score += 5
        if vol > 100 and oi > 200: score += 10

        results.append({
            'contract': row.get('contractSymbol', 'N/A'),
            'strike': row.get('strike', 0),
            'type': option_type,
            'price': safe_get_value(row, 'lastPrice', 0),
            'volume': vol,
            'openInterest': oi,
            'score': score,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        })

# Display results
if results:
    ranked = pd.DataFrame(results).sort_values(by='score', ascending=False)
    ranked['Buy Signal'] = ranked['score'].apply(lambda x: 'buy' if x >= 50 else 'no')
    st.dataframe(ranked[['contract', 'strike', 'type', 'price', 'volume', 'openInterest', 'delta', 'gamma', 'theta', 'vega', 'score', 'Buy Signal']].reset_index(drop=True))
else:
    st.warning("No options data available for scoring.")

# Price Chart with MACD
st.subheader("📊 Price Chart with EMA, RSI, VWAP, and MACD")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
# Price and indicators
ax1.plot(data.index, data['Close'], label='Close', color='blue')
ax1.plot(data.index, data['EMA9'], label='EMA 9', color='cyan')
ax1.plot(data.index, data['EMA21'], label='EMA 21', color='red')
ax1.plot(data.index, data['VWAP'], label='VWAP', color='green')
ax1.set_title(f"{ticker} Price Chart")
ax1.set_ylabel("Price")
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# RSI
ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
ax2.axhline(70, linestyle='--', color='red')
ax2.axhline(30, linestyle='--', color='green')
ax2.set_ylabel("RSI")
ax2.legend()

# MACD
ax3.plot(data.index, data['MACD'], label='MACD', color='blue')
ax3.plot(data.index, data['Signal'], label='Signal', color='orange')
ax3.set_ylabel("MACD")
ax3.legend()
plt.tight_layout()
st.pyplot(fig)

# Display metrics
current_time = datetime.now().strftime("%I:%M %p +01 on %B %d, %Y")
st.write(f"**Latest RSI:** {rsi:.2f} | **EMA9 > EMA21:** {ema_condition} | **Price > VWAP:** {vwap_condition} | **MACD Bullish:** {macd_bullish} | **Analysis Date:** {current_time}")
