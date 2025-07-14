# 📊 Streamlit App to Analyze Options Greeks and Provide Buy Signals (Robust Version)

import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Options Greek Signal Analyzer", layout="wide")
st.title("📈 Options Greeks Buy Signal Analyzer (Enhanced with Technicals)")

refresh_clicked = st.button("🔄 Refresh App")
ticker = st.text_input("Enter Ticker Symbol (e.g., IWM):", value="IWM", key="ticker" if not refresh_clicked else "ticker_refreshed")

def get_expiries(ticker):
    try:
        return yf.Ticker(ticker).options
    except:
        return []

expiries = get_expiries(ticker)
if expiries:
    expiry = st.selectbox("Select Expiry Date:", expiries)
else:
    st.warning("No expiry dates available. Please check the ticker.")
    st.stop()

# FIX: Ensure serializable output by converting to dict and reconstructing DataFrames
@st.cache_data(show_spinner=False)
def get_option_chain(ticker, expiry):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)
    # Convert to serializable dict format
    calls_dict = chain.calls.to_dict(orient='list')
    puts_dict = chain.puts.to_dict(orient='list')
    return calls_dict, puts_dict

calls_dict, puts_dict = get_option_chain(ticker, expiry)
calls_df = pd.DataFrame(calls_dict)
puts_df = pd.DataFrame(puts_dict)

st.subheader("Top Signals Across All Strikes")

# Load stock data
data = yf.download(ticker, period="2mo", interval="1d")
if data.empty:
    st.error("No data available for this ticker. Please try a different symbol.")
    st.stop()

# FIX: Handle technical indicator calculations with try-except
try:
    # Calculate EMAs using pandas native EWMA
    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    # Calculate RSI manually
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = data['RSI'].fillna(50)  # Fill NaN with neutral 50
    
    # Calculate VWAP
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    
    # Handle NaN values
    data = data.fillna(method='ffill').fillna(method='bfill')
except Exception as e:
    st.error(f"Error calculating technical indicators: {e}")
    # Create placeholder columns if calculations fail
    data['EMA9'] = data['Close']
    data['EMA21'] = data['Close']
    data['RSI'] = 50  # Neutral RSI
    data['VWAP'] = data['Close']

# Validate data for charting
if data.empty or data[['Close', 'EMA9', 'EMA21', 'VWAP']].isna().all().any():
    st.error("Insufficient or invalid data for charting. Please check the ticker or data period.")
    st.stop()

# FIX: Properly extract scalar values and handle missing keys
latest = {}
for col in ['Close', 'EMA9', 'EMA21', 'RSI', 'VWAP']:
    try:
        # Ensure we get a scalar value, not a Series
        value = data[col].iloc[-1] if col in data.columns else data['Close'].iloc[-1]
        # Convert to native Python float if it's a pandas object
        latest[col] = float(value) if hasattr(value, 'item') else value
    except:
        latest[col] = float(data['Close'].iloc[-1])  # Fallback to Close price

# FIX: Convert conditions to native Python booleans
ema_condition = bool(latest.get('EMA9', 0) > latest.get('EMA21', 0))
vwap_condition = bool(latest.get('Close', 0) > latest.get('VWAP', 0))
rsi = float(latest.get('RSI', 50))  # Default to neutral 50 if missing

# FIX: Handle NaN values in Greek calculations
def safe_get_value(row, key, default=0):
    value = row.get(key, default)
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)  # Ensure numeric type
    except:
        return default

# Combined scoring
results = []
for df, option_type in [(calls_df, 'call'), (puts_df, 'put')]:
    for _, row in df.iterrows():
        score = 0
        
        # Safely extract values with NaN handling
        delta = safe_get_value(row, 'delta')
        gamma = safe_get_value(row, 'gamma')
        theta = safe_get_value(row, 'theta')
        vega = safe_get_value(row, 'vega')
        vol = safe_get_value(row, 'volume', 0)
        oi = safe_get_value(row, 'openInterest', 0)

        if option_type == 'call':
            if delta >= 0.6: score += 30
            if gamma >= 0.1: score += 30
            if theta <= 0.03: score += 20
            if vega >= 0.1: score += 20
            if rsi < 30: score += 10
        else:
            if delta <= -0.6: score += 30
            if gamma >= 0.1: score += 30
            if theta <= 0.03: score += 20
            if vega >= 0.1: score += 20
            if rsi > 70: score += 10

        # Use native Python booleans in conditions
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
            'score': score
        })

# Show sorted signal table
if results:
    ranked = pd.DataFrame(results).sort_values(by='score', ascending=False)
    # Add buy signal based on score threshold (e.g., 50)
    ranked['Buy Signal'] = ranked['score'].apply(lambda x: 'buy' if x >= 50 else 'no')
    st.dataframe(ranked[['contract', 'strike', 'type', 'price', 'volume', 'openInterest', 'score', 'Buy Signal']].reset_index(drop=True))
else:
    st.warning("No options data available for scoring.")

# Price Chart
st.subheader("📊 Price Chart with EMA, RSI, VWAP")
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Close', color='blue')
plt.plot(data.index, data['EMA9'], label='EMA 9', color='cyan')
plt.plot(data.index, data['EMA21'], label='EMA 21', color='red')
plt.plot(data.index, data['VWAP'], label='VWAP', color='green')
plt.title(f"{ticker} Price Chart")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
st.pyplot(plt)

# Add current date and time
current_time = datetime.now().strftime("%I:%M %p +01 on %B %d, %Y")
st.write(f"**Latest RSI:** {rsi:.2f} | **EMA9 > EMA21:** {ema_condition} | **Price > VWAP:** {vwap_condition} | **Analysis Date:** {current_time}")
