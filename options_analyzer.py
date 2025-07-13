# 📊 Streamlit App to Analyze Options Greeks and Provide Buy Signals (Robust Version)

import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

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

# FIX: Replace problematic TA library calculations with pandas native methods
# Calculate EMAs using pandas native EWMA
data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()

# Calculate RSI manually
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Calculate VWAP
data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

# Handle NaN values
data = data.fillna(method='ffill').fillna(method='bfill')

latest = data.iloc[-1]
ema_condition = latest['EMA9'] > latest['EMA21']
vwap_condition = latest['Close'] > latest['VWAP']
rsi = latest['RSI']

# FIX: Handle NaN values in Greek calculations
def safe_get_value(row, key, default=0):
    value = row.get(key, default)
    if pd.isna(value) or value is None:
        return default
    return value

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
        vol = safe_get_value(row, 'volume')
        oi = safe_get_value(row, 'openInterest')

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

        if ema_condition: score += 5
        if vwap_condition: score += 5
        if vol > 100 and oi > 200: score += 10

        results.append({
            'contract': row.get('contractSymbol', 'N/A'),
            'strike': row.get('strike', 0),
            'type': option_type,
            'price': row.get('lastPrice', 0),
            'volume': vol,
            'openInterest': oi,
            'score': score
        })

# Show sorted signal table
ranked = pd.DataFrame(results).sort_values(by='score', ascending=False)
st.dataframe(ranked.reset_index(drop=True))

# Price Chart
st.subheader("📊 Price Chart with EMA, RSI, VWAP")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA9'], mode='lines', name='EMA 9'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA21'], mode='lines', name='EMA 21'))
fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

st.write(f"**Latest RSI:** {rsi:.2f} | **EMA9 > EMA21:** {ema_condition} | **Price > VWAP:** {vwap_condition}")
