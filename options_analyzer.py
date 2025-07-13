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

@st.cache_data(show_spinner=False)
def get_option_chain(ticker, expiry):
    stock = yf.Ticker(ticker)
    return stock.option_chain(expiry)

calls_df, puts_df = get_option_chain(ticker, expiry)

st.subheader("Top Signals Across All Strikes")

# Load stock data and calculate technicals
data = yf.download(ticker, period="2mo", interval="1d")
data['EMA9'] = EMAIndicator(close=data['Close'], window=9).ema_indicator()
data['EMA21'] = EMAIndicator(close=data['Close'], window=21).ema_indicator()
data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

latest = data.iloc[-1]
ema_condition = latest['EMA9'] > latest['EMA21']
vwap_condition = latest['Close'] > latest['VWAP']
rsi = latest['RSI']

# Combined scoring
results = []
for df, option_type in [(calls_df, 'call'), (puts_df, 'put')]:
    for _, row in df.iterrows():
        score = 0
        delta = row.get('delta', 0) or 0
        gamma = row.get('gamma', 0) or 0
        theta = row.get('theta', 0) or 0
        vega = row.get('vega', 0) or 0
        vol = row.get('volume', 0) or 0
        oi = row.get('openInterest', 0) or 0

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
            'contract': row['contractSymbol'],
            'strike': row['strike'],
            'type': option_type,
            'price': row['lastPrice'],
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
