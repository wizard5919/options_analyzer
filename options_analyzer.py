# 📊 Streamlit App to Analyze Options Greeks and Provide Buy Signals (Enhanced Version)

import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Options Greek Signal Analyzer", layout="wide")
st.title("📈 Options Greeks Buy Signal Analyzer (Enhanced)")

# --- Manual Refresh (Safe Version) ---
refresh_clicked = st.button("🔄 Refresh App")

# --- User Inputs ---
ticker = st.text_input("Enter Ticker Symbol (e.g., IWM):", value="IWM", key="ticker" if not refresh_clicked else "ticker_refreshed")

# Fetch expiry dates
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

# Get option chain
def get_options_chain_with_greeks(ticker_symbol, expiry_date):
    stock = yf.Ticker(ticker_symbol)
    options_chain = stock.option_chain(expiry_date)
    return options_chain.calls, options_chain.puts

calls_df, puts_df = get_options_chain_with_greeks(ticker, expiry)

# Filter strike
all_strikes = sorted(list(set(calls_df['strike']).union(set(puts_df['strike']))))
strike = st.selectbox("Select Strike Price:", all_strikes)
calls = calls_df[calls_df['strike'] == strike]
puts = puts_df[puts_df['strike'] == strike]

# Scoring Function
def score_greeks(option_row, option_type="call"):
    delta = option_row.get('delta', 0) or 0
    gamma = option_row.get('gamma', 0) or 0
    theta = option_row.get('theta', 0) or 0
    vega = option_row.get('vega', 0) or 0

    score = 0
    if option_type == "call":
        if delta >= 0.6: score += 30
        if gamma >= 0.1: score += 30
        if theta <= 0.03: score += 20
        if vega >= 0.1: score += 20
    elif option_type == "put":
        if delta <= -0.6: score += 30
        if gamma >= 0.1: score += 30
        if theta <= 0.03: score += 20
        if vega >= 0.1: score += 20

    return score

def interpret_score(score):
    if score >= 90:
        return f"🔥 Strong Buy Signal ({score})"
    elif score >= 60:
        return f"✅ Moderate Buy Signal ({score})"
    else:
        return f"❌ No Buy Signal ({score})"

# Display Call Option Info
st.subheader("Call Option Analysis")
if not calls.empty:
    call_cols = [col for col in ['contractSymbol', 'lastPrice', 'delta', 'gamma', 'theta', 'vega'] if col in calls.columns]
    st.dataframe(calls[call_cols])
    call_score = score_greeks(calls.iloc[0], "call")
    call_signal = interpret_score(call_score)
    st.success(f"Call Signal: {call_signal}")
else:
    st.error("No call data available for selected strike.")

# Display Put Option Info
st.subheader("Put Option Analysis")
if not puts.empty:
    put_cols = [col for col in ['contractSymbol', 'lastPrice', 'delta', 'gamma', 'theta', 'vega'] if col in puts.columns]
    st.dataframe(puts[put_cols])
    put_score = score_greeks(puts.iloc[0], "put")
    put_signal = interpret_score(put_score)
    st.success(f"Put Signal: {put_signal}")
else:
    st.error("No put data available for selected strike.")

# Price Chart (Optional Technical Confirmation)
st.subheader("📊 Price Chart with EMA")
data = yf.download(ticker, period="1mo", interval="1d")
data['EMA9'] = data['Close'].ewm(span=9).mean()
data['EMA21'] = data['Close'].ewm(span=21).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA9'], mode='lines', name='EMA 9'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA21'], mode='lines', name='EMA 21'))
fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
