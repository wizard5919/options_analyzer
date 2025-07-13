# 📊 Streamlit App to Analyze Options Greeks and Provide Buy Signals
import yfinance as yf
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Options Greek Signal Analyzer", layout="wide")
st.title("📈 Options Greeks Buy Signal Analyzer")

# --- User Inputs ---
ticker = st.text_input("Enter Ticker Symbol (e.g., IWM):", value="IWM")

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

# Analysis Function
def analyze_greeks(option_row, option_type="call"):
    if option_row.empty:
        return "❌ No data for analysis"
    
    # Get Greek values, checking if column exists
    delta = option_row.iloc[0]['delta'] if 'delta' in option_row.columns else 0
    gamma = option_row.iloc[0]['gamma'] if 'gamma' in option_row.columns else 0
    theta = option_row.iloc[0]['theta'] if 'theta' in option_row.columns else 0
    vega = option_row.iloc[0]['vega'] if 'vega' in option_row.columns else 0

    if option_type == "call":
        if delta >= 0.6 and gamma >= 0.1 and theta <= 0.03:
            return "📈 Call Buy Signal"
    elif option_type == "put":
        if delta <= -0.6 and gamma >= 0.1 and theta <= 0.03:
            return "📉 Put Buy Signal"

    return "❌ No Buy Signal"

# Display Call Option Info
st.subheader("Call Option Analysis")
if not calls.empty:
    # Select columns to display
    display_cols_calls = ['contractSymbol', 'lastPrice']
    for col in ['delta', 'gamma', 'theta', 'vega']:
        if col in calls.columns:
            display_cols_calls.append(col)
    st.dataframe(calls[display_cols_calls])
    call_signal = analyze_greeks(calls, "call")
    st.success(f"Call Signal: {call_signal}")
else:
    st.error("No call data available for selected strike.")

# Display Put Option Info
st.subheader("Put Option Analysis")
if not puts.empty:
    # Select columns to display
    display_cols_puts = ['contractSymbol', 'lastPrice']
    for col in ['delta', 'gamma', 'theta', 'vega']:
        if col in puts.columns:
            display_cols_puts.append(col)
    st.dataframe(puts[display_cols_puts])
    put_signal = analyze_greeks(puts, "put")
    st.success(f"Put Signal: {put_signal}")
else:
    st.error("No put data available for selected strike.")