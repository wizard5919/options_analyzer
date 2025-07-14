import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

st.set_page_config(page_title="Options Greeks Buy Signal Analyzer", layout="wide")

# =============================
# UTILITY FUNCTIONS
# =============================

def get_stock_data(ticker):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=10)
    data = yf.download(ticker, start=start, end=end, interval="5m")
    data.dropna(inplace=True)
    return data

def compute_indicators(df):
    df['EMA_9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['avg_vol'] = df['Volume'].rolling(window=20).mean()
    return df

def fetch_options_data(ticker, expiry):
    stock = yf.Ticker(ticker)
    try:
        chain = stock.option_chain(expiry)
        return chain.calls, chain.puts
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# =============================
# SIGNAL LOGIC
# =============================

def generate_signal(option, side, stock_df):
    latest = stock_df.iloc[-1]

    try:
        if side == "call":
            if (
                option['delta'] >= 0.6
                and option['gamma'] >= 0.08
                and option['theta'] <= 0.05
                and float(latest['Close']) > float(latest['EMA_9']) > float(latest['EMA_20'])
                and float(latest['RSI']) > 50
                and float(latest['Close']) > float(latest['VWAP'])
                and float(latest['Volume']) > 1.5 * float(latest['avg_vol'])
            ):
                return True
        elif side == "put":
            if (
                option['delta'] <= -0.6
                and option['gamma'] >= 0.08
                and option['theta'] <= 0.05
                and float(latest['Close']) < float(latest['EMA_9']) < float(latest['EMA_20'])
                and float(latest['RSI']) < 50
                and float(latest['Close']) < float(latest['VWAP'])
                and float(latest['Volume']) > 1.5 * float(latest['avg_vol'])
            ):
                return True
    except:
        return False

    return False

# =============================
# STREAMLIT INTERFACE
# =============================

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("This app uses **Greeks + technical indicators** to find smart entries for Calls and Puts.")

ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper()

if ticker:
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        expiry = st.selectbox("Choose Expiry Date:", expiries)

        # Fetch data
        st.write("Fetching stock and options data...")
        df = get_stock_data(ticker)
        df = compute_indicators(df)
        calls, puts = fetch_options_data(ticker, expiry)

        # Select strike
        strike_range = st.slider("Select Strike Range Around Spot Price:", -10, 10, (-5, 5))
        spot = df.iloc[-1]['Close']
        min_strike = spot + strike_range[0]
        max_strike = spot + strike_range[1]

        calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
        puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

        st.subheader("📊 Call Option Signals")
        call_signals = []
        for _, row in calls_filtered.iterrows():
            if generate_signal(row, "call", df):
                call_signals.append(row)
        if call_signals:
            st.dataframe(pd.DataFrame(call_signals).reset_index(drop=True))
        else:
            st.info("No CALL signals matched the criteria.")

        st.subheader("📉 Put Option Signals")
        put_signals = []
        for _, row in puts_filtered.iterrows():
            if generate_signal(row, "put", df):
                put_signals.append(row)
        if put_signals:
            st.dataframe(pd.DataFrame(put_signals).reset_index(drop=True))
        else:
            st.info("No PUT signals matched the criteria.")

    except Exception as e:
        st.error(f"Failed to fetch or analyze data: {e}")
