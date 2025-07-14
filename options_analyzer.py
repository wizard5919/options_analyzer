import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
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
    
    # Ensure the data is a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name='Close')  # Convert Series to DataFrame with a named column
    elif data.empty:
        raise ValueError("No data returned for the given ticker.")
    
    # Drop any rows with NaN values
    data.dropna(inplace=True)
    return data

def compute_indicators(df):
    # Ensure columns are Series (1D) and convert to float
    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)

    # Compute technical indicators
    df['EMA_9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['avg_vol'] = df['Volume'].rolling(window=20).mean()
    
    # Drop rows with NaN values after indicator computation
    df.dropna(inplace=True)
    return df

def fetch_all_expiries_data(ticker, expiries):
    stock = yf.Ticker(ticker)
    all_calls = pd.DataFrame()
    all_puts = pd.DataFrame()
    for expiry in expiries:
        try:
            chain = stock.option_chain(expiry)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
        except Exception as e:
            st.warning(f"Failed to fetch options data for expiry {expiry}: {e}")
            continue
    return all_calls, all_puts

def classify_moneyness(row, spot):
    if row['strike'] < spot:
        return 'ITM'
    elif row['strike'] == spot:
        return 'ATM'
    else:
        return 'OTM'

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
    except Exception as e:
        st.warning(f"Error generating signal for {side}: {e}")
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
        st.write("Fetching stock and options data...")
        df = get_stock_data(ticker)
        
        if df.empty:
            st.error("No valid stock data available for the given ticker.")
            st.stop()

        df = compute_indicators(df)
        
        if df.empty:
            st.error("No valid data after computing indicators.")
            st.stop()

        stock = yf.Ticker(ticker)
        all_expiries = stock.options
        expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"])

        today = datetime.date.today()
        if expiry_mode == "0DTE Only":
            expiries_to_use = [e for e in all_expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
        else:
            expiries_to_use = all_expiries[:3]  # Use first 3 expiries

        if not expiries_to_use:
            st.warning("No options expiries available for the selected mode.")
            st.stop()

        calls, puts = fetch_all_expiries_data(ticker, expiries_to_use)

        if calls.empty and puts.empty:
            st.warning("No options data available for the selected expiries.")
            st.stop()

        strike_range = st.slider("Select Strike Range Around Spot Price:", -10, 10, (-5, 5))
        spot = df['Close'].iloc[-1]
        min_strike = spot + strike_range[0]
        max_strike = spot + strike_range[1]

        calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
        puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()

        calls_filtered['moneyness'] = calls_filtered.apply(lambda row: classify_moneyness(row, spot), axis=1)
        puts_filtered['moneyness'] = puts_filtered.apply(lambda row: classify_moneyness(row, spot), axis=1)

        m_filter = st.multiselect("Filter by Moneyness:", options=["ITM", "ATM", "OTM"], default=["ITM", "ATM", "OTM"])

        calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
        puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]

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
        st.stop()
