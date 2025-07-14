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

    if data.empty:
        return pd.DataFrame()

    # Handle multi-level columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    # Drop rows with all NaN values
    data.dropna(how="all", inplace=True)

    # Ensure correct data types and handle any nested structures
    for col in ['Close', 'High', 'Low', 'Volume']:
        if col in data.columns:
            # Convert to numeric, handling any nested structures
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Remove rows where essential columns are NaN
    essential_cols = [col for col in ['Close', 'High', 'Low', 'Volume'] if col in data.columns]
    data.dropna(subset=essential_cols, inplace=True)
    
    return data

def compute_indicators(df):
    if df.empty:
        return df
    
    # Ensure we have the required columns
    required_cols = ['Close', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Required column '{col}' not found in data")
            return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure all values are numeric and 1-dimensional
    for col in required_cols:
        # Handle any potential nested structures
        if hasattr(df[col].iloc[0], '__len__') and not isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in essential columns
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:
        return df
    
    # Reset index to ensure continuous indexing
    df.reset_index(drop=True, inplace=True)

    # Extract series for calculations
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)

    # Calculate indicators with error handling
    try:
        if len(close) >= 9:
            df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        else:
            df['EMA_9'] = np.nan
            
        if len(close) >= 20:
            df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        else:
            df['EMA_20'] = np.nan
            
        if len(close) >= 14:
            df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        else:
            df['RSI'] = np.nan

        # Calculate VWAP
        typical_price = (high + low + close) / 3
        df['VWAP'] = (volume * typical_price).cumsum() / volume.cumsum()
        
        # Calculate average volume
        if len(volume) >= 20:
            df['avg_vol'] = volume.rolling(window=20).mean()
        else:
            df['avg_vol'] = volume.mean()
            
    except Exception as e:
        st.error(f"Error computing indicators: {e}")
        return pd.DataFrame()
    
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
            st.warning(f"Could not fetch options for expiry {expiry}: {e}")
            continue
    
    return all_calls, all_puts

def classify_moneyness(row, spot):
    strike = row['strike']
    if strike < spot * 0.99:  # Allow small tolerance for ATM
        return 'ITM'
    elif strike <= spot * 1.01:
        return 'ATM'
    else:
        return 'OTM'

# =============================
# SIGNAL LOGIC
# =============================

def generate_signal(option, side, stock_df):
    if stock_df.empty:
        return False
    
    latest = stock_df.iloc[-1]

    try:
        # Extract option Greeks with safe conversion
        delta = option.get('delta', np.nan)
        gamma = option.get('gamma', np.nan)
        theta = option.get('theta', np.nan)
        
        # Convert to float safely
        delta = float(delta) if not pd.isna(delta) else np.nan
        gamma = float(gamma) if not pd.isna(gamma) else np.nan
        theta = float(theta) if not pd.isna(theta) else np.nan
        
        # Extract stock data
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else np.nan
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else np.nan
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else np.nan
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else np.nan
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume

        # Check if we have enough data for analysis
        if pd.isna(delta) or pd.isna(gamma) or pd.isna(theta):
            return False

        if side == "call":
            if (
                delta >= 0.6 and gamma >= 0.08 and theta <= 0.05 and
                not pd.isna(ema_9) and not pd.isna(ema_20) and
                close > ema_9 > ema_20 and 
                not pd.isna(rsi) and rsi > 50 and 
                close > vwap and
                volume > 1.5 * avg_vol
            ):
                return True
        elif side == "put":
            if (
                delta <= -0.6 and gamma >= 0.08 and theta <= 0.05 and
                not pd.isna(ema_9) and not pd.isna(ema_20) and
                close < ema_9 < ema_20 and 
                not pd.isna(rsi) and rsi < 50 and 
                close < vwap and
                volume > 1.5 * avg_vol
            ):
                return True
    except Exception as e:
        st.error(f"Error in signal generation: {e}")
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
        with st.spinner("Fetching stock and options data..."):
            df = get_stock_data(ticker)

            if df.empty:
                st.error("No valid data retrieved for this ticker.")
                st.stop()

            df = compute_indicators(df)
            
            if df.empty:
                st.error("Could not compute technical indicators. Insufficient data.")
                st.stop()

            # Display current stock info
            current_price = df.iloc[-1]['Close']
            st.info(f"Current price of {ticker}: ${current_price:.2f}")

            stock = yf.Ticker(ticker)
            all_expiries = stock.options
            
            if not all_expiries:
                st.error("No options expiries available for this ticker.")
                st.stop()

            expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"])

            today = datetime.date.today()
            if expiry_mode == "0DTE Only":
                expiries_to_use = [e for e in all_expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
            else:
                expiries_to_use = all_expiries[:3]

            if not expiries_to_use:
                st.warning("No options expiries available for the selected mode.")
                st.stop()

            calls, puts = fetch_all_expiries_data(ticker, expiries_to_use)
            
            if calls.empty and puts.empty:
                st.error("No options data available for the selected expiries.")
                st.stop()

            strike_range = st.slider("Select Strike Range Around Spot Price:", -10, 10, (-5, 5))
            spot = current_price
            min_strike = spot + strike_range[0]
            max_strike = spot + strike_range[1]

            calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
            puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()

            if not calls_filtered.empty:
                calls_filtered['moneyness'] = calls_filtered.apply(lambda row: classify_moneyness(row, spot), axis=1)
            if not puts_filtered.empty:
                puts_filtered['moneyness'] = puts_filtered.apply(lambda row: classify_moneyness(row, spot), axis=1)

            m_filter = st.multiselect("Filter by Moneyness:", options=["ITM", "ATM", "OTM"], default=["ITM", "ATM", "OTM"])

            if not calls_filtered.empty:
                calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
            if not puts_filtered.empty:
                puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]

            st.subheader("📊 Call Option Signals")
            if not calls_filtered.empty:
                call_signals = [row for _, row in calls_filtered.iterrows() if generate_signal(row, "call", df)]
                if call_signals:
                    signals_df = pd.DataFrame(call_signals)
                    # Display key columns
                    display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'expiry', 'moneyness']
                    available_cols = [col for col in display_cols if col in signals_df.columns]
                    st.dataframe(signals_df[available_cols].reset_index(drop=True))
                else:
                    st.info("No CALL signals matched the criteria.")
            else:
                st.info("No call options available for the selected filters.")

            st.subheader("📉 Put Option Signals")
            if not puts_filtered.empty:
                put_signals = [row for _, row in puts_filtered.iterrows() if generate_signal(row, "put", df)]
                if put_signals:
                    signals_df = pd.DataFrame(put_signals)
                    # Display key columns
                    display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'expiry', 'moneyness']
                    available_cols = [col for col in display_cols if col in signals_df.columns]
                    st.dataframe(signals_df[available_cols].reset_index(drop=True))
                else:
                    st.info("No PUT signals matched the criteria.")
            else:
                st.info("No put options available for the selected filters.")

    except Exception as e:
        st.error(f"Failed to fetch or analyze data: {e}")
        st.write("Please try again or check if the ticker symbol is correct.")
