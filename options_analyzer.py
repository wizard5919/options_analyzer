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
    """
    Fetches historical stock data for a given ticker.
    Changed to fetch daily data over 1 year for more robust indicator calculations.
    """
    # Fetch 1 year of daily interval data for more robust indicator calculations
    # Using period instead of start/end for simplicity with yfinance
    data = yf.download(ticker, period="1y", interval="1d")
    
    # IMPORTANT: Check if data is None or empty immediately after download
    if data is None or data.empty or data.index.empty:
        st.error(f"No valid stock data downloaded for {ticker}. It might be an invalid ticker, or no data is available for the specified period/interval.")
        return pd.DataFrame()

    # Ensure data is a DataFrame and has required columns
    # yfinance typically returns a DataFrame, but adding checks for robustness
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series): # If yf.download returns a Series, convert to DataFrame
            data = data.to_frame('Close')
        else: # Handle unexpected data types, e.g., if data is not DataFrame or Series
            st.error("Downloaded stock data is not in a recognized DataFrame or Series format.")
            return pd.DataFrame() 
    
    # Ensure all required columns are present; fill with NaN if missing
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            # For missing columns, fill with previous valid observation or 0 if no previous
            data[col] = np.nan
    
    # Fill any NaNs that might appear due to missing columns or data points
    # Use ffill then bfill to handle NaNs at the beginning and end of the series
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    # Drop any rows that still have NaN values after filling, if any (should be rare now)
    data.dropna(inplace=True)
    return data

def compute_indicators(df):
    """
    Computes technical indicators (EMA, RSI, VWAP, Avg Volume) for the stock data.
    Includes checks for sufficient data before computation.
    """
    # Define the minimum number of rows needed for indicator calculations (based on max window size)
    # Changed to 20 for daily data, as 5-min data would have many more points
    min_rows_needed = 20 
    
    # Convert relevant columns to float explicitly to ensure correct type for calculations
    for col in ['Close', 'High', 'Low', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # Drop any remaining NaN values after type conversion
    df.dropna(inplace=True)

    # Check length after dropping NaNs, as more rows might have been removed
    if len(df) < min_rows_needed:
        st.warning(f"Not enough data points ({len(df)}) after cleaning for indicator calculation. Need at least {min_rows_needed} rows.")
        return pd.DataFrame() # Return empty DataFrame if not enough data
    
    # Create copies of Series and explicitly cast to pd.Series to ensure correct type for ta library
    close_series = pd.Series(df['Close'].squeeze().copy())
    high_series = pd.Series(df['High'].squeeze().copy())
    low_series = pd.Series(df['Low'].squeeze().copy())
    volume_series = pd.Series(df['Volume'].squeeze().copy())
    
    # IMPORTANT: Check if series is empty or too short AFTER extraction and copying
    # This prevents errors from ta.momentum/trend if the Series is not valid
    if close_series.empty or len(close_series) < min_rows_needed:
        st.warning(f"Close series is empty or too short ({len(close_series)}) for indicator calculation after extraction. Need at least {min_rows_needed} data points.")
        return pd.DataFrame()

    # EMA calculations
    df['EMA_9'] = EMAIndicator(close=close_series, window=9).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=close_series, window=20).ema_indicator()
    
    # RSI calculation
    df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
    
    # Calculate VWAP (Volume Weighted Average Price)
    typical_price = (high_series + low_series + close_series) / 3
    cumulative_volume = volume_series.cumsum()
    # Handle potential division by zero if cumulative_volume is zero
    df['VWAP'] = (typical_price * volume_series).cumsum() / cumulative_volume
    # Replace infinite values (from division by zero) with NaN
    df['VWAP'].replace([np.inf, -np.inf], np.nan, inplace=True) 

    # Calculate average volume over a 20-period window
    df['avg_vol'] = volume_series.rolling(window=20).mean()
    
    # Drop rows with NaN values that might result from indicator computation (e.g., initial periods)
    df.dropna(inplace=True)
    return df

def fetch_all_expiries_data(ticker, expiries):
    """
    Fetches call and put options data for a given ticker across specified expiries.
    """
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
    """
    Classifies an option as In-The-Money (ITM), At-The-Money (ATM), or Out-The-Money (OTM).
    """
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
    """
    Generates a buy signal for an option based on Greeks and technical indicators.
    """
    if stock_df.empty:
        st.warning("Stock data is empty for signal generation. Cannot generate signal.")
        return False

    latest = stock_df.iloc[-1]
    
    # Safely convert option Greeks and latest stock data to float
    try:
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9'])
        ema_20 = float(latest['EMA_20'])
        rsi = float(latest['RSI'])
        vwap = float(latest['VWAP'])
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol'])
    except (ValueError, KeyError) as e:
        st.warning(f"Error converting values for signal generation: {e}. Skipping option.")
        return False

    try:
        if side == "call":
            if (
                delta >= 0.6
                and gamma >= 0.08
                and theta <= 0.05
                and close > ema_9 > ema_20 # Bullish EMA crossover
                and rsi > 50 # RSI indicating upward momentum
                and close > vwap # Price above VWAP
                and volume > 1.5 * avg_vol # High volume
            ):
                return True
        elif side == "put":
            if (
                delta <= -0.6 # Negative delta for puts
                and gamma >= 0.08
                and theta <= 0.05
                and close < ema_9 < ema_20 # Bearish EMA crossover
                and rsi < 50 # RSI indicating downward momentum
                and close < vwap # Price below VWAP
                and volume > 1.5 * avg_vol # High volume
            ):
                return True
    except Exception as e:
        st.warning(f"An error occurred during signal evaluation for {side} option: {e}")
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
            st.error("No valid stock data available for the given ticker or insufficient data for analysis. Please try a different ticker or check connectivity.")
            st.stop()

        df = compute_indicators(df)
        
        if df.empty:
            st.error("No valid data after computing indicators. This might be due to insufficient data points after cleaning or calculation.")
            st.stop()

        stock = yf.Ticker(ticker)
        all_expiries = stock.options
        
        if not all_expiries:
            st.warning("No options expiries available for this ticker.")
            st.stop()

        expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"])

        today = datetime.date.today()
        if expiry_mode == "0DTE Only":
            # Filter for expiries that match today's date
            expiries_to_use = [e for e in all_expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
        else:
            # Use the first 3 available expiries for "Near-Term"
            expiries_to_use = all_expiries[:3] 

        if not expiries_to_use:
            st.warning("No options expiries available for the selected mode. Please adjust your filter.")
            st.stop()

        calls, puts = fetch_all_expiries_data(ticker, expiries_to_use)

        if calls.empty and puts.empty:
            st.warning("No options data available for the selected expiries. This might be due to data fetching issues.")
            st.stop()

        # Get the latest spot price from the fetched stock data
        spot = float(df['Close'].iloc[-1])
        
        # Allow user to select a strike price range around the spot price
        strike_range = st.slider("Select Strike Range Around Spot Price:", -10, 10, (-5, 5))
        min_strike = spot + strike_range[0]
        max_strike = spot + strike_range[1]

        # Filter calls and puts based on the selected strike range
        calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
        puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()

        # Classify moneyness for filtered options
        calls_filtered['moneyness'] = calls_filtered.apply(lambda row: classify_moneyness(row, spot), axis=1)
        puts_filtered['moneyness'] = puts_filtered.apply(lambda row: classify_moneyness(row, spot), axis=1)

        # Allow user to filter by moneyness (ITM, ATM, OTM)
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
            st.info("No CALL signals matched the criteria for the selected filters.")

        st.subheader("📉 Put Option Signals")
        put_signals = []
        for _, row in puts_filtered.iterrows():
            if generate_signal(row, "put", df):
                put_signals.append(row)
        if put_signals:
            st.dataframe(pd.DataFrame(put_signals).reset_index(drop=True))
        else:
            st.info("No PUT signals matched the criteria for the selected filters.")

    except Exception as e:
        st.error(f"An unexpected error occurred during data fetching or analysis: {str(e)}. Please check the ticker symbol, your internet connection, or try again later.")
        st.stop()
