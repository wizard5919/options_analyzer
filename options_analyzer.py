import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

st.set_page_config(page_title="Options Greek Signal Analyzer", layout="wide")
st.title("📈 Options Greeks Buy Signal Analyzer (Enhanced with Technicals)")

refresh_clicked = st.button("🔄 Refresh App")
ticker = st.text_input("Enter Ticker Symbol (e.g., IWM):", value="IWM", key="ticker" if not refresh_clicked else "ticker_refreshed").upper()

# =============================
# UTILITY FUNCTIONS
# =============================

def get_stock_data(ticker):
    """
    Fetches historical stock data for a given ticker with 5-minute interval.
    Fetches data for the last 7 days to ensure sufficient intraday points.
    """
    end = datetime.now()
    start = end - timedelta(days=7) # Fetch 7 days of 5-minute data for robust intraday analysis
    
    # Use yf.download with start and end dates for more control
    data = yf.download(ticker, start=start, end=end, interval="5m")
    
    if data is None or data.empty or data.index.empty:
        st.error(f"No valid 5-minute intraday data downloaded for {ticker}. It might be an invalid ticker, or no data is available for the specified period/interval.")
        return pd.DataFrame()

    # Ensure data is a DataFrame and has required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            data[col] = np.nan # Add missing columns as NaN
    
    # Convert all required columns to float, coercing errors to NaN
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop rows where essential columns are entirely NaN
    data.dropna(subset=required_cols, inplace=True)

    # Fill any remaining NaNs that might appear due to sporadic missing points
    # Use ffill then bfill to handle NaNs at the beginning and end of the series
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    # Final check for empty DataFrame after all cleaning
    if data.empty:
        st.error(f"Stock data for {ticker} became empty after cleaning. Cannot proceed with indicator calculations.")
    return data

def compute_indicators(df):
    """
    Computes technical indicators (EMA, RSI, VWAP, Avg Volume) for the stock data.
    Uses `ta` library for EMA and RSI, designed for robust calculations.
    """
    # Define the minimum number of rows needed for indicator calculations (based on max window size)
    min_rows_needed = 20 # For EMA20 and Avg Vol 20
    
    # Initialize indicator columns to NaN to ensure they always exist
    df['EMA_9'] = np.nan
    df['EMA_20'] = np.nan
    df['RSI'] = np.nan
    df['VWAP'] = np.nan
    df['avg_vol'] = np.nan

    if len(df) < min_rows_needed:
        st.warning(f"Not enough data points ({len(df)}) for full indicator calculation. Need at least {min_rows_needed} rows. Indicators will be NaN for initial periods or entirely if data is too short.")
        # Return df with NaN indicators if data is too short
        return df 

    # Create explicit Pandas Series for TA library inputs to avoid dimensionality issues
    close_series = pd.Series(df['Close'].copy())
    high_series = pd.Series(df['High'].copy())
    low_series = pd.Series(df['Low'].copy())
    volume_series = pd.Series(df['Volume'].copy())
    
    # Check if series are valid and long enough before passing to TA functions
    if not close_series.empty and len(close_series) >= 9: # Min window for EMA_9
        df['EMA_9'] = EMAIndicator(close=close_series, window=9, fillna=False).ema_indicator()
    if not close_series.empty and len(close_series) >= 20: # Min window for EMA_20
        df['EMA_20'] = EMAIndicator(close=close_series, window=20, fillna=False).ema_indicator()
    
    if not close_series.empty and len(close_series) >= 14: # Min window for RSI
        df['RSI'] = RSIIndicator(close=close_series, window=14, fillna=False).rsi()
    
    # Calculate VWAP (Volume Weighted Average Price)
    # Ensure typical_price calculation is robust and volume_series is not empty
    if not volume_series.empty and not typical_price.empty:
        typical_price = (high_series + low_series + close_series) / 3
        cumulative_volume = volume_series.cumsum()
        
        # Handle potential division by zero in VWAP
        with np.errstate(divide='ignore', invalid='ignore'):
            df['VWAP'] = (typical_price * volume_series).cumsum() / cumulative_volume
            df['VWAP'].replace([np.inf, -np.inf], np.nan, inplace=True) 

    # Calculate average volume over a 20-period window
    if not volume_series.empty and len(volume_series) >= 20:
        df['avg_vol'] = volume_series.rolling(window=20, min_periods=1).mean() # min_periods for initial values
    
    # Drop rows with NaN values that might result from indicator computation (e.g., initial periods)
    # Only drop rows where ALL indicators are NaN, or where critical indicators are NaN
    df.dropna(subset=['EMA_9', 'EMA_20', 'RSI', 'VWAP', 'avg_vol'], inplace=True)

    if df.empty:
        st.error("DataFrame became empty after computing indicators and dropping NaNs. This can happen if there's insufficient data.")
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
        return False

    # Ensure stock_df has enough rows for latest to be valid
    if len(stock_df) == 0:
        return False

    latest = stock_df.iloc[-1]
    
    # Safely convert option Greeks and latest stock data to float
    try:
        delta = float(option.get('delta', np.nan))
        gamma = float(option.get('gamma', np.nan))
        theta = float(option.get('theta', np.nan))
        
        # Ensure latest values exist before attempting conversion
        close = float(latest.get('Close', np.nan))
        ema_9 = float(latest.get('EMA_9', np.nan))
        ema_20 = float(latest.get('EMA_20', np.nan))
        rsi = float(latest.get('RSI', np.nan))
        vwap = float(latest.get('VWAP', np.nan))
        volume = float(latest.get('Volume', np.nan))
        avg_vol = float(latest.get('avg_vol', np.nan))

        # If any essential Greek or technical indicator is NaN, cannot generate signal
        if any(pd.isna([delta, gamma, theta, close, ema_9, ema_20, rsi, vwap, volume, avg_vol])):
            return False

    except (ValueError, KeyError, TypeError) as e:
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
        return False

    return False

# =============================
# STREAMLIT INTERFACE LOGIC
# =============================

if ticker:
    try:
        st.write("Fetching stock and options data...")
        df = get_stock_data(ticker)
        
        if df.empty:
            st.stop() # Stop if no valid stock data

        df = compute_indicators(df)
        
        if df.empty:
            st.error("No valid data after computing indicators. This might be due to insufficient data points after cleaning or calculation. Try a different ticker or refresh.")
            st.stop()

        stock = yf.Ticker(ticker)
        all_expiries = stock.options
        
        if not all_expiries:
            st.warning("No options expiries available for this ticker.")
            st.stop()

        expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"])

        today = datetime.now().date()
        if expiry_mode == "0DTE Only":
            expiries_to_use = [e for e in all_expiries if datetime.strptime(e, "%Y-%m-%d").date() == today]
        else:
            expiries_to_use = all_expiries[:3] # Use first 3 expiries

        if not expiries_to_use:
            st.warning("No options expiries available for the selected mode. Please adjust your filter.")
            st.stop()

        calls, puts = fetch_all_expiries_data(ticker, expiries_to_use)

        if calls.empty and puts.empty:
            st.warning("No options data available for the selected expiries. This might be due to data fetching issues.")
            st.stop()

        # Get the latest spot price from the fetched stock data
        # Ensure df is not empty before accessing iloc[-1]
        if not df.empty:
            spot = float(df['Close'].iloc[-1])
        else:
            st.error("Could not determine spot price as stock data is empty.")
            st.stop()
        
        strike_range = st.slider("Select Strike Range Around Spot Price:", -10, 10, (-5, 5))
        min_strike = spot + strike_range[0]
        max_strike = spot + strike_range[1]

        calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
        puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()

        # Classify moneyness for filtered options
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

        # Price Chart
        st.subheader("📊 Price Chart with EMA, RSI, VWAP")
        # Ensure df is not empty before plotting
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], mode='lines', name='EMA 9', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='green')))
            
            fig.update_layout(
                title=f"{ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # RSI Chart
            st.subheader("📈 RSI Chart")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(
                title=f"{ticker} RSI",
                xaxis_title="Date",
                yaxis_title="RSI Value",
                hovermode="x unified",
                height=300
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

            # Add current date and time and latest indicator values
            latest_close = df['Close'].iloc[-1]
            latest_ema9 = df['EMA_9'].iloc[-1]
            latest_ema20 = df['EMA_20'].iloc[-1]
            latest_rsi = df['RSI'].iloc[-1]
            latest_vwap = df['VWAP'].iloc[-1]
            latest_volume = df['Volume'].iloc[-1]
            latest_avg_vol = df['avg_vol'].iloc[-1]

            current_time = datetime.now().strftime("%I:%M %p +01 on %B %d, %Y")
            st.write(f"**Latest Close:** {latest_close:.2f} | **Latest EMA9:** {latest_ema9:.2f} | **Latest EMA20:** {latest_ema20:.2f} | **Latest RSI:** {latest_rsi:.2f} | **Latest VWAP:** {latest_vwap:.2f} | **Latest Volume:** {latest_volume:.0f} | **Latest Avg Vol (20 periods):** {latest_avg_vol:.0f} | **Analysis Date:** {current_time}")
        else:
            st.warning("No data available to plot charts.")

    except Exception as e:
        st.error(f"An unexpected error occurred during data fetching or analysis: {str(e)}. Please check the ticker symbol, your internet connection, or try again later.")
        st.stop()
