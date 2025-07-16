import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import pytz
import threading
from typing import Optional, Tuple, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(
    page_title="Options Greeks Buy Signal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CONFIGURATION & CONSTANTS
# =============================
CONFIG = {
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 1,
    'DATA_TIMEOUT': 30,
    'MIN_DATA_POINTS': 50,
    'CACHE_TTL': 300,  # 5 minutes for heavy analysis
    'RATE_LIMIT_COOLDOWN': 180,  # 3 minutes
    'MARKET_OPEN': datetime.time(9, 30),
    'MARKET_CLOSE': datetime.time(16, 0),
    'PREMARKET_START': datetime.time(4, 0),
    'VOLATILITY_THRESHOLDS': {'low': 0.015, 'medium': 0.03, 'high': 0.05},
    'PROFIT_TARGETS': {'call': 0.15, 'put': 0.15, 'stop_loss': 0.08}
}

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_base': 0.5, 'delta_vol_multiplier': 0.1, 'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.02, 'theta_base': -0.05, 'rsi_base': 50,
        'rsi_min': 50, 'rsi_max': 70, 'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.3, 'volume_min': 1000
    },
    'put': {
        'delta_base': -0.5, 'delta_vol_multiplier': 0.1, 'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.02, 'theta_base': -0.05, 'rsi_base': 50,
        'rsi_min': 30, 'rsi_max': 50, 'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.3, 'volume_min': 1000
    }
}


# =============================
# LIGHTWEIGHT REAL-TIME PRICE FETCHER
# =============================
def get_current_price_info(ticker: str) -> Optional[Dict]:
    """
    A lightweight function to get only the most recent price data.
    This is called every few seconds.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            # Use the previous close from the 1m chart for delta
            previous_close = data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[-1]
            delta = latest_price - previous_close
            return {"price": latest_price, "delta": delta}
    except Exception:
        return None # Fail silently to not interrupt the UI
    return None

# =============================
# HEAVY ANALYSIS FUNCTIONS (Wrapped and Cached)
# =============================
def is_market_open() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    return now.weekday() < 5 and CONFIG['MARKET_OPEN'] <= now.time() <= CONFIG['MARKET_CLOSE']

def is_premarket() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    return now.weekday() < 5 and CONFIG['PREMARKET_START'] <= now.time() < CONFIG['MARKET_OPEN']

def is_early_market() -> bool:
    if not is_market_open(): return False
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open_today = eastern.localize(datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN']))
    return (now - market_open_today).total_seconds() < 1800

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                st.warning("Yahoo Finance rate limit reached. Please wait a few minutes before retrying.")
                st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
                return None
            if attempt == max_retries - 1:
                st.error(f"API call failed after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(CONFIG['RETRY_DELAY'] * (attempt + 1))
    return None

def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=10)
        data = yf.download(ticker, start=start, end=end, interval="5m", auto_adjust=True, progress=False, prepost=True)
        if data.empty:
            st.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        required_cols = ['Close', 'High', 'Low', 'Volume']
        if any(col not in data.columns for col in required_cols):
            st.error(f"Missing required columns in data for {ticker}")
            return pd.DataFrame()

        data = data.dropna(how='all')
        eastern = pytz.timezone('US/Eastern')
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc)
        data.index = data.index.tz_convert(eastern)
        
        data['premarket'] = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        return data.reset_index(drop=False)
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Check for valid input
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    df = df.copy()
    required_cols = ['Close', 'High', 'Low', 'Volume']
    
    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in data: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    # Convert columns to numeric
    for col in required_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except TypeError:
            # Handle case where we might have a DataFrame instead of Series
            df[col] = pd.to_numeric(df[col].iloc[:, 0], errors='coerce')
    
    # Only drop NA values on columns that actually exist
    existing_cols = [col for col in required_cols if col in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols, how='all')
    
    if df.empty: 
        return df

    close, high, low = df['Close'], df['High'], df['Low']
    if len(close) >= 9: 
        df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
    if len(close) >= 20: 
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
    if len(close) >= 14: 
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
    if len(close) >= 14: 
        df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    
    # VWAP and Volume calculations
    df['VWAP'] = np.nan
    df['avg_vol'] = np.nan
    
    # Group by day only if 'Datetime' column exists
    if 'Datetime' in df.columns:
        for session, group in df.groupby(pd.Grouper(key='Datetime', freq='D')):
            if group.empty: 
                continue
            typical_price = (group['High'] + group['Low'] + group['Close']) / 3
            vwap_cumsum = (group['Volume'] * typical_price).cumsum()
            volume_cumsum = group['Volume'].cumsum()
            df.loc[group.index, 'VWAP'] = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
            df.loc[group.index, 'avg_vol'] = group['Volume'].expanding(min_periods=1).mean()
    else:
        # Fallback calculation without datetime grouping
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap_cumsum = (df['Volume'] * typical_price).cumsum()
        volume_cumsum = df['Volume'].cumsum()
        df['VWAP'] = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
        df['avg_vol'] = df['Volume'].expanding(min_periods=1).mean()
        
    df['ATR_pct'] = df['ATR'] / df['Close']
    df['avg_vol'] = df['avg_vol'].fillna(df['Volume'].mean())
    return df

def get_options_expiries(ticker: str) -> List[str]:
    try:
        stock = yf.Ticker(ticker)
        return stock.options or []
    except Exception as e:
        st.error(f"Error fetching expiries: {str(e)}")
        return []

def fetch_options_data(ticker: str, expiries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_calls, all_puts = pd.DataFrame(), pd.DataFrame()
    stock = yf.Ticker(ticker)
    for expiry in expiries:
        chain = safe_api_call(stock.option_chain, expiry)
        if chain:
            calls, puts = chain.calls.copy(), chain.puts.copy()
            calls['expiry'], puts['expiry'] = expiry, expiry
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
            time.sleep(0.5) # Small delay
    return all_calls, all_puts

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict:
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    volatility = stock_data.get('ATR_pct', 0.02)
    vol_multiplier = 1 + (volatility * 100)

    if side == 'call':
        thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_multiplier))
    else:
        thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_multiplier))
    
    if is_premarket() or is_early_market():
        thresholds['volume_min'] *= 0.6
    if is_0dte:
        thresholds['volume_min'] *= 0.7
    return thresholds

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    if stock_df.empty or not isinstance(option, pd.Series):
        return {'signal': False, 'reason': 'Invalid data'}
    
    latest = stock_df.iloc[-1]
    required_fields = ['strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta']
    if any(field not in option or pd.isna(option[field]) for field in required_fields):
        return {'signal': False, 'reason': 'Missing option data'}

    thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
    delta, gamma, theta, option_vol = option['delta'], option['gamma'], option['theta'], option['volume']
    close, ema9, ema20, rsi, vwap = latest.get('Close'), latest.get('EMA_9'), latest.get('EMA_20'), latest.get('RSI'), latest.get('VWAP')
    
    conditions = []
    if side == "call":
        conditions = [
            (delta >= thresholds['delta_min'], f"Delta >= {thresholds['delta_min']:.2f}"),
            (gamma >= thresholds['gamma_base'], f"Gamma >= {thresholds['gamma_base']:.3f}"),
            (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}"),
            (ema9 is not None and ema20 is not None and close > ema9 > ema20, "Price > EMA9 > EMA20"),
            (rsi is not None and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']}"),
            (vwap is not None and close > vwap, "Price > VWAP"),
            (option_vol > thresholds['volume_min'], f"Opt Vol > {thresholds['volume_min']:.0f}")
        ]
    else: # put
        conditions = [
            (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']:.2f}"),
            (gamma >= thresholds['gamma_base'], f"Gamma >= {thresholds['gamma_base']:.3f}"),
            (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}"),
            (ema9 is not None and ema20 is not None and close < ema9 < ema20, "Price < EMA9 < EMA20"),
            (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']}"),
            (vwap is not None and close < vwap, "Price < VWAP"),
            (option_vol > thresholds['volume_min'], f"Opt Vol > {thresholds['volume_min']:.0f}")
        ]

    passed = [desc for passed, desc in conditions if passed]
    signal = all(p for p, d in conditions)
    return {'signal': signal, 'passed_conditions': passed, 'score': len(passed) / len(conditions)}


@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def perform_full_analysis(ticker: str, selected_expiries: List[str]):
    """
    This function runs the entire heavy analysis pipeline.
    It's cached to avoid re-running on every small interaction.
    """
    stock_df = get_stock_data(ticker)
    if stock_df.empty:
        st.error("Could not retrieve stock data. Analysis halted.")
        return
        
    stock_df = compute_indicators(stock_df)
    if stock_df.empty:
        st.error("Failed to compute technical indicators. Analysis halted.")
        return

    calls_df, puts_df = fetch_options_data(ticker, selected_expiries)
    if calls_df.empty and puts_df.empty:
        st.warning("No options data could be fetched for the selected expiries.")
        return
    
    # --- Process and Display Results ---
    current_price = stock_df.iloc[-1]['Close']
    st.subheader(f"Analysis for ${ticker} at ${current_price:.2f}")
    
    # Find best signals
    call_signals, put_signals = [], []
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    if not calls_df.empty:
        calls_df['signal_data'] = calls_df.apply(
            lambda row: generate_signal(row, 'call', stock_df, row['expiry'] == today_str), axis=1
        )
        call_signals = calls_df[calls_df['signal_data'].apply(lambda x: x['signal'])].copy()

    if not puts_df.empty:
        puts_df['signal_data'] = puts_df.apply(
            lambda row: generate_signal(row, 'put', stock_df, row['expiry'] == today_str), axis=1
        )
        put_signals = puts_df[puts_df['signal_data'].apply(lambda x: x['signal'])].copy()

    # --- Display Signal Tables ---
    for signals, side in [(call_signals, "Call"), (put_signals, "Put")]:
        if not signals.empty:
            st.success(f"✅ Found {len(signals)} Potential **{side}** Buy Signals!")
            signals['Passed Conditions'] = signals['signal_data'].apply(lambda x: ", ".join(x['passed_conditions']))
            display_df = signals[['contractSymbol', 'lastPrice', 'strike', 'volume', 'expiry', 'Passed Conditions']].rename(columns={'contractSymbol': 'Symbol', 'lastPrice': 'Price', 'strike': 'Strike', 'volume': 'Volume', 'expiry': 'Expiry'})
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info(f"ℹ️ No strong **{side}** buy signals found based on current criteria.")

# =============================
# STREAMLIT INTERFACE
# =============================
st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Real-time price updates with on-demand deep analysis.**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Enter Stock Ticker", "NVDA").upper()

    st.subheader("Base Signal Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Calls**")
        SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider("Base Call Delta", 0.1, 1.0, 0.5, 0.05, key='c_delta')
        SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider("Base Call Gamma", 0.01, 0.2, 0.05, 0.01, key='c_gamma')
        SIGNAL_THRESHOLDS['call']['rsi_min'] = st.slider("Min Call RSI", 30, 70, 50, 1, key='c_rsi')
        SIGNAL_THRESHOLDS['call']['volume_min'] = st.number_input("Min Call Volume", 100, 10000, 1000, 100, key='c_vol')
    with col2:
        st.write("**Puts**")
        SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider("Base Put Delta", -1.0, -0.1, -0.5, 0.05, key='p_delta')
        SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider("Base Put Gamma", 0.01, 0.2, 0.05, 0.01, key='p_gamma')
        SIGNAL_THRESHOLDS['put']['rsi_max'] = st.slider("Max Put RSI", 30, 70, 50, 1, key='p_rsi')
        SIGNAL_THRESHOLDS['put']['volume_min'] = st.number_input("Min Put Volume", 100, 10000, 1000, 100, key='p_vol')

    expiries = get_options_expiries(ticker)
    if expiries:
        selected_expiries = st.multiselect(
            "Select Option Expiries for Analysis",
            options=expiries,
            default=expiries[:2] if len(expiries) > 1 else expiries
        )
    else:
        st.warning("Could not fetch option expiries for this ticker.")
        selected_expiries = []

# --- Main Display Area ---
col_price, col_main = st.columns([1, 3])

with col_price:
    # This placeholder will be updated by the loop at the end of the script
    price_placeholder = st.empty()
    # Initially populate it so it doesn't look empty
    price_placeholder.info("Fetching price...")

with col_main:
    # Button to trigger the heavy analysis
    if st.button("🚀 Run Full Analysis", use_container_width=True):
        if not ticker:
            st.error("Please enter a stock ticker in the sidebar.")
        elif not selected_expiries:
            st.error("Please select at least one expiry date for analysis.")
        else:
            with st.spinner(f"Performing deep analysis for ${ticker}... This may take a moment."):
                perform_full_analysis(ticker, selected_expiries)
    else:
        st.info("Click the 'Run Full Analysis' button to begin.")

# --- The "Real-Time" Update Loop ---
# This part of the script loops continuously to update the price.
# It MUST be at the end of the script.
while True:
    price_info = get_current_price_info(ticker)
    
    if price_info:
        with price_placeholder.container():
            st.metric(
                label=f"Current Price of ${ticker}",
                value=f"${price_info['price']:.2f}",
                delta=f"{price_info['delta']:.2f} (1m)"
            )
    
    # Refresh every 2 seconds - a safe interval to avoid rate limits
    time.sleep(2)
