import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
from typing import Optional, Tuple, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(
    page_title="Options Signal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- Configuration -----------------------------
CONFIG = {
    'AUTO_REFRESH': True,
    'REFRESH_INTERVAL_SEC': 60,
    'MAX_EXPIRIES_PER_BATCH': 2,
    'CACHE_TTL': 300
}

# -------------------------- Session State Init --------------------------
if 'rate_limited_until' not in st.session_state:
    st.session_state.rate_limited_until = None

if 'expiry_batch_index' not in st.session_state:
    st.session_state.expiry_batch_index = 0

# ---------------------------- Sidebar UI ----------------------------
st.sidebar.header("⚙️ Configuration")
auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Refresh", value=CONFIG['AUTO_REFRESH'])
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper()
max_expiry_batch = st.sidebar.slider("Expiries per batch", 1, 5, CONFIG['MAX_EXPIRIES_PER_BATCH'])
st.sidebar.button("🔄 Refresh Now", on_click=lambda: st.session_state.__setitem__('expiry_batch_index', 0))

# ---------------------------- Rate Limit Check ----------------------------
if st.session_state.rate_limited_until:
    if datetime.datetime.utcnow() < st.session_state.rate_limited_until:
        st.warning("⏳ Currently rate-limited. Please wait and try again later.")
        st.stop()
    else:
        st.session_state.rate_limited_until = None

# ---------------------------- Cached Ticker ----------------------------
@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_cached_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)

# ---------------------------- Safe API Call ----------------------------
def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if 'Too Many Requests' in str(e):
            st.session_state.rate_limited_until = datetime.datetime.utcnow() + datetime.timedelta(minutes=1)
            st.error("🚫 Error fetching data: Too Many Requests. Rate limited. Try again in a few minutes.")
        else:
            st.error(f"⚠️ Unexpected error: {e}")
        return None

# ---------------------------- Fetch Expiries ----------------------------
def get_options_expiries(ticker: str) -> List[str]:
    ticker_obj = get_cached_ticker(ticker)
    expiries = safe_api_call(lambda: ticker_obj.options)
    if not expiries:
        st.error("⚠️ No options expiries available for this ticker.")
        return []
    return expiries

# ---------------------------- Fetch Options Chain ----------------------------
def get_options_chain(ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ticker_obj = get_cached_ticker(ticker)
    opt = safe_api_call(lambda: ticker_obj.option_chain(expiry))
    if opt:
        return opt.calls, opt.puts
    return pd.DataFrame(), pd.DataFrame()

# ---------------------------- Technical Indicators ----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['EMA_5'] = EMAIndicator(close=df['Close'], window=5).ema_indicator()
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    return df

# ---------------------------- Main Analysis ----------------------------
if not ticker_input:
    st.warning("Please enter a valid stock ticker.")
    st.stop()

# Get price and RSI
ticker_obj = get_cached_ticker(ticker_input)
hist = safe_api_call(lambda: ticker_obj.history(period="1mo"))
if hist is None or hist.empty:
    st.error("No historical data found.")
    st.stop()

hist = compute_indicators(hist)
st.subheader(f"📊 {ticker_input} - Current Price: ${hist['Close'].iloc[-1]:.2f}")

# Get all expiries and process in batches
expiries = get_options_expiries(ticker_input)
if not expiries:
    st.stop()

batch_index = st.session_state.expiry_batch_index
start_idx = batch_index * max_expiry_batch
end_idx = start_idx + max_expiry_batch
expiries_to_use = expiries[start_idx:end_idx]

st.info(f"🔢 Processing expiries {start_idx + 1} to {min(end_idx, len(expiries))} of {len(expiries)}")

all_calls = []
all_puts = []

for expiry in expiries_to_use:
    st.markdown(f"📅 Expiry: `{expiry}`")
    calls, puts = get_options_chain(ticker_input, expiry)
    if not calls.empty:
        all_calls.append(calls.assign(expiry=expiry))
    if not puts.empty:
        all_puts.append(puts.assign(expiry=expiry))
    time.sleep(1)

# Update batch index for next refresh
if end_idx < len(expiries):
    st.session_state.expiry_batch_index += 1
else:
    st.session_state.expiry_batch_index = 0

# Display results
if all_calls:
    combined_calls = pd.concat(all_calls)
    st.subheader("📈 Call Options")
    st.dataframe(combined_calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility', 'expiry']])
else:
    st.warning("No call options data available.")

if all_puts:
    combined_puts = pd.concat(all_puts)
    st.subheader("📉 Put Options")
    st.dataframe(combined_puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility', 'expiry']])
else:
    st.warning("No put options data available.")
