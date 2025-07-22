import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import pytz
import aiohttp
import asyncio
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import plotly.graph_objects as go
import threading

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Apply custom CSS for modern, professional look
st.markdown("""
<style>
/* General Styling */
.stApp { background-color: #f8f9fa; font-family: 'Segoe UI', Arial, sans-serif; }
h1, h2, h3 { color: #1a3c6e; font-weight: 600; }
.stButton>button { background-color: #1a3c6e; color: white; border-radius: 8px; padding: 10px 20px; font-weight: 500; transition: background-color 0.3s; }
.stButton>button:hover { background-color: #2c5a9e; }
.stSelectbox, .stTextInput, .stSlider, .stCheckbox { background-color: #ffffff; border-radius: 8px; padding: 8px; border: 1px solid #d1d5db; }

/* Sidebar Styling */
.stSidebar { background-color: #ffffff !important; border-right: 1px solid #e5e7eb; padding: 20px; }
.stSidebar h2 { font-size: 1.4rem; color: #1a3c6e; }

/* Metric Styling */
.stMetric { background-color: #ffffff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px; }
.call-metric { border-left: 5px solid #28a745; }
.put-metric { border-left: 5px solid #dc3545; }

/* Table Styling */
.signal-table th { background-color: #1a3c6e; color: white; padding: 12px; font-weight: 500; }
.signal-table td { padding: 10px; }

/* Tooltip Styling */
.tooltip { position: relative; display: inline-block; cursor: pointer; }
.tooltip .tooltiptext { visibility: hidden; width: 220px; background-color: #1a3c6e; color: #fff; text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -110px; opacity: 0; transition: opacity 0.3s; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    .stApp { background-color: #1f2a44; }
    h1, h2, h3 { color: #e9ecef; }
    .stMetric { background-color: #2d3b5e; color: #ffffff !important; }
    .stMetric .stMetric-value { color: #ffffff !important; }
    .stSidebar { background-color: #2d3b5e !important; }
    .stSelectbox, .stTextInput, .stSlider, .stCheckbox { background-color: #3b4a6b; border: 1px solid #4b5e8a; }
    .signal-table th { background-color: #2c5a9e; }
}

/* Responsive Design */
@media (max-width: 768px) {
    .stApp { zoom: 0.9; }
    .stSidebar { padding: 15px; }
    .stMetric { padding: 10px; }
}

/* Animation for Alerts */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
.signal-alert { animation: pulse 1.5s infinite; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

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
    'CACHE_TTL': 300,
    'RATE_LIMIT_COOLDOWN': 180,
    'MARKET_OPEN': datetime.time(9, 30),
    'MARKET_CLOSE': datetime.time(16, 0),
    'PREMARKET_START': datetime.time(4, 0),
    'POSTMARKET_END': datetime.time(20, 0),
    'VOLATILITY_THRESHOLDS': {'low': 0.015, 'medium': 0.03, 'high': 0.05},
    'PROFIT_TARGETS': {'call': 0.15, 'put': 0.15, 'stop_loss': 0.08},
    'PRICE_ACTION': {'lookback_periods': 5, 'momentum_threshold': 0.01, 'breakout_threshold': 0.02}
}

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_base': 0.5, 'delta_vol_multiplier': 0.15, 'gamma_base': 0.05, 'gamma_vol_multiplier': 0.03,
        'theta_base': 0.05, 'rsi_base': 50.0, 'rsi_min': 50.0, 'rsi_max': 70.0, 'stoch_base': 60.0,
        'volume_multiplier_base': 1.0, 'volume_vol_multiplier': 0.4, 'volume_min': 1000.0, 'price_momentum_min': 0.005
    },
    'put': {
        'delta_base': -0.5, 'delta_vol_multiplier': 0.15, 'gamma_base': 0.05, 'gamma_vol_multiplier': 0.03,
        'theta_base': 0.05, 'rsi_base': 50.0, 'rsi_min': 30.0, 'rsi_max': 50.0, 'stoch_base': 40.0,
        'volume_multiplier_base': 1.0, 'volume_vol_multiplier': 0.4, 'volume_min': 1000.0, 'price_momentum_min': -0.005
    }
}

# =============================
# AUTO-REFRESH SYSTEM
# =============================
def manage_auto_refresh():
    if st.session_state.get('enable_auto_refresh', False) and 'refresh_interval' in st.session_state:
        refresh_interval = st.session_state['refresh_interval']
        last_refresh = st.session_state.get('last_refresh', time.time())
        if time.time() - last_refresh >= refresh_interval and not st.session_state.get('rerun_pending', False):
            st.session_state.last_refresh = time.time()
            st.session_state.rerun_pending = True
            st.experimental_rerun()
            st.session_state.rerun_pending = False

def get_dynamic_refresh_interval() -> int:
    market_state = get_market_state()
    return 30 if market_state in ["Premarket", "Open"] else 120 if market_state == "Postmarket" else 300

# =============================
# UTILITY FUNCTIONS
# =============================
def get_market_state() -> str:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    now_time = now.time()
    if now.weekday() >= 5:
        return "Closed"
    if CONFIG['PREMARKET_START'] <= now_time < CONFIG['MARKET_OPEN']:
        return "Premarket"
    elif CONFIG['MARKET_OPEN'] <= now_time <= CONFIG['MARKET_CLOSE']:
        return "Open"
    elif CONFIG['MARKET_CLOSE'] < now_time <= CONFIG['POSTMARKET_END']:
        return "Postmarket"
    return "Closed"

def is_market_open() -> bool:
    return get_market_state() == "Open"

def calculate_time_decay_factor() -> float:
    if not is_market_open():
        return 1.0
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'], tzinfo=eastern)
    market_close = datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE'], tzinfo=eastern)
    total_market_seconds = (market_close - market_open).total_seconds()
    elapsed_seconds = (now - market_open).total_seconds()
    return 1.0 + (elapsed_seconds / total_market_seconds) * 0.5

async def async_fetch_stock_data(ticker: str, session: aiohttp.ClientSession) -> pd.DataFrame:
    try:
        market_state = get_market_state()
        end = datetime.datetime.now()
        days = 1 if market_state == "Premarket" else 7 if market_state == "Open" else 10
        interval = "1m" if market_state in ["Premarket", "Open"] else "5m"
        data = await asyncio.to_thread(
            yf.download, ticker, start=end - datetime.timedelta(days=days), end=end,
            interval=interval, auto_adjust=True, progress=False, prepost=True
        )
        if data.empty or len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient data for {ticker}. Using fallback.")
            return pd.DataFrame(columns=['Datetime', 'Close', 'High', 'Low', 'Volume'])
        data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns
        required_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
        if not all(col in data.columns for col in required_cols):
            st.warning(f"Missing columns in {ticker} data. Filling with available data.")
            for col in required_cols:
                if col not in data.columns:
                    if 'Close' in data.columns:
                        data[col] = data['Close']
                    else:
                        data[col] = np.nan
        data = data[required_cols].dropna(how='all')
        data = data.astype(float).dropna()
        eastern = pytz.timezone('US/Eastern')
        data.index = data.index.tz_localize(pytz.utc).tz_convert(eastern)
        data['market_state'] = data.index.map(lambda x: get_market_state() if CONFIG['PREMARKET_START'] <= x.time() < CONFIG['MARKET_OPEN'] else "Open" if CONFIG['MARKET_OPEN'] <= x.time() <= CONFIG['MARKET_CLOSE'] else "Postmarket" if CONFIG['MARKET_CLOSE'] < x.time() <= CONFIG['POSTMARKET_END'] else "Closed")
        return data.reset_index(drop=False)
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return pd.DataFrame(columns=['Datetime', 'Close', 'High', 'Low', 'Volume'])

@st.cache_data(ttl=CONFIG['CACHE_TTL'], persist=True)
def get_stock_data(ticker: str, market_state: str = "Open") -> pd.DataFrame:
    async def fetch():
        async with aiohttp.ClientSession() as session:
            return await async_fetch_stock_data(ticker, session)
    return asyncio.run(fetch())

def get_current_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period='1d', interval='1m', prepost=True)
        return float(data['Close'].iloc[-1]) if not data.empty else 0.0
    except Exception:
        return 0.0

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                st.warning("Yahoo Finance rate limit reached. Waiting...")
                time.sleep(CONFIG['RATE_LIMIT_COOLDOWN'])
                return None
            if attempt == max_retries - 1:
                st.error(f"API call failed after {max_retries} attempts.")
                return None
            time.sleep(CONFIG['RETRY_DELAY'] * (attempt + 1))
    return None

def compute_indicators(df: pd.DataFrame, market_state: str = "Open") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    required_cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            if 'Close' in df.columns:
                df[col] = df['Close']
            else:
                df[col] = np.nan
    df = df[required_cols].dropna(how='all')
    if df.empty:
        return df
    close, high, low, volume = df['Close'].astype(float), df['High'].astype(float), df['Low'].astype(float), df['Volume'].astype(float)
    if market_state in ["Premarket", "Open"] and len(close) >= 14:
        df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        df['Stochastic'] = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch()
        df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        df['ATR_pct'] = df['ATR'] / close
    else:
        df[['EMA_9', 'EMA_20', 'RSI', 'Stochastic', 'ATR', 'ATR_pct']] = np.nan
    df['VWAP'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['price_momentum'] = df['Close'].pct_change(periods=CONFIG['PRICE_ACTION']['lookback_periods'])
    return df

@st.cache_data(ttl=CONFIG['CACHE_TTL'], persist=True)
def get_options_expiries(ticker: str) -> list[str]:
    try:
        return list(yf.Ticker(ticker).options) or []
    except Exception as e:
        if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
            st.warning("Rate limit reached. Waiting...")
            time.sleep(CONFIG['RATE_LIMIT_COOLDOWN'])
        return []

def fetch_options_data(ticker: str, expiries: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_calls, all_puts = pd.DataFrame(), pd.DataFrame()
    for expiry in expiries[:3]:
        chain = safe_api_call(yf.Ticker(ticker).option_chain, expiry)
        if chain:
            calls, puts = chain.calls, chain.puts
            calls['expiry'], puts['expiry'] = expiry, expiry
            all_calls = pd.concat([all_calls, calls[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]], ignore_index=True)
            all_puts = pd.concat([all_puts, puts[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']]], ignore_index=True)
        time.sleep(0.5)
    return all_calls, all_puts

def classify_moneyness(strike: float, spot: float) -> str:
    diff_pct = abs(strike - spot) / spot
    return 'ATM' if diff_pct < 0.01 else 'NTM' if diff_pct < 0.03 else 'ITM' if strike < spot else 'OTM'

def calculate_approximate_greeks(option: dict, spot_price: float) -> tuple[float, float, float]:
    moneyness = spot_price / option['strike']
    time_decay = calculate_time_decay_factor()
    if option['contractSymbol'].startswith('C'):
        delta = 0.95 if moneyness > 1.03 else 0.65 if moneyness > 1.0 else 0.50 if moneyness > 0.97 else 0.35
        gamma = 0.01 if moneyness > 1.03 else 0.05 if moneyness > 0.97 else 0.08
    else:
        delta = -0.95 if moneyness < 0.97 else -0.65 if moneyness < 1.0 else -0.50 if moneyness < 1.03 else -0.35
        gamma = 0.01 if moneyness < 0.97 else 0.05 if moneyness < 1.03 else 0.08
    theta = 0.05 if (datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date() - datetime.date.today()).days == 0 else 0.02
    return delta, gamma, theta * time_decay

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    required = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
    if any(pd.isna(option.get(f)) for f in required) or option['lastPrice'] <= 0:
        return False
    for g in ['delta', 'gamma', 'theta']:
        if pd.isna(option.get(g)):
            option[g] = calculate_approximate_greeks(option, spot_price)[['delta', 'gamma', 'theta'].index(g)]
    return all(not pd.isna(option.get(g)) for g in ['delta', 'gamma', 'theta'])

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> dict[str, float]:
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    volatility = stock_data.get('ATR_pct', 0.02)
    price_momentum = stock_data.get('price_momentum', 0.0)
    vol_factor = 1 + (volatility * 100)
    momentum_factor = 1 + abs(price_momentum) * 50
    thresholds['delta_base'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_factor * momentum_factor)) if side == 'call' else min(-0.3, max(-0.8, thresholds['delta_base'] * vol_factor * momentum_factor))
    thresholds['gamma_base'] = max(0.02, min(0.15, thresholds['gamma_base'] * vol_factor))
    thresholds['theta_base'] = min(0.1, thresholds['theta_base'] * calculate_time_decay_factor())
    thresholds['rsi_base'] = max(40.0, min(60.0, thresholds['rsi_min'] + (price_momentum * 1000))) if side == 'call' else max(30.0, min(50.0, thresholds['rsi_max'] + (price_momentum * 1000)))
    thresholds['stoch_base'] = max(50.0, min(80.0, thresholds['stoch_base'] + (price_momentum * 1000))) if side == 'call' else max(20.0, min(50.0, thresholds['stoch_base'] - (price_momentum * 1000)))
    thresholds['volume_min'] = max(500.0, min(3000.0, thresholds['volume_min'] * vol_factor))
    thresholds['volume_multiplier'] = max(0.8, min(2.5, thresholds['volume_multiplier_base'] * vol_factor))
    if is_0dte:
        thresholds['volume_multiplier'] *= 0.7
        thresholds['gamma_base'] *= 0.7
        thresholds['delta_base'] = max(0.4, thresholds['delta_base']) if side == 'call' else min(-0.4, thresholds['delta_base'])
    return {k: float(v) for k, v in thresholds.items()}

def calculate_holding_period(option: pd.Series, spot_price: float) -> str:
    days_to_expiry = (datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date() - datetime.date.today()).days
    intrinsic_value = max(0, spot_price - option['strike']) if option['contractSymbol'].startswith('C') else max(0, option['strike'] - spot_price)
    time_decay = calculate_time_decay_factor()
    return "Intraday (Exit before 3:30 PM)" if days_to_expiry == 0 else "1-2 days (Scalp quickly)" if intrinsic_value > 0 and option['theta'] * time_decay < -0.1 else "3-5 days (Swing trade)" if intrinsic_value > 0 else "1 day (Gamma play)" if days_to_expiry <= 3 else "3-7 days (Wait for move)"

def calculate_profit_targets(option: pd.Series) -> tuple[float, float]:
    entry_price = option['lastPrice'] * calculate_time_decay_factor()
    side = 'call' if option['contractSymbol'].startswith('C') else 'put'
    profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS'][side])
    stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
    return profit_target, stop_loss

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> dict:
    if stock_df.empty or not validate_option_data(option, stock_df.iloc[-1]['Close']):
        return {'signal': False, 'reason': 'Invalid data'}
    latest = stock_df.iloc[-1]
    thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
    conditions = [
        (float(option['delta']) >= thresholds['delta_base'] if side == 'call' else float(option['delta']) <= thresholds['delta_base'], f"Delta {'>=' if side == 'call' else '<='} {thresholds['delta_base']:.2f}"),
        (float(option['gamma']) >= thresholds['gamma_base'], f"Gamma >= {thresholds['gamma_base']:.3f}"),
        (float(option['theta']) <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}"),
        (float(latest['RSI']) > thresholds['rsi_base'] if side == 'call' else float(latest['RSI']) < thresholds['rsi_base'], f"RSI {'>' if side == 'call' else '<'} {thresholds['rsi_base']:.1f}"),
        (float(latest['Stochastic']) > thresholds['stoch_base'] if side == 'call' else float(latest['Stochastic']) < thresholds['stoch_base'], f"Stochastic {'>' if side == 'call' else '<'} {thresholds['stoch_base']:.1f}"),
        (float(option['volume']) > thresholds['volume_min'], f"Volume > {thresholds['volume_min']:.0f}")
    ]
    signal = all(c[0] for c in conditions)
    return {'signal': signal, 'passed_conditions': [c[1] for c in conditions if c[0]], 'failed_conditions': [c[1] for c in conditions if not c[0]], 'score': len([c for c in conditions if c[0]]) / len(conditions), 'profit_target': calculate_profit_targets(option)[0] if signal else None, 'stop_loss': calculate_profit_targets(option)[1] if signal else None, 'holding_period': calculate_holding_period(option, latest['Close']) if signal else None}

# =============================
# STREAMLIT INTERFACE
# =============================
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'ticker' not in st.session_state:
    st.session_state.ticker = "IWM"
if 'strike_range' not in st.session_state:
    st.session_state.strike_range = (-5.0, 5.0)
if 'moneyness_filter' not in st.session_state:
    st.session_state.moneyness_filter = ["NTM", "ATM"]
if 'show_0dte' not in st.session_state:
    st.session_state.show_0dte = False
if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True
if 'current_time' not in st.session_state:
    st.session_state.current_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = get_dynamic_refresh_interval()
if 'enable_auto_refresh' not in st.session_state:
    st.session_state.enable_auto_refresh = True
if 'rerun_pending' not in st.session_state:
    st.session_state.rerun_pending = False

# Welcome Modal
if st.session_state.show_welcome:
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        alert('Welcome to the Options Greeks Buy Signal Analyzer!\n\n' +
              '1. Select a stock ticker\n' +
              '2. Configure settings in the sidebar\n' +
              '3. View real-time signals and charts\n\n' +
              'Click OK to start analyzing!');
    });
    </script>
    """, unsafe_allow_html=True)
    st.session_state.show_welcome = False

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Real-time signal detection with dynamic thresholds and interactive charts**")

# Display market state and live clock
market_state = get_market_state()
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
with col1:
    if market_state == "Open":
        st.success(f"✅ Market {market_state}")
    elif market_state == "Premarket":
        st.warning(f"⏰ {market_state}")
    elif market_state == "Postmarket":
        st.info(f"🌙 {market_state}")
    else:
        st.info("💤 Market Closed")
with col2:
    current_price = get_current_price(st.session_state.ticker) if 'ticker' in st.session_state else 0.0
    st.metric("Current Price", f"${current_price:.2f}", help=f"Price for {st.session_state.ticker}")
with col3:
    st.session_state.time_placeholder = st.empty()
    st.session_state.time_placeholder.metric(
        "Current Market Time",
        st.session_state.current_time.strftime('%H:%M:%S %Z')
    )
with col4:
    if st.button("🔁 Refresh", key="manual_refresh"):
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.experimental_rerun()

st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    with st.expander("🔄 Auto-Refresh", expanded=True):
        try:
            st.session_state.enable_auto_refresh = st.checkbox(
                "Enable Auto-Refresh",
                value=st.session_state.get('enable_auto_refresh', True),
                key="auto_refresh"
            )
            if st.session_state.enable_auto_refresh:
                default_interval = get_dynamic_refresh_interval()
                st.session_state.refresh_interval = st.selectbox(
                    "Refresh Interval",
                    options=[30, 60, 120, 300],
                    index=[30, 60, 120, 300].index(st.session_state.get('refresh_interval', default_interval)) if st.session_state.get('refresh_interval', default_interval) in [30, 60, 120, 300] else 2,
                    format_func=lambda x: f"{x} seconds",
                    key="refresh_interval_select",
                    help="Select how often data refreshes automatically (adjusted by market state)"
                )
                st.info(f"Refreshing every {st.session_state.refresh_interval} seconds (Market: {market_state})")
            else:
                st.session_state.enable_auto_refresh = False
        except Exception as e:
            st.error(f"Error configuring auto-refresh: {str(e)}")
            st.session_state.refresh_interval = get_dynamic_refresh_interval()
    
    with st.expander("📊 Stock Selection", expanded=True):
        ticker_options = ["SPY", "QQQ", "AAPL", "IWM", "TSLA", "GLD", "TLT", "Other"]
        selected_ticker = st.selectbox(
            "Select Stock Ticker",
            ticker_options,
            index=ticker_options.index(st.session_state.get('ticker', 'IWM')),
            key="ticker_select",
            help="Choose a stock or select 'Other' for a custom ticker"
        )
        if selected_ticker == "Other":
            ticker = st.text_input(
                "Enter Custom Ticker",
                value=st.session_state.get('ticker', ''),
                key="custom_ticker",
                help="Enter a valid stock ticker (e.g., SPY, AAPL)"
            ).upper()
        else:
            ticker = selected_ticker.upper()
        if ticker:
            try:
                ticker_info = yf.Ticker(ticker).info
                st.success(f"Valid ticker: {ticker} ({ticker_info.get('shortName', ticker)})")
                if st.session_state.get('ticker') != ticker:
                    st.session_state.ticker = ticker
                    st.cache_data.clear()
                    st.experimental_rerun()
            except:
                st.error("Invalid ticker. Try again (e.g., SPY, AAPL).")
                ticker = st.session_state.get('ticker', 'IWM')
        else:
            ticker = st.session_state.get('ticker', 'IWM')
    
    with st.expander("📈 Signal Thresholds", expanded=True):
        if ticker:
            try:
                df = get_stock_data(ticker, market_state)
                if not df.empty:
                    df = compute_indicators(df, market_state)
                    latest = df.iloc[-1]
                    
                    call_thresholds = calculate_dynamic_thresholds(latest, "call", is_0dte=False)
                    put_thresholds = calculate_dynamic_thresholds(latest, "put", is_0dte=False)
                    
                    st.markdown('<p style="font-size: 0.9rem; margin-bottom: 8px;">Thresholds adapt to market volatility and momentum</p>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div style="font-size: 0.9rem; margin-bottom: 8px;"><strong>Calls</strong></div>', unsafe_allow_html=True)
                        st.markdown('<div style="font-size: 0.85rem; padding: 5px; border-radius: 5px; background-color: #e6f4ea;">', unsafe_allow_html=True)
                        SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider(
                            "Base Delta", 0.3, 0.8, float(call_thresholds['delta_base']), 0.01,
                            key="call_delta_slider",
                            help="Minimum delta for call signals"
                        )
                        SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider(
                            "Base Gamma", 0.02, 0.15, float(call_thresholds['gamma_base']), 0.001,
                            key="call_gamma_slider",
                            help="Minimum gamma for call signals"
                        )
                        SIGNAL_THRESHOLDS['call']['rsi_min'] = st.slider(
                            "Min RSI", 40.0, 70.0, float(call_thresholds['rsi_min']), 0.1,
                            key="call_rsi_slider",
                            help="Minimum RSI for call signals"
                        )
                        SIGNAL_THRESHOLDS['call']['stoch_base'] = st.slider(
                            "Stochastic", 50.0, 80.0, float(call_thresholds['stoch_base']), 0.1,
                            key="call_stoch_slider",
                            help="Minimum stochastic for call signals"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div style="font-size: 0.9rem; margin-bottom: 8px;"><strong>Puts</strong></div>', unsafe_allow_html=True)
                        st.markdown('<div style="font-size: 0.85rem; padding: 5px; border-radius: 5px; background-color: #f8d7da;">', unsafe_allow_html=True)
                        SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider(
                            "Base Delta", -0.8, -0.3, float(put_thresholds['delta_base']), 0.01,
                            key="put_delta_slider",
                            help="Maximum delta for put signals"
                        )
                        SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider(
                            "Base Gamma", 0.02, 0.15, float(put_thresholds['gamma_base']), 0.001,
                            key="put_gamma_slider",
                            help="Minimum gamma for put signals"
                        )
                        SIGNAL_THRESHOLDS['put']['rsi_max'] = st.slider(
                            "Max RSI", 30.0, 60.0, float(put_thresholds['rsi_max']), 0.1,
                            key="put_rsi_slider",
                            help="Maximum RSI for put signals"
                        )
                        SIGNAL_THRESHOLDS['put']['stoch_base'] = st.slider(
                            "Stochastic", 20.0, 50.0, float(put_thresholds['stoch_base']), 0.1,
                            key="put_stoch_slider",
                            help="Maximum stochastic for put signals"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading signal thresholds: {str(e)}")
    
    with st.expander("🎯 Trade Settings", expanded=True):
        st.markdown('<p style="font-size: 0.9rem; margin-bottom: 8px;">Adjust profit targets and stop loss</p>', unsafe_allow_html=True)
        CONFIG['PROFIT_TARGETS']['call'] = st.slider(
            "Call Profit Target (%)", 5.0, 50.0, CONFIG['PROFIT_TARGETS']['call'] * 100, 1.0,
            key="call_profit_target",
            help="Target profit percentage for call options"
        ) / 100
        CONFIG['PROFIT_TARGETS']['put'] = st.slider(
            "Put Profit Target (%)", 5.0, 50.0, CONFIG['PROFIT_TARGETS']['put'] * 100, 1.0,
            key="put_profit_target",
            help="Target profit percentage for put options"
        ) / 100
        CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider(
            "Stop Loss (%)", 5.0, 20.0, CONFIG['PROFIT_TARGETS']['stop_loss'] * 100, 1.0,
            key="stop_loss",
            help="Stop loss percentage for all trades"
        ) / 100
    
    with st.expander("🔍 Filters", expanded=True):
        if ticker and current_price > 0:
            try:
                strike_min = current_price * (1 + st.session_state.strike_range[0] / 100)
                strike_max = current_price * (1 + st.session_state.strike_range[1] / 100)
                st.session_state.strike_range = st.slider(
                    "Strike Price Range (% from current)",
                    -20.0, 20.0, st.session_state.strike_range, 0.5,
                    key="strike_range_slider",
                    help="Filter options by strike price relative to current stock price"
                )
                st.session_state.moneyness_filter = st.multiselect(
                    "Moneyness Filter",
                    ["ITM", "NTM", "ATM", "OTM"],
                    default=st.session_state.moneyness_filter,
                    key="moneyness_filter_select",
                    help="Filter options by moneyness (In-The-Money, Near-The-Money, At-The-Money, Out-Of-The-Money)"
                )
                st.session_state.show_0dte = st.checkbox(
                    "Show 0DTE Options Only",
                    value=st.session_state.get('show_0dte', False),
                    key="show_0dte",
                    help="Show only options expiring today (0 days to expiration)"
                )
            except Exception as e:
                st.error(f"Error configuring filters: {str(e)}")

# Main Content
if ticker:
    try:
        with st.spinner("Fetching and analyzing data..."):
            df = get_stock_data(ticker, market_state)
            if df.empty:
                st.error("No stock data available.")
                st.stop()
            df = compute_indicators(df, market_state)
            expiries = get_options_expiries(ticker)
            if not expiries:
                st.warning("No options data available.")
                st.stop()
            calls, puts = fetch_options_data(ticker, expiries)
            if calls.empty and puts.empty:
                st.warning("No options data retrieved.")
                st.stop()
            current_price = get_current_price(ticker)
            if current_price == 0.0:
                st.error("Unable to fetch current price.")
                st.stop()
            calls['moneyness'] = calls['strike'].apply(lambda x: classify_moneyness(x, current_price))
            puts['moneyness'] = puts['strike'].apply(lambda x: classify_moneyness(x, current_price))
            calls['is_0dte'] = calls['expiry'].apply(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d").date() - datetime.date.today()).days == 0)
            puts['is_0dte'] = puts['expiry'].apply(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d").date() - datetime.date.today()).days == 0)
            if st.session_state.show_0dte:
                calls, puts = calls[calls['is_0dte']], puts[puts['is_0dte']]
            strike_min, strike_max = current_price * (1 + st.session_state.strike_range[0] / 100), current_price * (1 + st.session_state.strike_range[1] / 100)
            calls, puts = calls[(calls['strike'] >= strike_min) & (calls['strike'] <= strike_max)], puts[(puts['strike'] >= strike_min) & (puts['strike'] <= strike_max)]
            if st.session_state.moneyness_filter:
                calls, puts = calls[calls['moneyness'].isin(st.session_state.moneyness_filter)], puts[puts['moneyness'].isin(st.session_state.moneyness_filter)]
            call_signals, put_signals = [], []
            for _, option in calls.iterrows():
                signal = generate_signal(option, "call", df, option['is_0dte'])
                if signal['signal']:
                    call_signals.append({**option, **{k: v for k, v in signal.items() if k in ['score', 'profit_target', 'stop_loss', 'holding_period', 'passed_conditions']}})
            for _, option in puts.iterrows():
                signal = generate_signal(option, "put", df, option['is_0dte'])
                if signal['signal']:
                    put_signals.append({**option, **{k: v for k, v in signal.items() if k in ['score', 'profit_target', 'stop_loss', 'holding_period', 'passed_conditions']}})
            call_signals_df, put_signals_df = pd.DataFrame(call_signals), pd.DataFrame(put_signals)
    
        tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Chart", "🔍 Technicals"])
        with tab1:
            st.subheader("Buy Signals")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Call Signals")
                for _, signal in call_signals_df.iterrows():
                    with st.container():
                        st.markdown(f'<div class="signal-alert call-metric">**{signal.name}** (Score: {signal["score"]:.2%})</div>', unsafe_allow_html=True)
                        st.markdown(f"Strike: ${signal['strike']:.2f} | Price: ${signal['lastPrice']:.2f} | Volume: {signal['volume']:.0f}")
                        st.markdown(f"Profit Target: ${signal['profit_target']:.2f} | Stop Loss: ${signal['stop_loss']:.2f} | Holding: {signal['holding_period']}")
            with col2:
                st.markdown("### Put Signals")
                for _, signal in put_signals_df.iterrows():
                    with st.container():
                        st.markdown(f'<div class="signal-alert put-metric">**{signal.name}** (Score: {signal['score']:.2%})</div>', unsafe_allow_html=True)
                        st.markdown(f"Strike: ${signal['strike']:.2f} | Price: ${signal['lastPrice']:.2f} | Volume: {signal['volume']:.0f}")
                        st.markdown(f"Profit Target: ${signal['profit_target']:.2f} | Stop Loss: ${signal['stop_loss']:.2f} | Holding: {signal['holding_period']}")
        with tab2:
            st.subheader("Price Chart")
            if not df.empty:
                fig = go.Figure(data=[go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                for col, color in [('EMA_9', 'blue'), ('EMA_20', 'orange'), ('VWAP', 'purple')]:
                    if col in df.columns and not df[col].isna().all():
                        fig.add_trace(go.Scatter(x=df['Datetime'], y=df[col], name=col, line=dict(color=color, dash='dash' if col == 'VWAP' else 'solid')))
                fig.update_layout(title=f"{ticker} Price Action", xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.subheader("Technical Indicators")
            if not df.empty:
                latest = df.iloc[-1]
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("RSI", f"{latest['RSI']:.1f}" if not pd.isna(latest['RSI']) else "N/A")
                with col2: st.metric("ATR %", f"{latest['ATR_pct']*100:.2f}%" if not pd.isna(latest['ATR_pct']) else "N/A")
                with col3: st.metric("Volume", f"{latest['Volume']:.0f}")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please select a ticker in the sidebar to begin analysis.")

manage_auto_refresh()

# Dynamic time update in a non-blocking manner
def update_time():
    while True:
        st.session_state.current_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
        if 'time_placeholder' in st.session_state:
            st.session_state.time_placeholder.metric(
                "Current Market Time",
                st.session_state.current_time.strftime('%H:%M:%S %Z')
            )
        time.sleep(1)

if __name__ == "__main__":
    time_thread = threading.Thread(target=update_time, daemon=True)
    time_thread.start()
