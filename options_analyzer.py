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

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Apply custom CSS for modern, professional look
st.markdown("""
<style>
/* General Styling */
.stApp { background-color: #f8f9fa; font-family: 'Segoe UI', Arial, sans-serif; }
h1, h2, h3 { color: #1a3c6e; font-weight: 600; }
.stButton>button { 
    background-color: #1a3c6e; 
    color: white; 
    border-radius: 8px; 
    padding: 10px 20px; 
    font-weight: 500; 
    transition: background-color 0.3s; 
}
.stButton>button:hover { background-color: #2c5a9e; }
.stSelectbox, .stTextInput, .stSlider, .stCheckbox { 
    background-color: #ffffff; 
    border-radius: 8px; 
    padding: 8px; 
    border: 1px solid #d1d5db; 
}

/* Sidebar Styling */
.stSidebar { 
    background-color: #ffffff !important; 
    border-right: 1px solid #e5e7eb; 
    padding: 20px; 
}
.stSidebar h2 { font-size: 1.4rem; color: #1a3c6e; }

/* Metric Styling */
.stMetric { 
    background-color: #ffffff; 
    border-radius: 8px; 
    padding: 15px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    margin-bottom: 15px; 
}
.call-metric { border-left: 5px solid #28a745; }
.put-metric { border-left: 5px solid #dc3545; }

/* Table Styling */
.signal-table th { 
    background-color: #1a3c6e; 
    color: white; 
    padding: 12px; 
    font-weight: 500; 
}
.signal-table td { padding: 10px; }

/* Tooltip Styling */
.tooltip { position: relative; display: inline-block; cursor: pointer; }
.tooltip .tooltiptext { 
    visibility: hidden; 
    width: 220px; 
    background-color: #1a3c6e; 
    color: #fff; 
    text-align: center; 
    border-radius: 6px; 
    padding: 8px; 
    position: absolute; 
    z-index: 1; 
    bottom: 125%; 
    left: 50%; 
    margin-left: -110px; 
    opacity: 0; 
    transition: opacity 0.3s; 
}
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
.signal-alert { 
    animation: pulse 1.5s infinite; 
    border-radius: 8px; 
    padding: 15px; 
    margin-bottom: 15px; 
}
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
    'CACHE_TTL': 300,  # forceful cache refresh after 5 minutes
    'RATE_LIMIT_COOLDOWN': 180,  # 3 minutes
    'MARKET_OPEN': datetime.time(9, 30),
    'MARKET_CLOSE': datetime.time(16, 0),
    'PREMARKET_START': datetime.time(4, 0),
    'POSTMARKET_END': datetime.time(20, 0),
    'VOLATILITY_THRESHOLDS': {
        'low': 0.015,
        'medium': 0.03,
        'high': 0.05
    },
    'PROFIT_TARGETS': {
        'call': 0.15,
        'put': 0.15,
        'stop_loss': 0.08
    },
    'PRICE_ACTION': {
        'lookback_periods': 5,
        'momentum_threshold': 0.01,
        'breakout_threshold': 0.02
    }
}

CONSTANTS = {
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 1,
    'DATA_TIMEOUT': 30,
    'MIN_DATA_POINTS': 50,
    'CACHE_TTL': 300,
    'RATE_LIMIT_COOLDOWN': 180,
    'MARKET_OPEN': datetime.time(9, 30),
    'MARKET_CLOSE': datetime.time(16, 0),
    'PREMARKET_START': datetime.time(4, 0)
}

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_base': 0.5,
        'delta_vol_multiplier': 0.15,
        'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.03,
        'theta_base': 0.05,
        'rsi_base': 50.0,
        'rsi_min': 50.0,
        'rsi_max': 70.0,
        'stoch_base': 60.0,
        'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.4,
        'volume_min': 1000.0,
        'price_momentum_min': 0.005
    },
    'put': {
        'delta_base': -0.5,
        'delta_vol_multiplier': 0.15,
        'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.03,
        'theta_base': 0.05,
        'rsi_base': 50.0,
        'rsi_min': 30.0,
        'rsi_max': 50.0,
        'stoch_base': 40.0,
        'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.4,
        'volume_min': 1000.0,
        'price_momentum_min': -0.005
    }
}

# =============================
# AUTO-REFRESH SYSTEM
# =============================
def manage_auto_refresh():
    if st.session_state.get('enable_auto_refresh', False) and 'refresh_interval' in st.session_state:
        refresh_interval = st.session_state.refresh_interval
        last_refresh = st.session_state.get('last_refresh', time.time())
        if time.time() - last_refresh >= refresh_interval:
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_counter += 1
            st.cache_data.clear()
            st.rerun()

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

def get_dynamic_refresh_interval() -> int:
    market_state = get_market_state()
    if market_state in ["Premarket", "Open"]:
        return 30
    elif market_state == "Postmarket":
        return 120
    return 300

def is_market_open() -> bool:
    return get_market_state() == "Open"

def is_premarket() -> bool:
    return get_market_state() == "Premarket"

def is_early_market() -> bool:
    if not is_market_open():
        return False
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open_today = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'])
    market_open_today = eastern.localize(market_open_today)
    return (now - market_open_today).total_seconds() < 1800

def calculate_time_decay_factor() -> float:
    if not is_market_open():
        return 1.0
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'], tzinfo=eastern)
    market_close = datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE'], tzinfo=eastern)
    total_market_seconds = (market_close - market_open).total_seconds()
    elapsed_seconds = (now - market_open).total_seconds()
    decay_factor = 1.0 + (elapsed_seconds / total_market_seconds) * 0.5
    return decay_factor

async def async_fetch_stock_data(ticker: str, session: aiohttp.ClientSession) -> pd.DataFrame:
    try:
        market_state = get_market_state()
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=10 if market_state != "Premarket" else 1)
        interval = "1m" if market_state in ["Premarket", "Open"] else "5m"
        data = await asyncio.to_thread(
            yf.download,
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            prepost=True
        )
        if data.empty:
            st.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()

        data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        data = data.dropna(how='all')
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=required_cols)

        if len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient data points ({len(data)}). Need at least {CONFIG['MIN_DATA_POINTS']}.")
            return pd.DataFrame()

        eastern = pytz.timezone('US/Eastern')
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc)
        data.index = data.index.tz_convert(eastern)
        data['market_state'] = data.index.map(
            lambda x: "Premarket" if CONFIG['PREMARKET_START'] <= x.time() < CONFIG['MARKET_OPEN']
            else "Open" if CONFIG['MARKET_OPEN'] <= x.time() <= CONFIG['MARKET_CLOSE']
            else "Postmarket" if CONFIG['MARKET_CLOSE'] < x.time() <= CONFIG['POSTMARKET_END']
            else "Closed"
        )
        return data.reset_index(drop=False)
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data(ticker: str, market_state: str = "Open") -> pd.DataFrame:
    async def fetch():
        async with aiohttp.ClientSession() as session:
            return await async_fetch_stock_data(ticker, session)
    return asyncio.run(fetch())

def get_current_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        st.warning(f"No current price data for {ticker}")
        return 0.0
    except Exception as e:
        st.error(f"Error getting current price for {ticker}: {str(e)}")
        return 0.0

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                st.warning("Yahoo Finance rate limit reached. Please wait a few minutes.")
                st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
                return None
            if attempt == max_retries - 1:
                st.error(f"API call failed after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(CONFIG['RETRY_DELAY'] * (attempt + 1))
    return None

def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df['avg_vol'] = np.nan
    for date, group in df.groupby(df['Datetime'].dt.date):
        regular = group[group['market_state'] == "Open"]
        if not regular.empty:
            regular_avg_vol = regular['Volume'].expanding(min_periods=1).mean()
            df.loc[regular.index, 'avg_vol'] = regular_avg_vol
        premarket = group[group['market_state'] == "Premarket"]
        if not premarket.empty:
            premarket_avg_vol = premarket['Volume'].expanding(min_periods=1).mean()
            df.loc[premarket.index, 'avg_vol'] = premarket_avg_vol
        postmarket = group[group['market_state'] == "Postmarket"]
        if not postmarket.empty:
            postmarket_avg_vol = postmarket['Volume'].expanding(min_periods=1).mean()
            df.loc[postmarket.index, 'avg_vol'] = postmarket_avg_vol
    overall_avg = df['Volume'].mean()
    df['avg_vol'] = df['avg_vol'].fillna(overall_avg)
    return df

def compute_price_action(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    lookback = CONFIG['PRICE_ACTION']['lookback_periods']
    df['price_momentum'] = df['Close'].pct_change(periods=lookback)
    df['range_high'] = df['High'].rolling(window=lookback).max()
    df['range_low'] = df['Low'].rolling(window=lookback).min()
    df['breakout_up'] = df['Close'] > df['range_high'].shift(1) * (1 + CONFIG['PRICE_ACTION']['breakout_threshold'])
    df['breakout_down'] = df['Close'] < df['range_low'].shift(1) * (1 - CONFIG['PRICE_ACTION']['breakout_threshold'])
    return df

def compute_indicators(df: pd.DataFrame, market_state: str = "Open") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    required_cols = ['Close', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)

    if market_state in ["Premarket", "Open"]:
        if len(close) >= 9:
            df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        if len(close) >= 20:
            df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        if len(close) >= 14:
            df['RSI'] = RSIIndicator(close=close, window=14).rsi()
            df['Stochastic'] = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch()
        if len(close) >= 14:
            df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
            df['ATR_pct'] = df['ATR'] / close
    else:
        df['EMA_9'] = np.nan
        df['EMA_20'] = np.nan
        df['RSI'] = np.nan
        df['Stochastic'] = np.nan
        df['ATR'] = np.nan
        df['ATR_pct'] = np.nan

    df['VWAP'] = np.nan
    for session, group in df.groupby(pd.Grouper(key='Datetime', freq='D')):
        if group.empty:
            continue
        session_vwap = ((group['High'] + group['Low'] + group['Close']) / 3 * group['Volume']).cumsum() / group['Volume'].cumsum()
        df.loc[group.index, 'VWAP'] = session_vwap

    df = calculate_volume_averages(df)
    df = compute_price_action(df)
    return df

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> list[str]:
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        return list(expiries) if expiries else []
    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
            st.warning("Yahoo Finance rate limit reached. Please wait a few minutes.")
            st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
        else:
            st.error(f"Error fetching expiries: {error_msg}")
        return []

def fetch_options_data(ticker: str, expiries: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_calls = pd.DataFrame()
    all_puts = pd.DataFrame()
    failed_expiries = []
    stock = yf.Ticker(ticker)
    
    for expiry in expiries:
        try:
            chain = safe_api_call(stock.option_chain, expiry)
            if chain is None:
                failed_expiries.append(expiry)
                continue
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            required_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
            for df_name, df in [('calls', calls), ('puts', puts)]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing columns in {df_name} for {expiry}: {missing_cols}")
                    if 'delta' not in df.columns:
                        df['delta'] = np.nan
                    if 'gamma' not in df.columns:
                        df['gamma'] = np.nan
                    if 'theta' not in df.columns:
                        df['theta'] = np.nan
                else:
                    if 'delta' not in df.columns:
                        df['delta'] = np.nan
                    if 'gamma' not in df.columns:
                        df['gamma'] = np.nan
                    if 'theta' not in df.columns:
                        df['theta'] = np.nan
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
            time.sleep(0.5)
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                st.warning("Yahoo Finance rate limit reached. Please wait a few minutes.")
                st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
                break
            st.warning(f"Failed to fetch options for {expiry}: {error_msg}")
            failed_expiries.append(expiry)
            continue
    if failed_expiries:
        st.info(f"Failed to fetch data for expiries: {failed_expiries}")
    return all_calls, all_puts

def classify_moneyness(strike: float, spot: float) -> str:
    diff = abs(strike - spot)
    diff_pct = diff / spot
    if diff_pct < 0.01:
        return 'ATM'
    elif strike < spot:
        if diff_pct < 0.03:
            return 'NTM'
        else:
            return 'ITM'
    else:
        if diff_pct < 0.03:
            return 'NTM'
        else:
            return 'OTM'

def calculate_approximate_greeks(option: dict, spot_price: float) -> tuple[float, float, float]:
    moneyness = spot_price / option['strike']
    time_decay = calculate_time_decay_factor()
    
    if option['contractSymbol'].startswith('C'):
        if moneyness > 1.03:
            delta = 0.95
            gamma = 0.01
        elif moneyness > 1.0:
            delta = 0.65
            gamma = 0.05
        elif moneyness > 0.97:
            delta = 0.50
            gamma = 0.08
        else:
            delta = 0.35
            gamma = 0.05
    else:
        if moneyness < 0.97:
            delta = -0.95
            gamma = 0.01
        elif moneyness < 1.0:
            delta = -0.65
            gamma = 0.05
        elif moneyness < 1.03:
            delta = -0.50
            gamma = 0.08
        else:
            delta = -0.35
            gamma = 0.05
    
    expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - datetime.date.today()).days
    theta = (0.05 if days_to_expiry == 0 else 0.02) * time_decay
    return delta, gamma, theta

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    required_fields = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
    missing_fields = [field for field in required_fields if field not in option or pd.isna(option[field])]
    if missing_fields:
        return False
    if option['lastPrice'] <= 0:
        return False
    if pd.isna(option.get('delta')) or pd.isna(option.get('gamma')) or pd.isna(option.get('theta')):
        delta, gamma, theta = calculate_approximate_greeks(option, spot_price)
        option['delta'] = delta
        option['gamma'] = gamma
        option['theta'] = theta
    if pd.isna(option['delta']) or pd.isna(option['gamma']) or pd.isna(option['theta']):
        return False
    return True

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> dict[str, float]:
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    volatility = float(stock_data.get('ATR_pct', 0.02)) if not pd.isna(stock_data.get('ATR_pct')) else 0.02
    price_momentum = float(stock_data.get('price_momentum', 0.0)) if not pd.isna(stock_data.get('price_momentum')) else 0.0
    breakout_up = bool(stock_data.get('breakout_up', False))
    breakout_down = bool(stock_data.get('breakout_down', False))
    time_decay = calculate_time_decay_factor()
    
    vol_factor = 1 + (volatility * 100)
    momentum_factor = 1 + abs(price_momentum) * 50
    
    if side == 'call':
        thresholds['delta_base'] = max(0.3, min(0.8, SIGNAL_THRESHOLDS['call']['delta_base'] * vol_factor * momentum_factor))
        if breakout_up:
            thresholds['delta_base'] = min(thresholds['delta_base'] * 1.2, 0.85)
        thresholds['delta_min'] = thresholds['delta_base']
    else:
        thresholds['delta_base'] = min(-0.3, max(-0.8, SIGNAL_THRESHOLDS['put']['delta_base'] * vol_factor * momentum_factor))
        if breakout_down:
            thresholds['delta_base'] = max(thresholds['delta_base'] * 0.8, -0.85)
        thresholds['delta_max'] = thresholds['delta_base']
    
    thresholds['gamma_base'] = max(0.02, min(0.15, SIGNAL_THRESHOLDS[side]['gamma_base'] * vol_factor))
    thresholds['theta_base'] = min(0.1, SIGNAL_THRESHOLDS[side]['theta_base'] * time_decay)
    
    if side == 'call':
        rsi_base = float(max(40.0, min(60.0, SIGNAL_THRESHOLDS['call']['rsi_min'] + (price_momentum * 1000))))
        thresholds['rsi_base'] = rsi_base
        thresholds['rsi_min'] = rsi_base
        stoch_base = float(max(50.0, min(80.0, SIGNAL_THRESHOLDS['call']['stoch_base'] + (price_momentum * 1000))))
        thresholds['stoch_base'] = stoch_base
        thresholds['volume_min'] = float(max(500.0, min(3000.0, SIGNAL_THRESHOLDS[side]['volume_min'] * vol_factor)))
        thresholds['volume_multiplier'] = float(max(0.8, min(2.5, thresholds['volume_multiplier_base'] * vol_factor)))
    else:
        rsi_base = float(max(30.0, min(50.0, SIGNAL_THRESHOLDS['put']['rsi_max'] + (price_momentum * 1000))))
        thresholds['rsi_base'] = rsi_base
        thresholds['rsi_max'] = rsi_base
        stoch_base = float(max(20.0, min(50.0, SIGNAL_THRESHOLDS['put']['stoch_base'] - (price_momentum * 1000))))
        thresholds['stoch_base'] = stoch_base
        thresholds['volume_min'] = float(max(500.0, min(3000.0, SIGNAL_THRESHOLDS[side]['volume_min'] * vol_factor)))
        thresholds['volume_multiplier'] = float(max(0.8, min(2.5, thresholds['volume_multiplier_base'] * vol_factor)))
    
    if is_premarket() or is_early_market():
        if side == 'call':
            thresholds['delta_min'] = max(0.35, thresholds['delta_min'])
        else:
            thresholds['delta_max'] = min(-0.35, thresholds['delta_max'])
        thresholds['volume_multiplier'] = float(thresholds['volume_multiplier'] * 0.6)
        thresholds['gamma_base'] = float(thresholds['gamma_base'] * 0.8)
        thresholds['volume_min'] = float(max(300.0, thresholds['volume_min'] * 0.5))

    if is_0dte:
        thresholds['volume_multiplier'] = float(thresholds['volume_multiplier'] * 0.7)
        thresholds['gamma_base'] = float(thresholds['gamma_base'] * 0.7)
        if side == 'call':
            thresholds['delta_min'] = max(0.4, thresholds['delta_min'])
        else:
            thresholds['delta_max'] = min(-0.4, thresholds['delta_max'])
    
    for key, value in thresholds.items():
        if not isinstance(value, (int, float)) or pd.isna(value):
            st.warning(f"Invalid threshold value for {key}: {value}. Using default.")
            thresholds[key] = float(SIGNAL_THRESHOLDS[side][key])
    
    return thresholds

def calculate_holding_period(option: pd.Series, spot_price: float) -> str:
    expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - datetime.date.today()).days
    if days_to_expiry == 0:
        return "Intraday (Exit before 3:30 PM)"
    if option['contractSymbol'].startswith('C'):
        intrinsic_value = max(0, spot_price - option['strike'])
    else:
        intrinsic_value = max(0, option['strike'] - spot_price)
    time_decay = calculate_time_decay_factor()
    if intrinsic_value > 0:
        if option['theta'] * time_decay < -0.1:
            return "1-2 days (Scalp quickly)"
        else:
            return "3-5 days (Swing trade)"
    else:
        if days_to_expiry <= 3:
            return "1 day (Gamma play)"
        else:
            return "3-7 days (Wait for move)"

def calculate_profit_targets(option: pd.Series) -> tuple[float, float]:
    entry_price = option['lastPrice'] * calculate_time_decay_factor()
    profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['call' if option['contractSymbol'].startswith('C') else 'put'])
    stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
    return profit_target, stop_loss

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> dict:
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available'}
    
    current_price = stock_df.iloc[-1]['Close']
    if not validate_option_data(option, current_price):
        return {'signal': False, 'reason': 'Insufficient option data'}
    
    latest = stock_df.iloc[-1]
    try:
        thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        option_volume = float(option['volume'])
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        stochastic = float(latest['Stochastic']) if not pd.isna(latest['Stochastic']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume
        price_momentum = float(latest['price_momentum']) if not pd.isna(latest['price_momentum']) else 0.0
        breakout_up = latest['breakout_up']
        breakout_down = latest['breakout_down']
        
        conditions = []
        if side == "call":
            volume_ok = option_volume > thresholds['volume_min']
            conditions = [
                (delta >= thresholds['delta_min'], f"Delta ≥ {thresholds['delta_min']:.2f}", delta),
                (gamma >= thresholds['gamma_base'], f"Gamma ≥ {thresholds['gamma_base']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta ≤ {theta:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']:.1f}", rsi),
                (stochastic is not None and stochastic > thresholds['stoch_base'], f"Stochastic > {thresholds['stoch_base']:.1f}", stochastic),
                (vwap is not None and close > vwap, "Price > VWAP", f"{close:.2f} > {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}"),
                (price_momentum >= thresholds['price_momentum_min'] or breakout_up, f"Price Momentum ≥ {thresholds['price_momentum_min']*100:.2f}% or Breakout", f"{price_momentum*100:.2f}%")
            ]
        else:
            volume_ok = option_volume > thresholds['volume_min']
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta ≤ {thresholds['delta_max']:.2f}", delta),
                (gamma >= thresholds['gamma_base'], f"Gamma ≥ {thresholds['gamma_base']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta ≤ {theta:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']:.1f}", rsi),
                (stochastic is not None and stochastic < thresholds['stoch_base'], f"Stochastic < {thresholds['stoch_base']:.1f}", stochastic),
                (vwap is not None and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}"),
                (price_momentum <= thresholds['price_momentum_min'] or breakout_down, f"Price Momentum ≤ {thresholds['price_momentum_min']*100:.2f}% or Breakout", f"{price_momentum*100:.2f}%")
            ]
        
        passed_conditions = [desc for passed, desc, val in conditions if passed]
        failed_conditions = [f"{desc} (got {val})" for passed, desc, val in conditions if not passed]
        signal = all(passed for passed, desc, val in conditions)
        
        profit_target = None
        stop_loss = None
        holding_period = None
        if signal:
            profit_target, stop_loss = calculate_profit_targets(option)
            holding_period = calculate_holding_period(option, current_price)
        
        return {
            'signal': signal,
            'passed_conditions': passed_conditions,
            'failed_conditions': failed_conditions,
            'score': len(passed_conditions) / len(conditions),
            'thresholds': thresholds,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period
        }
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}'}

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

if 'rate_limited_until' in st.session_state:
    if time.time() < st.session_state['rate_limited_until']:
        remaining = int(st.session_state['rate_limited_until'] - time.time())
        st.warning(f"Yahoo Finance API rate limited. Please wait {remaining} seconds.")
        with st.expander("ℹ️ About Rate Limiting"):
            st.markdown("""
            Yahoo Finance may restrict data retrieval frequency. If rate limited, please:
            - Wait a few minutes before refreshing
            - Avoid auto-refresh intervals below 30 seconds
            - Use one ticker at a time
            """)
        st.stop()
    else:
        del st.session_state['rate_limited_until']

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
    st.metric("Current Price", f"${current_price:.2f}")
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
        st.rerun()

st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    with st.expander("🔄 Auto-Refresh", expanded=True):
        st.session_state.enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.get('enable_auto_refresh', True), key="auto_refresh")
        if st.session_state.enable_auto_refresh:
            default_interval = get_dynamic_refresh_interval()
            try:
                st.session_state.refresh_interval = st.selectbox(
                    "Refresh Interval",
                    options=[30, 60, 120, 300],
                    index=[30, 60, 120, 300].index(st.session_state.get('refresh_interval', default_interval)) if st.session_state.get('refresh_interval', default_interval) in [30, 60, 120, 300] else 1,
                    format_func=lambda x: f"{x} seconds",
                    key="refresh_interval_select",
                    help="Select how often data refreshes automatically (adjusted by market state)"
                )
                st.info(f"Refreshing every {st.session_state.refresh_interval} seconds (Market: {market_state})")
            except Exception as e:
                st.error(f"Error setting refresh interval: {str(e)}")
                st.session_state.refresh_interval = default_interval
        else:
            st.session_state.enable_auto_refresh = False
    
    with st.expander("📊 Stock Selection", expanded=True):
        ticker_options = ["SPY", "QQQ", "AAPL", "IWM", "TSLA", "GLD", "TLT", "Other"]
        selected_ticker = st.selectbox(
            "Select Stock Ticker",
            ticker_options,
            key="ticker_select",
            help="Choose a stock or select 'Other' for a custom ticker"
        )
        if selected_ticker == "Other":
            ticker = st.text_input(
                "Enter Custom Ticker",
                value=st.session_state.ticker,
                key="custom_ticker",
                help="Enter a valid stock ticker (e.g., SPY, AAPL)"
            ).upper()
        else:
            ticker = selected_ticker.upper()
        if ticker:
            try:
                ticker_info = yf.Ticker(ticker).info
                st.success(f"Valid ticker: {ticker} ({ticker_info.get('shortName', ticker)})")
                st.session_state.ticker = ticker
            except:
                st.error("Invalid ticker. Try again (e.g., SPY, AAPL).")
                ticker = ""
    
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
        if ticker:
            current_price = get_current_price(ticker)
            if current_price > 0:
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
                    value=st.session_state.show_0dte,
                    key="show_0dte",
                    help="Show only options expiring today (0 days to expiration)"
                )

# Main Content
if ticker:
    try:
        with st.spinner("Fetching and analyzing data..."):
            df = get_stock_data(ticker, market_state)
            if df.empty:
                st.error("No stock data available. Try a different ticker or refresh later.")
                st.stop()
            
            df = compute_indicators(df, market_state)
            expiries = get_options_expiries(ticker)
            if not expiries:
                st.warning("No options data available for this ticker.")
                st.stop()
            
            calls, puts = fetch_options_data(ticker, expiries[:3])
            if calls.empty and puts.empty:
                st.warning("No options data retrieved. Try refreshing or selecting a different ticker.")
                st.stop()
            
            current_price = get_current_price(ticker)
            if current_price == 0.0:
                st.error("Unable to fetch current price. Try again later.")
                st.stop()
            
            calls['moneyness'] = calls['strike'].apply(lambda x: classify_moneyness(x, current_price))
            puts['moneyness'] = puts['strike'].apply(lambda x: classify_moneyness(x, current_price))
            
            calls['is_0dte'] = calls['expiry'].apply(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d").date() - datetime.date.today()).days == 0)
            puts['is_0dte'] = puts['expiry'].apply(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d").date() - datetime.date.today()).days == 0)
            
            if st.session_state.show_0dte:
                calls = calls[calls['is_0dte']]
                puts = puts[puts['is_0dte']]
            
            strike_min = current_price * (1 + st.session_state.strike_range[0] / 100)
            strike_max = current_price * (1 + st.session_state.strike_range[1] / 100)
            calls = calls[(calls['strike'] >= strike_min) & (calls['strike'] <= strike_max)]
            puts = puts[(puts['strike'] >= strike_min) & (puts['strike'] <= strike_max)]
            
            if st.session_state.moneyness_filter:
                calls = calls[calls['moneyness'].isin(st.session_state.moneyness_filter)]
                puts = puts[puts['moneyness'].isin(st.session_state.moneyness_filter)]
            
            call_signals = []
            put_signals = []
            for _, call in calls.iterrows():
                signal = generate_signal(call, "call", df, call['is_0dte'])
                if signal['signal']:
                    call_signals.append({
                        'Contract': call['contractSymbol'],
                        'Strike': call['strike'],
                        'Expiry': call['expiry'],
                        'Moneyness': call['moneyness'],
                        'Last Price': call['lastPrice'],
                        'Volume': call['volume'],
                        'Implied Volatility': call['impliedVolatility'],
                        'Delta': call['delta'],
                        'Gamma': call['gamma'],
                        'Theta': call['theta'],
                        'Signal Score': signal['score'],
                        'Profit Target': signal['profit_target'],
                        'Stop Loss': signal['stop_loss'],
                        'Holding Period': signal['holding_period'],
                        'Conditions': "; ".join(signal['passed_conditions']),
                        'Is 0DTE': call['is_0dte']
                    })
            for _, put in puts.iterrows():
                signal = generate_signal(put, "put", df, put['is_0dte'])
                if signal['signal']:
                    put_signals.append({
                        'Contract': put['contractSymbol'],
                        'Strike': put['strike'],
                        'Expiry': put['expiry'],
                        'Moneyness': put['moneyness'],
                        'Last Price': put['lastPrice'],
                        'Volume': put['volume'],
                        'Implied Volatility': put['impliedVolatility'],
                        'Delta': put['delta'],
                        'Gamma': put['gamma'],
                        'Theta': put['theta'],
                        'Signal Score': signal['score'],
                        'Profit Target': signal['profit_target'],
                        'Stop Loss': signal['stop_loss'],
                        'Holding Period': signal['holding_period'],
                        'Conditions': "; ".join(signal['passed_conditions']),
                        'Is 0DTE': put['is_0dte']
                    })
            
            call_signals_df = pd.DataFrame(call_signals)
            put_signals_df = pd.DataFrame(put_signals)
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Signals", "📈 Chart", "🔍 Technicals", "📅 0DTE Analytics"])
            
            with tab1:
                st.subheader("Buy Signals")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Call Signals")
                    if not call_signals_df.empty:
                        for _, signal in call_signals_df.iterrows():
                            with st.container():
                                st.markdown(f'<div class="signal-alert call-metric">', unsafe_allow_html=True)
                                st.markdown(f"**{signal['Contract']}** (Score: {signal['Signal Score']:.2%})")
                                st.markdown(f"Strike: ${signal['Strike']:.2f} | Expiry: {signal['Expiry']} | Moneyness: {signal['Moneyness']}")
                                st.markdown(f"Price: ${signal['Last Price']:.2f} | Volume: {signal['Volume']:.0f} | IV: {signal['Implied Volatility']:.2%}")
                                st.markdown(f"Delta: {signal['Delta']:.2f} | Gamma: {signal['Gamma']:.3f} | Theta: {signal['Theta']:.3f}")
                                st.markdown(f"Profit Target: ${signal['Profit Target']:.2f} | Stop Loss: ${signal['Stop Loss']:.2f}")
                                st.markdown(f"Holding: {signal['Holding Period']}")
                                st.markdown('<div class="tooltip">ℹ️ Details<span class="tooltiptext">' + signal['Conditions'] + '</span></div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No call signals found with current settings.")
                
                with col2:
                    st.markdown("### Put Signals")
                    if not put_signals_df.empty:
                        for _, signal in put_signals_df.iterrows():
                            with st.container():
                                st.markdown(f'<div class="signal-alert put-metric">', unsafe_allow_html=True)
                                st.markdown(f"**{signal['Contract']}** (Score: {signal['Signal Score']:.2%})")
                                st.markdown(f"Strike: ${signal['Strike']:.2f} | Expiry: {signal['Expiry']} | Moneyness: {signal['Moneyness']}")
                                st.markdown(f"Price: ${signal['Last Price']:.2f} | Volume: {signal['Volume']:.0f} | IV: {signal['Implied Volatility']:.2%}")
                                st.markdown(f"Delta: {signal['Delta']:.2f} | Gamma: {signal['Gamma']:.3f} | Theta: {signal['Theta']:.3f}")
                                st.markdown(f"Profit Target: ${signal['Profit Target']:.2f} | Stop Loss: ${signal['Stop Loss']:.2f}")
                                st.markdown(f"Holding: {signal['Holding Period']}")
                                st.markdown('<div class="tooltip">ℹ️ Details<span class="tooltiptext">' + signal['Conditions'] + '</span></div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No put signals found with current settings.")
            
            with tab2:
                st.subheader("Price Chart")
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df['Datetime'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="OHLC"
                    ))
                    if 'EMA_9' in df.columns and not df['EMA_9'].isna().all():
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df['EMA_9'], name="EMA 9", line=dict(color='blue')
                        ))
                    if 'EMA_20' in df.columns and not df['EMA_20'].isna().all():
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df['EMA_20'], name="EMA 20", line=dict(color='orange')
                        ))
                    if 'VWAP' in df.columns and not df['VWAP'].isna().all():
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df['VWAP'], name="VWAP", line=dict(color='purple', dash='dash')
                        ))
                    fig.update_layout(
                        title=f"{ticker} Price Action",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Technical Indicators")
                if not df.empty:
                    latest = df.iloc[-1]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI", f"{latest['RSI']:.1f}" if not pd.isna(latest['RSI']) else "N/A")
                        st.metric("Stochastic", f"{latest['Stochastic']:.1f}" if not pd.isna(latest['Stochastic']) else "N/A")
                    with col2:
                        st.metric("ATR %", f"{latest['ATR_pct']*100:.2f}%" if not pd.isna(latest['ATR_pct']) else "N/A")
                        st.metric("Price Momentum", f"{latest['price_momentum']*100:.2f}%" if not pd.isna(latest['price_momentum']) else "N/A")
                    with col3:
                        st.metric("Volume", f"{latest['Volume']:.0f}")
                        st.metric("Avg Volume", f"{latest['avg_vol']:.0f}" if not pd.isna(latest['avg_vol']) else "N/A")
                    
                    if not call_signals_df.empty or not put_signals_df.empty:
                        st.markdown("### Signal Summary")
                        if not call_signals_df.empty:
                            st.markdown("#### Call Signals")
                            st.dataframe(call_signals_df[['Contract', 'Strike', 'Expiry', 'Moneyness', 'Last Price', 'Signal Score']].style.format({
                                'Strike': "${:.2f}",
                                'Last Price': "${:.2f}",
                                'Signal Score': "{:.2%}"
                            }))
                        if not put_signals_df.empty:
                            st.markdown("#### Put Signals")
                            st.dataframe(put_signals_df[['Contract', 'Strike', 'Expiry', 'Moneyness', 'Last Price', 'Signal Score']].style.format({
                                'Strike': "${:.2f}",
                                'Last Price': "${:.2f}",
                                'Signal Score': "{:.2%}"
                            }))
            
            with tab4:
                st.subheader("0DTE Options Analytics")
                if st.session_state.show_0dte:
                    call_0dte = call_signals_df[call_signals_df['Is 0DTE']] if not call_signals_df.empty else pd.DataFrame()
                    put_0dte = put_signals_df[put_signals_df['Is 0DTE']] if not put_signals_df.empty else pd.DataFrame()
                    
                    if not call_0dte.empty or not put_0dte.empty:
                        st.markdown("### 0DTE Signal Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total 0DTE Call Signals", len(call_0dte))
                            if not call_0dte.empty:
                                st.metric("Avg Call IV", f"{call_0dte['Implied Volatility'].mean():.2%}")
                                st.metric("Max Call Premium", f"${call_0dte['Last Price'].max():.2f}")
                        with col2:
                            st.metric("Total 0DTE Put Signals", len(put_0dte))
                            if not put_0dte.empty:
                                st.metric("Avg Put IV", f"{put_0dte['Implied Volatility'].mean():.2%}")
                                st.metric("Max Put Premium", f"${put_0dte['Last Price'].max():.2f}")
                        
                        if not call_0dte.empty:
                            st.markdown("#### 0DTE Call Signals")
                            st.dataframe(call_0dte[['Contract', 'Strike', 'Expiry', 'Moneyness', 'Last Price', 'Signal Score']].style.format({
                                'Strike': "${:.2f}",
                                'Last Price': "${:.2f}",
                                'Signal Score': "{:.2%}"
                            }))
                        
                        if not put_0dte.empty:
                            st.markdown("#### 0DTE Put Signals")
                            st.dataframe(put_0dte[['Contract', 'Strike', 'Expiry', 'Moneyness', 'Last Price', 'Signal Score']].style.format({
                                'Strike': "${:.2f}",
                                'Last Price': "${:.2f}",
                                'Signal Score': "{:.2%}"
                            }))
                        
                        if not call_0dte.empty or not put_0dte.empty:
                            st.markdown("#### Premium Distribution")
                            fig = go.Figure()
                            if not call_0dte.empty:
                                fig.add_trace(go.Histogram(
                                    x=call_0dte['Last Price'],
                                    name="Call Premiums",
                                    marker_color='green',
                                    opacity=0.5
                                ))
                            if not put_0dte.empty:
                                fig.add_trace(go.Histogram(
                                    x=put_0dte['Last Price'],
                                    name="Put Premiums",
                                    marker_color='red',
                                    opacity=0.5
                                ))
                            fig.update_layout(
                                title="0DTE Options Premium Distribution",
                                xaxis_title="Premium ($)",
                                yaxis_title="Count",
                                barmode='overlay',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No 0DTE signals found with current settings.")
                else:
                    st.info("Enable 'Show 0DTE Options Only' in the sidebar to view 0DTE analytics.")
            
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please try refreshing or selecting a different ticker.")
else:
    st.info("Please select a ticker in the sidebar to begin analysis.")

# Run auto-refresh
manage_auto_refresh()
