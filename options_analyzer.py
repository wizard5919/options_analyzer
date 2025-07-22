import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import pytz
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import plotly.graph_objects as go
import uuid

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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

# Configuration constants
CONFIG = {
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 2,
    'RATE_LIMIT_COOLDOWN': 300,
    'MIN_DATA_POINTS': 20,
    'CACHE_TTL': 300,
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

# Signal thresholds
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
# UTILITY FUNCTIONS
# =============================
def get_market_state() -> str:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern).time()
    if datetime.datetime.now(eastern).weekday() >= 5:
        return "Closed"
    if CONFIG['PREMARKET_START'] <= now < CONFIG['MARKET_OPEN']:
        return "Premarket"
    elif CONFIG['MARKET_OPEN'] <= now <= CONFIG['MARKET_CLOSE']:
        return "Open"
    elif CONFIG['MARKET_CLOSE'] < now <= CONFIG['POSTMARKET_END']:
        return "Postmarket"
    return "Closed"

def get_dynamic_refresh_interval() -> int:
    market_state = get_market_state()
    if market_state in ["Premarket", "Open"]:
        return 60
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

def validate_ticker(ticker: str) -> bool:
    if not ticker:
        st.error("Ticker cannot be empty.")
        return False
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', auto_adjust=True, prepost=True)
        if not data.empty:
            return True
        st.error(f"No data available for ticker {ticker}. Try another ticker.")
        return False
    except Exception as e:
        st.error(f"Invalid ticker {ticker}: {str(e)}")
        return False

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data(ticker: str) -> pd.DataFrame:
    intervals = ["1m", "5m"] if get_market_state() in ["Premarket", "Open"] else ["5m", "15m"]
    lookback_days = 3 if get_market_state() == "Premarket" else 7
    df = pd.DataFrame()
    
    for attempt in range(CONFIG['MAX_RETRIES']):
        for interval in intervals:
            try:
                end = datetime.datetime.now()
                start = end - datetime.timedelta(days=lookback_days)
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    prepost=True
                )
                
                if data.empty:
                    continue
                    
                # Reset index and clean columns
                data = data.reset_index()
                if 'Datetime' not in data.columns and 'Date' in data.columns:
                    data = data.rename(columns={'Date': 'Datetime'})
                if 'Datetime' not in data.columns:
                    st.error("Could not find datetime index in data")
                    return pd.DataFrame()
                
                # Ensure datetime is proper type
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                
                # Timezone conversion
                eastern = pytz.timezone('US/Eastern')
                if data['Datetime'].dt.tz is None:
                    data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert(eastern)
                else:
                    data['Datetime'] = data['Datetime'].dt.tz_convert(eastern)
                
                # Filter and clean data
                required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                data = data[required_cols].dropna()
                
                if len(data) >= CONFIG['MIN_DATA_POINTS']:
                    return data
                    
            except Exception as e:
                if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                    time.sleep(CONFIG['RATE_LIMIT_COOLDOWN'])
                continue
    
    st.error(f"Failed to fetch valid stock data for {ticker} after {CONFIG['MAX_RETRIES']} attempts.")
    return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
        
    df = df.copy()
    
    # Handle missing values
    df = df.dropna(subset=['Close', 'High', 'Low'])
    
    # Get series for indicator calculations
    close_series = df['Close'].squeeze()
    high_series = df['High'].squeeze()
    low_series = df['Low'].squeeze()
    
    # Calculate technical indicators
    try:
        if len(df) >= 9:
            df['EMA_9'] = EMAIndicator(close=close_series, window=9).ema_indicator()
        if len(df) >= 20:
            df['EMA_20'] = EMAIndicator(close=close_series, window=20).ema_indicator()
        if len(df) >= 14:
            df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
            df['Stochastic'] = StochasticOscillator(
                high=high_series, low=low_series, close=close_series, window=14, smooth_window=3
            ).stoch()
            df['ATR'] = AverageTrueRange(
                high=high_series, low=low_series, close=close_series, window=14
            ).average_true_range()
            df['ATR_pct'] = df['ATR'] / close_series
    except Exception as e:
        st.warning(f"Indicator calculation warning: {str(e)}")
    
    # Calculate VWAP
    try:
        # Simplified VWAP calculation
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TPV'] = df['Typical_Price'] * df['Volume']
        df['Cumulative_TPV'] = df['TPV'].cumsum()
        df['Cumulative_Volume'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Volume']
    except Exception as e:
        st.warning(f"VWAP calculation warning: {str(e)}")
        df['VWAP'] = np.nan
    
    # Calculate volume averages
    try:
        df['avg_vol'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    except Exception as e:
        st.warning(f"Volume average calculation warning: {str(e)}")
        df['avg_vol'] = np.nan
    
    # Price momentum
    try:
        lookback = CONFIG['PRICE_ACTION']['lookback_periods']
        df['price_momentum'] = df['Close'].pct_change(periods=lookback)
        df['range_high'] = df['High'].rolling(window=lookback).max()
        df['range_low'] = df['Low'].rolling(window=lookback).min()
        
        # Fixed breakout calculations with proper alignment
        shifted_range_high = df['range_high'].shift(1)
        shifted_range_low = df['range_low'].shift(1)
        
        df['breakout_up'] = (df['Close'] > (shifted_range_high * 
                            (1 + CONFIG['PRICE_ACTION']['breakout_threshold'])))
        df['breakout_down'] = (df['Close'] < (shifted_range_low * 
                              (1 - CONFIG['PRICE_ACTION']['breakout_threshold'])))
    except Exception as e:
        st.warning(f"Price momentum calculation warning: {str(e)}")
    
    return df

def get_current_price(ticker: str) -> float:
    if not ticker:
        return 0.0
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> list[str]:
    if not ticker:
        return []
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        return list(expiries)[:3] if expiries else []
    except Exception:
        return []

def fetch_options_data(ticker: str, expiries: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ticker:
        return pd.DataFrame(), pd.DataFrame()
    
    all_calls = pd.DataFrame()
    all_puts = pd.DataFrame()
    stock = yf.Ticker(ticker)
    
    for expiry in expiries:
        try:
            chain = stock.option_chain(expiry)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            
            # Add required greek columns if missing
            for col in ['delta', 'gamma', 'theta']:
                if col not in calls:
                    calls[col] = np.nan
                if col not in puts:
                    puts[col] = np.nan
            
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
            time.sleep(0.5)  # Rate limit prevention
        except Exception:
            continue
    
    return all_calls, all_puts

def classify_moneyness(strike: float, spot: float) -> str:
    if spot <= 0:
        return "N/A"
    diff_pct = abs(strike - spot) / spot
    if diff_pct < 0.01:
        return 'ATM'
    elif strike < spot:
        return 'ITM' if diff_pct >= 0.03 else 'NTM'
    else:
        return 'OTM' if diff_pct >= 0.03 else 'NTM'

def calculate_approximate_greeks(option: pd.Series, spot_price: float) -> tuple[float, float, float]:
    if spot_price <= 0:
        return np.nan, np.nan, np.nan
        
    try:
        expiry_date = pd.to_datetime(option['expiry']).date()
        today = datetime.date.today()
        days_to_expiry = (expiry_date - today).days
        is_0dte = days_to_expiry == 0
        theta = -0.050 if is_0dte else -0.020

        moneyness = spot_price / option['strike']
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
        return delta, gamma, theta
    except Exception:
        return np.nan, np.nan, np.nan

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    required_fields = ['strike', 'lastPrice', 'volume', 'openInterest']
    for field in required_fields:
        if field not in option or pd.isna(option[field]):
            return False
            
    if option['lastPrice'] <= 0:
        return False
        
    # Ensure greeks exist
    for greek in ['delta', 'gamma', 'theta']:
        if greek not in option or pd.isna(option[greek]):
            delta, gamma, theta = calculate_approximate_greeks(option, spot_price)
            option['delta'] = delta
            option['gamma'] = gamma
            option['theta'] = theta
            
    return True

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> dict[str, float]:
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    
    # Get volatility from ATR if available
    volatility = stock_data.get('ATR_pct', 0.02)
    if pd.isna(volatility):
        volatility = 0.02
        
    # Get momentum
    price_momentum = stock_data.get('price_momentum', 0.0)
    if pd.isna(price_momentum):
        price_momentum = 0.0
        
    # Volatility factor
    vol_factor = 1 + (volatility * 100)
    
    # Time decay factor
    time_decay = calculate_time_decay_factor()
    
    # Adjust thresholds based on volatility and momentum
    if side == 'call':
        thresholds['delta_base'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_factor))
        thresholds['gamma_base'] = max(0.02, min(0.15, thresholds['gamma_base'] * vol_factor))
        thresholds['theta_base'] = min(0.1, thresholds['theta_base'] * time_decay)
        thresholds['rsi_min'] = max(40.0, min(70.0, 50 + (price_momentum * 500)))
        thresholds['stoch_base'] = max(50.0, min(80.0, 60 + (price_momentum * 500)))
        thresholds['volume_min'] = max(500.0, min(3000.0, thresholds['volume_min'] * vol_factor))
    else:
        thresholds['delta_base'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_factor))
        thresholds['gamma_base'] = max(0.02, min(0.15, thresholds['gamma_base'] * vol_factor))
        thresholds['theta_base'] = min(0.1, thresholds['theta_base'] * time_decay)
        thresholds['rsi_max'] = max(30.0, min(60.0, 50 + (price_momentum * 500)))
        thresholds['stoch_base'] = max(20.0, min(50.0, 40 + (price_momentum * 500)))
        thresholds['volume_min'] = max(500.0, min(3000.0, thresholds['volume_min'] * vol_factor))
    
    # Adjust for market conditions
    if is_premarket() or is_early_market():
        if side == 'call':
            thresholds['delta_base'] = max(0.35, thresholds['delta_base'])
        else:
            thresholds['delta_base'] = min(-0.35, thresholds['delta_base'])
        thresholds['volume_min'] = max(300.0, thresholds['volume_min'] * 0.5)
        thresholds['gamma_base'] = thresholds['gamma_base'] * 0.8

    if is_0dte:
        thresholds['volume_min'] = thresholds['volume_min'] * 0.7
        thresholds['gamma_base'] = thresholds['gamma_base'] * 0.7

    return thresholds

def calculate_holding_period(option: pd.Series, spot_price: float) -> str:
    try:
        expiry_date = pd.to_datetime(option['expiry']).date()
        days_to_expiry = (expiry_date - datetime.date.today()).days
        if days_to_expiry == 0:
            return "Intraday (Exit before 3:30 PM)"
        if option['contractSymbol'].startswith('C'):
            intrinsic_value = max(0, spot_price - option['strike'])
        else:
            intrinsic_value = max(0, option['strike'] - spot_price)
            
        if intrinsic_value > 0:
            return "1-3 days (Swing trade)"
        else:
            return "1 day (Gamma play)"
    except Exception:
        return "N/A"

def calculate_profit_targets(option: pd.Series) -> tuple[float, float]:
    try:
        entry_price = option['lastPrice']
        profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['call' if option['contractSymbol'].startswith('C') else 'put'])
        stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
        return profit_target, stop_loss
    except Exception:
        return np.nan, np.nan

def generate_signal(option: pd.Series, side: str, stock_data: pd.Series, is_0dte: bool) -> dict:
    if stock_data.empty:
        return {'signal': False, 'reason': 'No stock data available'}
    
    current_price = stock_data['Close'] if 'Close' in stock_data else 0
    if not validate_option_data(option, current_price):
        return {'signal': False, 'reason': 'Insufficient option data'}
    
    try:
        thresholds = calculate_dynamic_thresholds(stock_data, side, is_0dte)
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        option_volume = float(option['volume'])
        
        conditions = []
        if side == "call":
            conditions = [
                (delta >= thresholds['delta_base'], f"Delta ≥ {thresholds['delta_base']:.2f}"),
                (gamma >= thresholds['gamma_base'], f"Gamma ≥ {thresholds['gamma_base']:.3f}"),
                (theta <= -thresholds['theta_base'], f"Theta ≤ {-thresholds['theta_base']:.3f}"),
                (option_volume >= thresholds['volume_min'], f"Volume ≥ {thresholds['volume_min']:.0f}")
            ]
        else:
            conditions = [
                (delta <= thresholds['delta_base'], f"Delta ≤ {thresholds['delta_base']:.2f}"),
                (gamma >= thresholds['gamma_base'], f"Gamma ≥ {thresholds['gamma_base']:.3f}"),
                (theta <= -thresholds['theta_base'], f"Theta ≤ {-thresholds['theta_base']:.3f}"),
                (option_volume >= thresholds['volume_min'], f"Volume ≥ {thresholds['volume_min']:.0f}")
            ]
            
        # Add technical indicators if available
        if 'EMA_9' in stock_data and 'EMA_20' in stock_data:
            ema9 = stock_data['EMA_9']
            ema20 = stock_data['EMA_20']
            close = stock_data['Close']
            if side == "call":
                # Fixed chained comparison
                condition1 = close > ema9
                condition2 = ema9 > ema20
                conditions.append((condition1 and condition2, "Price > EMA9 > EMA20"))
            else:
                condition1 = close < ema9
                condition2 = ema9 < ema20
                conditions.append((condition1 and condition2, "Price < EMA9 < EMA20"))
                
        if 'RSI' in stock_data:
            rsi = stock_data['RSI']
            if side == "call":
                conditions.append((rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']:.1f}"))
            else:
                conditions.append((rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']:.1f}"))
                
        if 'Stochastic' in stock_data:
            stoch = stock_data['Stochastic']
            if side == "call":
                conditions.append((stoch > thresholds['stoch_base'], f"Stoch > {thresholds['stoch_base']:.1f}"))
            else:
                conditions.append((stoch < thresholds['stoch_base'], f"Stoch < {thresholds['stoch_base']:.1f}"))
                
        if 'VWAP' in stock_data:
            vwap = stock_data['VWAP']
            close = stock_data['Close']
            if side == "call":
                conditions.append((close > vwap, "Price > VWAP"))
            else:
                conditions.append((close < vwap, "Price < VWAP"))
                
        if 'price_momentum' in stock_data:
            momentum = stock_data['price_momentum']
            if side == "call":
                conditions.append((momentum >= thresholds['price_momentum_min'], f"Momentum ≥ {thresholds['price_momentum_min']*100:.2f}%"))
            else:
                conditions.append((momentum <= thresholds['price_momentum_min'], f"Momentum ≤ {thresholds['price_momentum_min']*100:.2f}%"))

        passed_conditions = [desc for condition, desc in conditions if condition]
        failed_conditions = [desc for condition, desc in conditions if not condition]
        signal_score = len(passed_conditions) / len(conditions) if conditions else 0

        profit_target = stop_loss = np.nan
        holding_period = "N/A"
        if passed_conditions:
            profit_target, stop_loss = calculate_profit_targets(option)
            holding_period = calculate_holding_period(option, current_price)

        return {
            'signal': bool(passed_conditions),
            'passed_conditions': passed_conditions,
            'failed_conditions': failed_conditions,
            'score': signal_score,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period
        }
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}'}

# =============================
# STREAMLIT INTERFACE
# =============================
def initialize_session_state():
    defaults = {
        'refresh_counter': 0,
        'last_refresh': time.time(),
        'ticker': "SPY",
        'strike_range': (-5.0, 5.0),
        'moneyness_filter': ["NTM", "ATM"],
        'show_welcome': True,
        'current_time': datetime.datetime.now(pytz.timezone('US/Eastern')),
        'refresh_interval': get_dynamic_refresh_interval(),
        'enable_auto_refresh': True,
        'show_0dte': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def manage_auto_refresh():
    if st.session_state.enable_auto_refresh:
        if time.time() - st.session_state.last_refresh >= st.session_state.refresh_interval:
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_counter += 1
            st.rerun()

def update_current_time():
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    if now.second != st.session_state.current_time.second:
        st.session_state.current_time = now
        if 'time_placeholder' in st.session_state:
            st.session_state.time_placeholder.metric(
                "Current Market Time",
                st.session_state.current_time.strftime('%H:%M:%S %Z')
            )

# Initialize session state
initialize_session_state()

# Welcome Message
if st.session_state.show_welcome:
    st.info("""
    Welcome to the Options Greeks Buy Signal Analyzer!
    
    1. Select a stock ticker
    2. Configure settings in the sidebar
    3. View real-time signals and charts
    """)
    st.session_state.show_welcome = False

# Display market state and live clock
market_state = get_market_state()
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
with col1:
    market_status = "💤 Market Closed"
    if market_state == "Open":
        market_status = "✅ Market Open"
    elif market_state == "Premarket":
        market_status = "⏰ Premarket"
    elif market_state == "Postmarket":
        market_status = "🌙 Postmarket"
    st.markdown(f"**{market_status}**")

with col2:
    current_price = get_current_price(st.session_state.ticker)
    st.metric("Current Price", f"${current_price:.2f}" if current_price > 0 else "N/A")

with col3:
    if 'time_placeholder' not in st.session_state:
        st.session_state.time_placeholder = st.empty()
    update_current_time()

with col4:
    if st.button("🔁 Refresh", key="manual_refresh"):
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.rerun()

st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    with st.expander("🔄 Auto-Refresh", expanded=True):
        st.session_state.enable_auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.enable_auto_refresh,
            key="auto_refresh"
        )
        if st.session_state.enable_auto_refresh:
            st.session_state.refresh_interval = st.selectbox(
                "Refresh Interval",
                options=[60, 120, 300],
                index=[60, 120, 300].index(st.session_state.refresh_interval),
                format_func=lambda x: f"{x} seconds"
            )
            st.info(f"Refreshing every {st.session_state.refresh_interval} seconds")
    
    with st.expander("📊 Stock Selection", expanded=True):
        ticker_options = ["SPY", "QQQ", "AAPL", "IWM", "TSLA", "GLD", "TLT", "Other"]
        selected_ticker = st.selectbox(
            "Select Stock Ticker",
            ticker_options,
            index=ticker_options.index(st.session_state.ticker) if st.session_state.ticker in ticker_options else 0
        )
        
        if selected_ticker == "Other":
            custom_ticker = st.text_input(
                "Enter Custom Ticker",
                value=st.session_state.ticker if st.session_state.ticker not in ticker_options else "",
                key="custom_ticker"
            ).upper()
            if custom_ticker:
                st.session_state.ticker = custom_ticker
        else:
            st.session_state.ticker = selected_ticker
        
        if st.button("Validate Ticker"):
            if validate_ticker(st.session_state.ticker):
                st.success(f"Valid ticker: {st.session_state.ticker}")
            else:
                st.error(f"Invalid ticker: {st.session_state.ticker}")
    
    with st.expander("🔍 Filters", expanded=True):
        if current_price > 0:
            st.session_state.strike_range = st.slider(
                "Strike Price Range (% from current)",
                -20.0, 20.0, st.session_state.strike_range, 0.5
            )
            st.session_state.moneyness_filter = st.multiselect(
                "Moneyness Filter",
                ["ITM", "NTM", "ATM", "OTM"],
                default=st.session_state.moneyness_filter
            )
            st.session_state.show_0dte = st.checkbox(
                "Show 0DTE Options Only",
                value=st.session_state.show_0dte
            )
        else:
            st.warning("Cannot configure filters without valid price")

# Main Content
if st.session_state.ticker and validate_ticker(st.session_state.ticker):
    try:
        with st.spinner("Fetching and analyzing data..."):
            # Get stock data
            df = get_stock_data(st.session_state.ticker)
            if df.empty:
                st.error("No stock data available. Try a different ticker or refresh later.")
                st.stop()
                
            df = compute_indicators(df)
            latest_data = df.iloc[-1] if not df.empty else pd.Series()
            
            # Get options data
            expiries = get_options_expiries(st.session_state.ticker)
            if not expiries:
                st.warning("No options data available for this ticker.")
                st.stop()
                
            calls, puts = fetch_options_data(st.session_state.ticker, expiries)
            if calls.empty and puts.empty:
                st.warning("No options data retrieved. Try refreshing or selecting a different ticker.")
                st.stop()
                
            # Filter options
            strike_min = current_price * (1 + st.session_state.strike_range[0] / 100)
            strike_max = current_price * (1 + st.session_state.strike_range[1] / 100)
            
            calls = calls[(calls['strike'] >= strike_min) & (calls['strike'] <= strike_max)]
            puts = puts[(puts['strike'] >= strike_min) & (puts['strike'] <= strike_max)]
            
            if st.session_state.moneyness_filter:
                calls['moneyness'] = calls['strike'].apply(lambda x: classify_moneyness(x, current_price))
                puts['moneyness'] = puts['strike'].apply(lambda x: classify_moneyness(x, current_price))
                calls = calls[calls['moneyness'].isin(st.session_state.moneyness_filter)]
                puts = puts[puts['moneyness'].isin(st.session_state.moneyness_filter)]
                
            if st.session_state.show_0dte:
                today = datetime.date.today()
                calls['is_0dte'] = calls['expiry'].apply(lambda x: (pd.to_datetime(x).date() - today).days == 0)
                puts['is_0dte'] = puts['expiry'].apply(lambda x: (pd.to_datetime(x).date() - today).days == 0)
                calls = calls[calls['is_0dte']]
                puts = puts[puts['is_0dte']]
            
            # Generate signals
            call_signals = []
            for _, row in calls.iterrows():
                signal = generate_signal(row, "call", latest_data, st.session_state.show_0dte)
                if signal['signal']:
                    call_signals.append({
                        'Contract': row['contractSymbol'],
                        'Strike': row['strike'],
                        'Expiry': row['expiry'],
                        'Price': row['lastPrice'],
                        'Delta': row.get('delta', np.nan),
                        'Gamma': row.get('gamma', np.nan),
                        'Theta': row.get('theta', np.nan),
                        'IV': row.get('impliedVolatility', np.nan),
                        'Volume': row['volume'],
                        'Score': signal['score'],
                        'Profit Target': signal['profit_target'],
                        'Stop Loss': signal['stop_loss'],
                        'Holding': signal['holding_period'],
                        'Conditions': ", ".join(signal['passed_conditions'])
                    })
                    
            put_signals = []
            for _, row in puts.iterrows():
                signal = generate_signal(row, "put", latest_data, st.session_state.show_0dte)
                if signal['signal']:
                    put_signals.append({
                        'Contract': row['contractSymbol'],
                        'Strike': row['strike'],
                        'Expiry': row['expiry'],
                        'Price': row['lastPrice'],
                        'Delta': row.get('delta', np.nan),
                        'Gamma': row.get('gamma', np.nan),
                        'Theta': row.get('theta', np.nan),
                        'IV': row.get('impliedVolatility', np.nan),
                        'Volume': row['volume'],
                        'Score': signal['score'],
                        'Profit Target': signal['profit_target'],
                        'Stop Loss': signal['stop_loss'],
                        'Holding': signal['holding_period'],
                        'Conditions': ", ".join(signal['passed_conditions'])
                    })
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Chart", "⚙️ Analysis"])
            
            with tab1:
                st.subheader("Buy Signals")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Call Signals")
                    if call_signals:
                        for signal in call_signals:
                            with st.container():
                                st.markdown(f'<div class="signal-alert call-metric">', unsafe_allow_html=True)
                                st.markdown(f"**{signal['Contract']}** (Score: {signal['Score']:.0%})")
                                st.markdown(f"**${signal['Strike']:.2f}** | {signal['Expiry']}")
                                st.markdown(f"Price: ${signal['Price']:.2f} | Δ: {signal['Delta']:.2f} | Γ: {signal['Gamma']:.3f}")
                                st.markdown(f"IV: {signal['IV']:.0%} | Volume: {signal['Volume']:.0f}")
                                st.markdown(f"Target: ${signal['Profit Target']:.2f} | Stop: ${signal['Stop Loss']:.2f}")
                                st.markdown(f"Holding: {signal['Holding']}")
                                st.caption(signal['Conditions'])
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No call signals found")
                
                with col2:
                    st.markdown("### Put Signals")
                    if put_signals:
                        for signal in put_signals:
                            with st.container():
                                st.markdown(f'<div class="signal-alert put-metric">', unsafe_allow_html=True)
                                st.markdown(f"**{signal['Contract']}** (Score: {signal['Score']:.0%})")
                                st.markdown(f"**${signal['Strike']:.2f}** | {signal['Expiry']}")
                                st.markdown(f"Price: ${signal['Price']:.2f} | Δ: {signal['Delta']:.2f} | Γ: {signal['Gamma']:.3f}")
                                st.markdown(f"IV: {signal['IV']:.0%} | Volume: {signal['Volume']:.0f}")
                                st.markdown(f"Target: ${signal['Profit Target']:.2f} | Stop: ${signal['Stop Loss']:.2f}")
                                st.markdown(f"Holding: {signal['Holding']}")
                                st.caption(signal['Conditions'])
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No put signals found")
            
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
                        name="Price"
                    ))
                    
                    # Add indicators if available
                    if 'EMA_9' in df and not df['EMA_9'].isnull().all():
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df['EMA_9'], name="EMA 9", line=dict(color='blue', width=1)
                        ))
                    if 'EMA_20' in df and not df['EMA_20'].isnull().all():
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df['EMA_20'], name="EMA 20", line=dict(color='orange', width=1)
                        ))
                    if 'VWAP' in df and not df['VWAP'].isnull().all():
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df['VWAP'], name="VWAP", line=dict(color='purple', width=1, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f"{st.session_state.ticker} Price Action",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        height=500,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for chart")
            
            with tab3:
                st.subheader("Technical Analysis")
                if not df.empty:
                    latest = df.iloc[-1]
                    cols = st.columns(4)
                    metrics = [
                        ('RSI', 'RSI', '{:.1f}'),
                        ('Stochastic', 'Stochastic', '{:.1f}'),
                        ('ATR %', 'ATR_pct', '{:.2%}'),
                        ('Momentum', 'price_momentum', '{:.2%}'),
                        ('Volume', 'Volume', '{:.0f}'),
                        ('Avg Volume', 'avg_vol', '{:.0f}'),
                        ('Close', 'Close', '${:.2f}'),
                        ('VWAP', 'VWAP', '${:.2f}')
                    ]
                    
                    for i, (name, key, fmt) in enumerate(metrics):
                        with cols[i % 4]:
                            value = latest.get(key, None)
                            if value is not None and not pd.isna(value):
                                st.metric(name, fmt.format(value))
                            else:
                                st.metric(name, "N/A")
                else:
                    st.warning("No technical data available")
                
                st.subheader("Signal Summary")
                if call_signals or put_signals:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Call Signals", len(call_signals))
                    with col2:
                        st.metric("Total Put Signals", len(put_signals))
                else:
                    st.info("No signals to display")
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()
else:
    st.info("Please select a valid ticker to begin analysis")

# Run auto-refresh
manage_auto_refresh()
