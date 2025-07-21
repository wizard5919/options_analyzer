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
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# Suppress future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Apply custom CSS for visual polish and clarity
st.markdown("""
<style>
.stMetric { background-color: #f0f2f6; border-radius: 5px; padding: 10px; margin-bottom: 10px; }
.call-metric { border-left: 4px solid #28a745; }
.put-metric { border-left: 4px solid #dc3545; }
.stButton>button { background-color: #007bff; color: white; border-radius: 5px; padding: 8px 16px; }
.stSidebar { background-color: #e9ecef !important; }
@media (prefers-color-scheme: dark) {
    .stSidebar { background-color: #343a40 !important; }
    .stMetric { color: #ffffff !important; } /* White text in dark mode */
    .stMetric .stMetric-value { color: #ffffff !important; } /* Ensure dark mode value visibility */
}
.sidebar .stMetric { color: #000000 !important; } /* Force black text in sidebar (light mode) */
.sidebar .stMetric .stMetric-value { color: #000000 !important; } /* Target metric value specifically */
.sidebar .stMetric .stMetric-label + * { color: #000000 !important; } /* Target the value after the label */
.sidebar .stSelectbox, .sidebar .stTextInput { background-color: #ffffff; border-radius: 5px; }
.signal-table th { background-color: #007bff; color: white; }
.tooltip { position: relative; display: inline-block; cursor: pointer; margin-left: 5px; }
.tooltip .tooltiptext { visibility: hidden; width: 200px; background-color: #555; color: #fff; 
    text-align: center; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; 
    bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
body { font-size: 14px; line-height: 1.5; }
.stApp { zoom: 1; }
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
    'CACHE_TTL': 300,  # 5 minutes
    'RATE_LIMIT_COOLDOWN': 180,  # 3 minutes
    'MARKET_OPEN': datetime.time(9, 30),
    'MARKET_CLOSE': datetime.time(16, 0),
    'PREMARKET_START': datetime.time(4, 0),
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

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_base': 0.5,
        'delta_vol_multiplier': 0.15,
        'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.03,
        'theta_base': 0.05,
        'rsi_base': 50,
        'rsi_min': 50,
        'rsi_max': 70,
        'stoch_base': 60,
        'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.4,
        'volume_min': 1000,
        'price_momentum_min': 0.005
    },
    'put': {
        'delta_base': -0.5,
        'delta_vol_multiplier': 0.15,
        'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.03,
        'theta_base': 0.05,
        'rsi_base': 50,
        'rsi_min': 30,
        'rsi_max': 50,
        'stoch_base': 40,
        'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.4,
        'volume_min': 1000,
        'price_momentum_min': -0.005
    }
}

# =============================
# AUTO-REFRESH SYSTEM
# =============================
class AutoRefreshSystem:
    def __init__(self):
        self.running = False
        self.thread = None
        self.refresh_interval = 60
        
    def start(self, interval):
        if self.running and interval == self.refresh_interval:
            return
        self.stop()
        self.running = True
        self.refresh_interval = interval
        
        def refresh_loop():
            while self.running:
                time.sleep(interval)
                if self.running:
                    st.rerun()
        
        self.thread = threading.Thread(target=refresh_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

# =============================
# UTILITY FUNCTIONS
# =============================
def is_market_open() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    now_time = now.time()
    if now.weekday() >= 5:
        return False
    return CONFIG['MARKET_OPEN'] <= now_time <= CONFIG['MARKET_CLOSE']

def is_premarket() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    now_time = now.time()
    if now.weekday() >= 5:
        return False
    return CONFIG['PREMARKET_START'] <= now_time < CONFIG['MARKET_OPEN']

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

def get_current_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return data['Close'].iloc[-1]
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

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=10)
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="5m",
            auto_adjust=True,
            progress=False,
            prepost=True
        )
        if data.empty:
            st.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        data = data.dropna(how='all')
        for col in required_cols:
            if col in data.columns:
                if hasattr(data[col].iloc[0], '__len__') and not isinstance(data[col].iloc[0], str):
                    data[col] = data[col].apply(lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x)
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna(subset=required_cols)
        if len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient data points ({len(data)}). Need at least {CONFIG['MIN_DATA_POINTS']}.")
            return pd.DataFrame()
        
        eastern = pytz.timezone('US/Eastern')
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc)
        data.index = data.index.tz_convert(eastern)
        data['premarket'] = False
        premarket_mask = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        data.loc[premarket_mask, 'premarket'] = True
        return data.reset_index(drop=False)
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['avg_vol'] = np.nan
    for date, group in df.groupby(df['Datetime'].dt.date):
        regular = group[~group['premarket']]
        if not regular.empty:
            regular_avg_vol = regular['Volume'].expanding(min_periods=1).mean()
            df.loc[regular.index, 'avg_vol'] = regular_avg_vol
        premarket = group[group['premarket']]
        if not premarket.empty:
            premarket_avg_vol = premarket['Volume'].expanding(min_periods=1).mean()
            df.loc[premarket.index, 'avg_vol'] = premarket_avg_vol
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

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        df = df.copy()
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        if df.empty:
            return df
        
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        
        if len(close) >= 9:
            ema_9 = EMAIndicator(close=close, window=9)
            df['EMA_9'] = ema_9.ema_indicator()
        else:
            df['EMA_9'] = np.nan
            
        if len(close) >= 20:
            ema_20 = EMAIndicator(close=close, window=20)
            df['EMA_20'] = ema_20.ema_indicator()
        else:
            df['EMA_20'] = np.nan
            
        if len(close) >= 14:
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
        else:
            df['RSI'] = np.nan
            
        if len(close) >= 14:
            stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
            df['Stochastic'] = stoch.stoch()
        else:
            df['Stochastic'] = np.nan
            
        df['VWAP'] = np.nan
        for session, group in df.groupby(pd.Grouper(key='Datetime', freq='D')):
            if group.empty:
                continue
            premarket = group[group['premarket']]
            regular = group[~group['premarket']]
            if not regular.empty:
                typical_price = (regular['High'] + regular['Low'] + regular['Close']) / 3
                vwap_cumsum = (regular['Volume'] * typical_price).cumsum()
                volume_cumsum = regular['Volume'].cumsum()
                regular_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                df.loc[regular.index, 'VWAP'] = regular_vwap
            if not premarket.empty:
                prev_day = session - datetime.timedelta(days=1)
                prev_close = df[df['Datetime'].dt.date == prev_day.date()]['Close'].iloc[-1] if not df[df['Datetime'].dt.date == prev_day.date()].empty else premarket['Close'].iloc[0]
                typical_price = (premarket['High'] + premarket['Low'] + premarket['Close']) / 3
                vwap_cumsum = (premarket['Volume'] * typical_price).cumsum()
                volume_cumsum = premarket['Volume'].cumsum()
                premarket_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                df.loc[premarket.index, 'VWAP'] = premarket_vwap
                
        if len(close) >= 14:
            atr = AverageTrueRange(high=high, low=low, close=close, window=14)
            df['ATR'] = atr.average_true_range()
            df['ATR_pct'] = df['ATR'] / close
        else:
            df['ATR'] = np.nan
            df['ATR_pct'] = np.nan
            
        df = calculate_volume_averages(df)
        df = compute_price_action(df)
        return df
    except Exception as e:
        st.error(f"Error in compute_indicators: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> List[str]:
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

def fetch_options_data(ticker: str, expiries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def calculate_approximate_greeks(option: dict, spot_price: float) -> Tuple[float, float, float]:
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

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    volatility = stock_data.get('ATR_pct', 0.02)
    price_momentum = stock_data.get('price_momentum', 0.0)
    breakout_up = stock_data.get('breakout_up', False)
    breakout_down = stock_data.get('breakout_down', False)
    time_decay = calculate_time_decay_factor()
    
    vol_factor = 1 + (volatility * 100)
    momentum_factor = 1 + abs(price_momentum) * 50
    
    if side == 'call':
        thresholds['delta_base'] = max(0.3, min(0.8, 0.5 * vol_factor * momentum_factor))
        if breakout_up:
            thresholds['delta_base'] = min(thresholds['delta_base'] * 1.2, 0.85)
        thresholds['delta_min'] = thresholds['delta_base']
    else:
        thresholds['delta_base'] = min(-0.3, max(-0.8, -0.5 * vol_factor * momentum_factor))
        if breakout_down:
            thresholds['delta_base'] = max(thresholds['delta_base'] * 0.8, -0.85)
        thresholds['delta_max'] = thresholds['delta_base']
    
    thresholds['gamma_base'] = max(0.02, min(0.15, 0.05 * vol_factor))
    thresholds['theta_base'] = min(0.1, 0.05 * time_decay)
    
    if side == 'call':
        thresholds['rsi_base'] = max(40, min(60, 50 + (price_momentum * 1000)))
        thresholds['rsi_min'] = thresholds['rsi_base']
        thresholds['stoch_base'] = max(50, min(80, 60 + (price_momentum * 1000)))
    else:
        thresholds['rsi_base'] = max(40, min(60, 50 - abs(price_momentum * 1000)))
        thresholds['rsi_max'] = thresholds['rsi_base']
        thresholds['stoch_base'] = max(20, min(50, 40 - abs(price_momentum * 1000)))
    
    thresholds['volume_min'] = max(500, min(3000, 1000 * vol_factor))
    thresholds['volume_multiplier'] = max(0.8, min(2.5, thresholds['volume_multiplier_base'] * vol_factor))
    
    thresholds['price_momentum_min'] = 0.005 * vol_factor if side == 'call' else -0.005 * vol_factor
    
    if is_premarket() or is_early_market():
        if side == 'call':
            thresholds['delta_min'] = 0.35
        else:
            thresholds['delta_max'] = -0.35
        thresholds['volume_multiplier'] *= 0.6
        thresholds['gamma_base'] *= 0.8
        thresholds['volume_min'] = max(300, thresholds['volume_min'] * 0.5)
    
    if is_0dte:
        thresholds['volume_multiplier'] *= 0.7
        thresholds['gamma_base'] *= 0.7
        if side == 'call':
            thresholds['delta_min'] = max(0.4, thresholds['delta_min'])
        else:
            thresholds['delta_max'] = min(-0.4, thresholds['delta_max'])
    
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

def calculate_profit_targets(option: pd.Series) -> Tuple[float, float]:
    entry_price = option['lastPrice'] * calculate_time_decay_factor()
    profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['call' if option['contractSymbol'].startswith('C') else 'put'])
    stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
    return profit_target, stop_loss

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
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
                (delta >= thresholds['delta_min'], f"Delta >= {thresholds['delta_min']:.2f}", delta),
                (gamma >= thresholds['gamma_base'], f"Gamma >= {thresholds['gamma_base']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']:.1f}", rsi),
                (stochastic is not None and stochastic > thresholds['stoch_base'], f"Stochastic > {thresholds['stoch_base']:.1f}", stochastic),
                (vwap is not None and close > vwap, "Price > VWAP", f"{close:.2f} > {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}"),
                (price_momentum >= thresholds['price_momentum_min'] or breakout_up, f"Price Momentum >= {thresholds['price_momentum_min']*100:.2f}% or Breakout", f"{price_momentum*100:.2f}%")
            ]
        else:
            volume_ok = option_volume > thresholds['volume_min']
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']:.2f}", delta),
                (gamma >= thresholds['gamma_base'], f"Gamma >= {thresholds['gamma_base']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']:.1f}", rsi),
                (stochastic is not None and stochastic < thresholds['stoch_base'], f"Stochastic < {thresholds['stoch_base']:.1f}", stochastic),
                (vwap is not None and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}"),
                (price_momentum <= thresholds['price_momentum_min'] or breakout_down, f"Price Momentum <= {thresholds['price_momentum_min']*100:.2f}% or Breakout", f"{price_momentum*100:.2f}%")
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
if 'refresh_system' not in st.session_state:
    st.session_state.refresh_system = AutoRefreshSystem()
if 'ticker' not in st.session_state:
    st.session_state.ticker = "IWM"
if 'strike_range' not in st.session_state:
    st.session_state.strike_range = (-5.0, 5.0)
if 'moneyness_filter' not in st.session_state:
    st.session_state.moneyness_filter = ["NTM", "ATM"]

if 'rate_limited_until' in st.session_state:
    if time.time() < st.session_state['rate_limited_until']:
        remaining = int(st.session_state['rate_limited_until'] - time.time())
        st.warning(f"Yahoo Finance API rate limited. Please wait {remaining} seconds.")
        with st.expander("ℹ️ About Rate Limiting"):
            st.markdown("""
            Yahoo Finance may restrict data retrieval frequency. If rate limited, please:
            - Wait a few minutes before refreshing
            - Avoid auto-refresh intervals below 1 minute
            - Use one ticker at a time
            """)
        st.stop()
    else:
        del st.session_state['rate_limited_until']

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Auto-adjusted for market conditions** with swift signal detection")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Sidebar is active")  # Debug line
    with st.expander("🔄 Auto-Refresh Settings", expanded=True):
        enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=True, key="auto_refresh")
        if enable_auto_refresh:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                options=[60, 120, 300],
                index=1,
                format_func=lambda x: f"{x} seconds",
                key="refresh_interval"
            )
            st.session_state.refresh_system.start(refresh_interval)
            st.info(f"Data will refresh every {refresh_interval} seconds")
        else:
            st.session_state.refresh_system.stop()
    
    with st.expander("📊 Stock Selection", expanded=True):
        ticker_options = ["SPY", "QQQ", "AAPL", "IWM", "TSLA", "GLD", "TLT", "Other"]
        selected_ticker = st.selectbox("Select or Enter Stock Ticker", ticker_options, key="ticker_select")
        if selected_ticker == "Other":
            ticker = st.text_input("Enter Custom Ticker:", value=st.session_state.ticker, key="custom_ticker").upper()
        else:
            ticker = selected_ticker.upper()
        if ticker:
            try:
                ticker_info = yf.Ticker(ticker).info
                st.success(f"Valid ticker: {ticker} ({ticker_info.get('shortName', ticker)})")
                st.session_state.ticker = ticker
            except:
                st.error("Invalid ticker. Please try again (e.g., SPY, AAPL).")
                ticker = ""
    
    if ticker:
        df = get_stock_data(ticker)
        if not df.empty:
            df = compute_indicators(df)
            latest = df.iloc[-1]
            call_thresholds = calculate_dynamic_thresholds(latest, "call", is_0dte=False)
            put_thresholds = calculate_dynamic_thresholds(latest, "put", is_0dte=False)
            
            with st.expander("📈 Auto-Adjusted Signal Thresholds", expanded=True):
                st.write("Thresholds are dynamically set based on market conditions")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="call-metric">', unsafe_allow_html=True)
                    st.write("**Calls**")
                    st.metric("Base Delta", call_thresholds['delta_base'], help="Minimum delta for call signals, adjusted for volatility")
                    st.metric("Base Gamma", call_thresholds['gamma_base'], help="Minimum gamma for call signals, sensitive to price movements")
                    st.metric("Base RSI", call_thresholds['rsi_base'], help="Base RSI level for calls, reflecting momentum")
                    st.metric("Min RSI", call_thresholds['rsi_min'], help="Minimum RSI for call signals")
                    st.metric("Stochastic", call_thresholds['stoch_base'], help="Minimum stochastic oscillator value for calls")
                    st.metric("Min Volume", call_thresholds['volume_min'], help="Minimum option volume for valid signals")
                    st.metric("Min Price Momentum (%)", f"{call_thresholds['price_momentum_min']*100:.2f}", help="Minimum price change for call signals")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="put-metric">', unsafe_allow_html=True)
                    st.write("**Puts**")
                    st.metric("Base Delta", put_thresholds['delta_base'], help="Maximum delta for put signals, adjusted for volatility")
                    st.metric("Base Gamma", put_thresholds['gamma_base'], help="Minimum gamma for put signals, sensitive to price movements")
                    st.metric("Base RSI", put_thresholds['rsi_base'], help="Base RSI level for puts, reflecting momentum")
                    st.metric("Max RSI", put_thresholds['rsi_max'], help="Maximum RSI for put signals")
                    st.metric("Stochastic", put_thresholds['stoch_base'], help="Maximum stochastic oscillator value for puts")
                    st.metric("Min Volume", put_thresholds['volume_min'], help="Minimum option volume for valid signals")
                    st.metric("Min Price Momentum (%)", f"{put_thresholds['price_momentum_min']*100:.2f}", help="Minimum price change for put signals")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.write("**Common**")
                st.metric("Max Theta", call_thresholds['theta_base'], help="Maximum theta for signals, accounting for time decay")
                st.metric("Volume Multiplier", call_thresholds['volume_multiplier'], help="Multiplier for volume thresholds based on volatility")
    
    with st.expander("🎯 Profit Targets"):
        CONFIG['PROFIT_TARGETS']['call'] = st.slider("Call Profit Target (%)", 0.05, 0.50, 0.15, 0.01, key="call_profit_target")
        CONFIG['PROFIT_TARGETS']['put'] = st.slider("Put Profit Target (%)", 0.05, 0.50, 0.15, 0.01, key="put_profit_target")
        CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01, key="stop_loss")
    
    with st.expander("📈 Dynamic Threshold Parameters"):
        st.write("Sensitivities are fixed but can be adjusted if needed")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Call Sensitivities**")
            SIGNAL_THRESHOLDS['call']['delta_vol_multiplier'] = st.slider("Delta Vol Sensitivity (Calls)", 0.0, 0.5, 0.15, 0.01, key="call_delta_vol")
            SIGNAL_THRESHOLDS['call']['gamma_vol_multiplier'] = st.slider("Gamma Vol Sensitivity (Calls)", 0.0, 0.5, 0.03, 0.01, key="call_gamma_vol")
        with col2:
            st.write("**Put Sensitivities**")
            SIGNAL_THRESHOLDS['put']['delta_vol_multiplier'] = st.slider("Delta Vol Sensitivity (Puts)", 0.0, 0.5, 0.15, 0.01, key="put_delta_vol")
            SIGNAL_THRESHOLDS['put']['gamma_vol_multiplier'] = st.slider("Gamma Vol Sensitivity (Puts)", 0.0, 0.5, 0.03, 0.01, key="put_gamma_vol")
        st.write("**Volume Sensitivity**")
        SIGNAL_THRESHOLDS['call']['volume_vol_multiplier'] = SIGNAL_THRESHOLDS['put']['volume_vol_multiplier'] = st.slider(
            "Volume Vol Multiplier", 0.0, 1.0, 0.4, 0.05, key="volume_vol_multiplier")
        if st.button("Preview Thresholds", key="preview_thresholds"):
            if 'df' in locals() and not df.empty:
                latest = df.iloc[-1]
                preview_call = calculate_dynamic_thresholds(latest, "call", is_0dte=False)
                preview_put = calculate_dynamic_thresholds(latest, "put", is_0dte=False)
                st.json({'call': preview_call, 'put': preview_put})

# Main interface
if ticker:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if is_market_open():
            st.success("✅ Market is OPEN")
            time_decay = calculate_time_decay_factor()
            st.info(f"⏰ Time Decay Factor: {time_decay:.2f}x")
        elif is_premarket():
            st.warning("⏰ PREMARKET Session")
        else:
            st.info("💤 Market is CLOSED")
    with col2:
        current_price = get_current_price(ticker)
        st.metric("Current Price", f"${current_price:.2f}")
    with col3:
        if 'last_refresh' in st.session_state:
            last_update = datetime.datetime.fromtimestamp(st.session_state.last_refresh)
            st.caption(f"📅 Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("📅 Last updated: Never")
    with col4:
        manual_refresh = st.button("🔁 Refresh Now", key="manual_refresh")
    
    if manual_refresh:
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.rerun()
    
    st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Stock Data", "⚙️ Analysis Details"])
    
    with tab1:
        try:
            with st.spinner("Fetching and analyzing data..."):
                if df.empty:
                    st.error("Unable to fetch stock data. Please check the ticker symbol.")
                    st.stop()
                
                df = compute_indicators(df)
                if df.empty:
                    st.error("Unable to compute technical indicators.")
                    st.stop()
                
                current_price = df.iloc[-1]['Close']
                st.success(f"✅ **{ticker}** - Current Price: **${current_price:.2f}**")
                
                atr_pct = df.iloc[-1].get('ATR_pct', 0)
                volatility_status = "Low"
                if not pd.isna(atr_pct):
                    if atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['high']:
                        volatility_status = "Extreme"
                    elif atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['medium']:
                        volatility_status = "High"
                    elif atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['low']:
                        volatility_status = "Medium"
                    st.info(f"📈 Current Volatility (ATR%): {atr_pct*100:.2f}% - **{volatility_status}**")
                
                st.subheader("🧠 Diagnostic Information")
                if is_premarket():
                    st.warning("⚠️ PREMARKET CONDITIONS: Volume requirements relaxed, delta thresholds adjusted")
                elif is_early_market():
                    st.warning("⚠️ EARLY MARKET CONDITIONS: Volume requirements relaxed, delta thresholds adjusted")
                
                st.write("📏 Current Signal Thresholds:")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"**Calls:** Δ ≥ {call_thresholds['delta_base']:.2f} | "
                              f"Γ ≥ {call_thresholds['gamma_base']:.3f} | "
                              f"Vol > {call_thresholds['volume_min']:.0f} | "
                              f"Momentum ≥ {call_thresholds['price_momentum_min']*100:.2f}%")
                with col2:
                    st.caption(f"**Puts:** Δ ≤ {put_thresholds['delta_base']:.2f} | "
                              f"Γ ≥ {put_thresholds['gamma_base']:.3f} | "
                              f"Vol > {put_thresholds['volume_min']:.0f} | "
                              f"Momentum ≤ {put_thresholds['price_momentum_min']*100:.2f}%")
                
                expiries = get_options_expiries(ticker)
                if not expiries:
                    st.error("No options expiries available. Please wait due to rate limits.")
                    st.stop()
                
                expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"], index=0)
                today = datetime.date.today()
                if expiry_mode == "0DTE Only":
                    expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
                else:
                    expiries_to_use = expiries[:3]
                
                if not expiries_to_use:
                    st.warning("No options expiries available for the selected mode.")
                    st.stop()
                
                st.info(f"Analyzing {len(expiries_to_use)} expiries: {', '.join(expiries_to_use)}")
                
                calls, puts = fetch_options_data(ticker, expiries_to_use)
                if calls.empty and puts.empty:
                    st.error("No options data available. Possible network issue or invalid ticker.")
                    st.stop()
                
                for option_df in [calls, puts]:
                    option_df['is_0dte'] = option_df['expiry'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today)
                
                max_range = max(10, current_price * 0.1) if current_price > 0 else 10
                strike_range = st.slider(
                    "Strike Range Around Current Price ($):",
                    -max_range, max_range, st.session_state.strike_range, 1.0,
                    key="strike_range"
                )
                min_strike = current_price + strike_range[0]
                max_strike = current_price + strike_range[1]
                
                calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
                puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
                
                if not calls_filtered.empty:
                    calls_filtered['moneyness'] = calls_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                    invalid_calls = calls_filtered[~calls_filtered.apply(lambda x: validate_option_data(x, current_price), axis=1)]
                    if not invalid_calls.empty:
                        st.warning(f"Skipped {len(invalid_calls)} call options due to insufficient data (e.g., missing Greeks, low volume).")
                if not puts_filtered.empty:
                    puts_filtered['moneyness'] = puts_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                    invalid_puts = puts_filtered[~puts_filtered.apply(lambda x: validate_option_data(x, current_price), axis=1)]
                    if not invalid_puts.empty:
                        st.warning(f"Skipped {len(invalid_puts)} put options due to insufficient data (e.g., missing Greeks, low volume).")
                
                # Use a separate variable to avoid session state modification after widget instantiation
                default_moneyness = st.session_state.moneyness_filter
                moneyness_filter = st.multiselect(
                    "Filter by Moneyness:",
                    options=["ITM", "NTM", "ATM", "OTM"],
                    default=default_moneyness,
                    key="moneyness_filter"
                )
                # Update session state only if the selection changes
                if moneyness_filter != st.session_state.moneyness_filter:
                    st.session_state.moneyness_filter = moneyness_filter
                
                if not calls_filtered.empty:
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(moneyness_filter)]
                if not puts_filtered.empty:
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(moneyness_filter)]
                
                st.write(f"🔍 Filtered Options: {len(calls_filtered)} calls, {len(puts_filtered)} puts "
                         f"(Strike range: ${min_strike:.2f}-${max_strike:.2f})")
                
                sort_by = st.selectbox("Sort Signals By:", ["signal_score", "strike", "lastPrice", "volume"], key="sort_signals")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Call Option Signals")
                    if not calls_filtered.empty:
                        call_signals = []
                        for _, row in calls_filtered.iterrows():
                            is_0dte = row.get('is_0dte', False)
                            signal_result = generate_signal(row, "call", df, is_0dte)
                            if signal_result['signal']:
                                row_dict = row.to_dict()
                                row_dict['signal_score'] = signal_result['score']
                                row_dict['thresholds'] = signal_result['thresholds']
                                row_dict['passed_conditions'] = signal_result['passed_conditions']
                                row_dict['is_0dte'] = is_0dte
                                row_dict['profit_target'] = signal_result['profit_target']
                                row_dict['stop_loss'] = signal_result['stop_loss']
                                row_dict['holding_period'] = signal_result['holding_period']
                                call_signals.append(row_dict)
                        
                        if call_signals:
                            signals_df = pd.DataFrame(call_signals)
                            signals_df = signals_df.sort_values(sort_by, ascending=False)
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta',
                                           'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            st.dataframe(signals_df[available_cols].round(4), use_container_width=True, hide_index=True)
                            
                            top_signal = signals_df.iloc[0]
                            st.success(
                                f"🚨 **CALL SIGNAL DETECTED!** "
                                f"Contract: {top_signal['contractSymbol']} | "
                                f"Strike: ${top_signal['strike']:.2f} | "
                                f"Price: ${top_signal['lastPrice']:.2f} | "
                                f"Holding: {top_signal['holding_period']}",
                                icon="✅"
                            )
                            
                            st.markdown("""
                            <script>
                            var audio = new Audio('https://www.soundjay.com/buttons/beep-01a.mp3');
                            audio.play();
                            </script>
                            """, unsafe_allow_html=True)
                            
                            st.write("**Signal Strength**")
                            st.markdown("""
                            <chartjs
                                type="doughnut"
                                data='{
                                    "labels": ["Signal Strength", "Remaining"],
                                    "datasets": [{
                                        "data": [""" + str(top_signal['signal_score'] * 100) + """, """ + str((1 - top_signal['signal_score']) * 100) + """],
                                        "backgroundColor": ["#28a745", "#e9ecef"],
                                        "borderWidth": 1
                                    }]
                                }'
                                options='{
                                    "circumference": 180,
                                    "rotation": -90,
                                    "cutout": "70%",
                                    "plugins": {
                                        "legend": { "display": false },
                                        "title": { "display": true, "text": "Top Call Signal Strength" }
                                    }
                                }'
                            ></chartjs>
                            """, unsafe_allow_html=True)
                            
                            if top_signal['thresholds']:
                                th = top_signal['thresholds']
                                st.info(f"Applied Thresholds: Δ ≥ {th['delta_min']:.2f} | "
                                        f"Γ ≥ {th['gamma_base']:.3f} | Θ ≤ {th['theta_base']:.3f} | "
                                        f"RSI > {th['rsi_min']:.1f} | Stoch > {th['stoch_base']:.1f} | "
                                        f"Vol > {th['volume_min']:.0f}")
                            
                            with st.expander("View Conditions for Top Signal"):
                                if top_signal['passed_conditions']:
                                    st.write("✅ Passed Conditions:")
                                    for condition in top_signal['passed_conditions']:
                                        st.write(f"- {condition}")
                                else:
                                    st.info("No conditions passed")
                            
                            st.success(f"Found {len(call_signals)} call signals!")
                        else:
                            st.info("No call signals found matching criteria.")
                            if not calls_filtered.empty:
                                sample_call = calls_filtered.iloc[0]
                                is_0dte = sample_call.get('is_0dte', False)
                                result = generate_signal(sample_call, "call", df, is_0dte)
                                if result and 'failed_conditions' in result:
                                    st.write("Top call option failed conditions:")
                                    for condition in result['failed_conditions']:
                                        st.write(f"- {condition}")
                    else:
                        st.info("No call options available for selected filters.")
                
                with col2:
                    st.subheader("📉 Put Option Signals")
                    if not puts_filtered.empty:
                        put_signals = []
                        for _, row in puts_filtered.iterrows():
                            is_0dte = row.get('is_0dte', False)
                            signal_result = generate_signal(row, "put", df, is_0dte)
                            if signal_result['signal']:
                                row_dict = row.to_dict()
                                row_dict['signal_score'] = signal_result['score']
                                row_dict['thresholds'] = signal_result['thresholds']
                                row_dict['passed_conditions'] = signal_result['passed_conditions']
                                row_dict['is_0dte'] = is_0dte
                                row_dict['profit_target'] = signal_result['profit_target']
                                row_dict['stop_loss'] = signal_result['stop_loss']
                                row_dict['holding_period'] = signal_result['holding_period']
                                put_signals.append(row_dict)
                        
                        if put_signals:
                            signals_df = pd.DataFrame(put_signals)
                            signals_df = signals_df.sort_values(sort_by, ascending=False)
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta',
                                           'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            st.dataframe(signals_df[available_cols].round(4), use_container_width=True, hide_index=True)
                            
                            top_signal = signals_df.iloc[0]
                            st.error(
                                f"🚨 **PUT SIGNAL DETECTED!** "
                                f"Contract: {top_signal['contractSymbol']} | "
                                f"Strike: ${top_signal['strike']:.2f} | "
                                f"Price: ${top_signal['lastPrice']:.2f} | "
                                f"Holding: {top_signal['holding_period']}",
                                icon="✅"
                            )
                            
                            st.markdown("""
                            <script>
                            var audio = new Audio('https://www.soundjay.com/buttons/beep-01a.mp3');
                            audio.play();
                            </script>
                            """, unsafe_allow_html=True)
                            
                            st.write("**Signal Strength**")
                            st.markdown("""
                            <chartjs
                                type="doughnut"
                                data='{
                                    "labels": ["Signal Strength", "Remaining"],
                                    "datasets": [{
                                        "data": [""" + str(top_signal['signal_score'] * 100) + """, """ + str((1 - top_signal['signal_score']) * 100) + """],
                                        "backgroundColor": ["#dc3545", "#e9ecef"],
                                        "borderWidth": 1
                                    }]
                                }'
                                options='{
                                    "circumference": 180,
                                    "rotation": -90,
                                    "cutout": "70%",
                                    "plugins": {
                                        "legend": { "display": false },
                                        "title": { "display": true, "text": "Top Put Signal Strength" }
                                    }
                                }'
                            ></chartjs>
                            """, unsafe_allow_html=True)
                            
                            if top_signal['thresholds']:
                                th = top_signal['thresholds']
                                st.info(f"Applied Thresholds: Δ ≤ {th['delta_max']:.2f} | "
                                        f"Γ ≥ {th['gamma_base']:.3f} | Θ ≤ {th['theta_base']:.3f} | "
                                        f"RSI < {th['rsi_max']:.1f} | Stoch < {th['stoch_base']:.1f} | "
                                        f"Vol > {th['volume_min']:.0f}")
                            
                            with st.expander("View Conditions for Top Signal"):
                                if top_signal['passed_conditions']:
                                    st.write("✅ Passed Conditions:")
                                    for condition in top_signal['passed_conditions']:
                                        st.write(f"- {condition}")
                                else:
                                    st.info("No conditions passed")
                            
                            st.success(f"Found {len(put_signals)} put signals!")
                        else:
                            st.info("No put signals found matching criteria.")
                            if not puts_filtered.empty:
                                sample_put = puts_filtered.iloc[0]
                                is_0dte = sample_put.get('is_0dte', False)
                                result = generate_signal(sample_put, "put", df, is_0dte)
                                if result and 'failed_conditions' in result:
                                    st.write("Top put option failed conditions:")
                                    for condition in result['failed_conditions']:
                                        st.write(f"- {condition}")
                    else:
                        st.info("No put options available for selected filters.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please refresh the page and try again.")
    
    with tab2:
        if 'df' in locals() and not df.empty:
            st.subheader("📊 Stock Data & Indicators")
            if is_premarket():
                st.info("🔔 Currently showing premarket data")
            elif not is_market_open():
                st.info("🔔 Showing after-hours data")
            
            latest = df.iloc[-1]
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Current Price", f"${latest['Close']:.2f}")
            with col2:
                ema_9 = latest['EMA_9']
                st.metric("EMA 9", f"${ema_9:.2f}" if not pd.isna(ema_9) else "N/A")
            with col3:
                ema_20 = latest['EMA_20']
                st.metric("EMA 20", f"${ema_20:.2f}" if not pd.isna(ema_20) else "N/A")
            with col4:
                rsi = latest['RSI']
                st.metric("RSI", f"{rsi:.1f}" if not pd.isna(rsi) else "N/A")
            with col5:
                stochastic = latest['Stochastic']
                st.metric("Stochastic", f"{stochastic:.1f}" if not pd.isna(stochastic) else "N/A")
            with col6:
                atr_pct = latest['ATR_pct']
                st.metric("Volatility (ATR%)", f"{atr_pct*100:.2f}%" if not pd.isna(atr_pct) else "N/A")
            
            st.subheader("Recent Data")
            display_df = df.tail(10)[['Close', 'EMA_9', 'EMA_20', 'RSI', 'Stochastic', 'VWAP', 'ATR_pct', 'Volume', 'avg_vol', 'price_momentum']].round(2)
            display_df['ATR_pct'] = display_df['ATR_pct'] * 100
            display_df['price_momentum'] = display_df['price_momentum'] * 100
            display_df['Volume Ratio'] = display_df['Volume'] / display_df['avg_vol']
            st.dataframe(display_df.rename(columns={
                'ATR_pct': 'ATR%',
                'avg_vol': 'Avg Vol',
                'price_momentum': 'Momentum%'
            }), use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Analysis Details")
        if enable_auto_refresh:
            st.info(f"🔄 Auto-refresh enabled: Every {refresh_interval} seconds")
        else:
            st.info("🔄 Auto-refresh disabled")
        
        if 'calls_filtered' in locals() and not calls_filtered.empty:
            st.write("**Sample Call Analysis:**")
            sample_call = calls_filtered.iloc[0]
            if 'df' in locals():
                is_0dte = sample_call.get('is_0dte', False)
                result = generate_signal(sample_call, "call", df, is_0dte)
                st.json(result)
        
        st.write("**Current Signal Thresholds:**")
        st.json({'call': call_thresholds, 'put': put_thresholds})
        st.write("**Profit Targets:**")
        st.json(CONFIG['PROFIT_TARGETS'])
        st.write("**System Configuration:**")
        st.json(CONFIG)
    
    with st.expander("ℹ️ About Rate Limiting"):
        st.markdown("""
        Yahoo Finance may restrict data retrieval frequency. If rate limited, please:
        - Wait a few minutes before refreshing
        - Avoid auto-refresh intervals below 1 minute
        - Use one ticker at a time
        """)
else:
    st.info("Please select or enter a stock ticker to begin analysis.")
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Steps to analyze options:**
        1. Select a stock ticker (e.g., SPY, IWM, TSLA) or enter a custom one
        2. Configure auto-refresh settings in the sidebar
        3. Select expiration filter (0DTE or near-term)
        4. Adjust strike range around current price
        5. Filter by moneyness (ITM, ATM, OTM)
        6. Sort signals by score, strike, price, or volume
        7. Receive immediate signal notifications with strength gauges and audible alerts
        
        **Key Features:**
        - **Auto-Adjusted Thresholds:** Adapt to volatility and momentum
        - **Swift Signal Detection:** Immediate alerts with audible notifications
        - **Price Action Analysis:** Incorporates momentum and breakouts
        - **Time Decay Adjustment:** Accounts for theta during market hours
        - **Seamless Auto-Refresh:** Updates at chosen intervals
        - **Profit Targets & Exit Strategy:** Clear targets and holding periods
        """)
