import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import pytz
import math
import threading
from typing import Optional, Tuple, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, KeltnerChannel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from polygon import RESTClient  # For real-time data

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
    'CACHE_TTL': 300,  # 5 minutes
    'RATE_LIMIT_COOLDOWN': 180,  # 3 minutes
    'MARKET_OPEN': datetime.time(9, 30),  # 9:30 AM Eastern
    'MARKET_CLOSE': datetime.time(16, 0),  # 4:00 PM Eastern
    'PREMARKET_START': datetime.time(4, 0),  # 4:00 AM Eastern,
    'VOLATILITY_THRESHOLDS': {
        'low': 0.015,
        'medium': 0.03,
        'high': 0.05
    },
    'PROFIT_TARGETS': {
        'call': 0.15,  # 15% profit target
        'put': 0.15,   # 15% profit target
        'stop_loss': 0.08  # 8% stop loss
    },
    'TRADING_HOURS_PER_DAY': 6.5  # For time decay approximation
}

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_base': 0.5,
        'delta_vol_multiplier': 0.1,
        'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.02,
        'theta_base': 0.05,
        'rsi_base': 50,
        'rsi_min': 50,
        'rsi_max': 50,
        'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.3,
        'volume_min': 1000  # Minimum volume threshold
    },
    'put': {
        'delta_base': -0.5,
        'delta_vol_multiplier': 0.1,
        'gamma_base': 0.05,
        'gamma_vol_multiplier': 0.02,
        'theta_base': 0.05,
        'rsi_base': 50,
        'rsi_min': 50,
        'rsi_max': 50,
        'volume_multiplier_base': 1.0,
        'volume_vol_multiplier': 0.3,
        'volume_min': 1000  # Minimum volume threshold
    }
}

# =============================
# AUTO-REFRESH SYSTEM
# =============================

class AutoRefreshSystem:
    def __init__(self):
        self.running = False
        self.thread = None
        self.refresh_interval = 60  # Default interval
        
    def start(self, interval):
        if self.running and interval == self.refresh_interval:
            return  # Already running with same interval
        
        self.stop()  # Stop any existing thread
        self.running = True
        self.refresh_interval = interval
        
        def refresh_loop():
            while self.running:
                time.sleep(interval)
                if self.running:  # Double-check after sleep
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
    """Check if market is currently open based on Eastern Time"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    now_time = now.time()
    
    # Check if weekday (Monday=0, Sunday=6)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check time
    return CONFIG['MARKET_OPEN'] <= now_time <= CONFIG['MARKET_CLOSE']

def is_premarket() -> bool:
    """Check if we're in premarket hours"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    now_time = now.time()
    
    # Only consider weekdays
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    return CONFIG['PREMARKET_START'] <= now_time < CONFIG['MARKET_OPEN']

def is_early_market() -> bool:
    """Check if we're in the first 30 minutes of market open"""
    if not is_market_open():
        return False
    
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open_today = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'])
    market_open_today = eastern.localize(market_open_today)
    
    return (now - market_open_today).total_seconds() < 1800  # First 30 minutes

def calculate_remaining_trading_hours() -> float:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    close_time = datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE'])
    close_time = eastern.localize(close_time)
    
    if now >= close_time:
        return 0.0
    
    return (close_time - now).total_seconds() / 3600

def get_current_price(ticker: str, polygon_api_key: str = None) -> float:
    """Get the most current price including premarket, using Polygon if key provided"""
    if polygon_api_key:
        try:
            client = RESTClient(polygon_api_key)
            quote = client.get_last_quote(ticker)
            return (quote.ask_price + quote.bid_price) / 2 if quote.ask_price and quote.bid_price else 0.0
        except Exception as e:
            st.warning(f"Polygon API error: {str(e)}. Falling back to Yahoo Finance.")
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return data['Close'].iloc[-1]
        return 0.0
    except Exception as e:
        st.error(f"Error getting current price: {str(e)}")
        return 0.0

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    """Safely call API functions with retry logic"""
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

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch stock data with caching, error handling, and premarket support"""
    try:
        # Determine time range
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=10)
        
        data = yf.download(
            ticker, 
            start=start, 
            end=end, 
            interval="5m",
            auto_adjust=True,
            progress=False,
            prepost=True  # Include pre-market and after-hours data
        )

        if data.empty:
            st.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()

        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure we have required columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Clean and validate data
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
        
        data['premarket'] = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        
        return data.reset_index(drop=False)
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate more accurate volume averages with separate premarket handling"""
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

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators with comprehensive error handling"""
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
            
        if len(close) >= 50:
            ema_50 = EMAIndicator(close=close, window=50)
            df['EMA_50'] = ema_50.ema_indicator()
        else:
            df['EMA_50'] = np.nan
            
        if len(close) >= 200:
            ema_200 = EMAIndicator(close=close, window=200)
            df['EMA_200'] = ema_200.ema_indicator()
        else:
            df['EMA_200'] = np.nan
            
        if len(close) >= 14:
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
        else:
            df['RSI'] = np.nan

        df['VWAP'] = np.nan
        df['avg_vol'] = np.nan
        
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
        
        # Add MACD and Keltner Channels
        if len(close) >= 26:
            macd = MACD(close=close)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
            
            kc = KeltnerChannel(high=high, low=low, close=close)
            df['KC_upper'] = kc.keltner_channel_hband()
            df['KC_middle'] = kc.keltner_channel_mband()
            df['KC_lower'] = kc.keltner_channel_lband()
        else:
            df['MACD'] = df['MACD_signal'] = df['MACD_hist'] = np.nan
            df['KC_upper'] = df['KC_middle'] = df['KC_lower'] = np.nan
            
        df = calculate_volume_averages(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error in compute_indicators: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> List[str]:
    """Get options expiries with error handling and rate limit detection"""
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        return list(expiries) if expiries else []
    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
            st.warning("Yahoo Finance rate limit reached. Please wait a few minutes before retrying.")
            st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
        else:
            st.error(f"Error fetching expiries: {error_msg}")
        return []

def fetch_options_data(ticker: str, expiries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch options data with comprehensive error handling and delays"""
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
            
            time.sleep(1)  # 1-second delay between each expiry fetch
            
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                st.warning("Yahoo Finance rate limit reached. Please wait a few minutes before retrying.")
                st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
                break
            st.warning(f"Failed to fetch options for {expiry}: {error_msg}")
            failed_expiries.append(expiry)
            continue
    
    if failed_expiries:
        st.info(f"Failed to fetch data for expiries: {failed_expiries}")
    
    return all_calls, all_puts

def classify_moneyness(strike: float, spot: float) -> str:
    """Classify option moneyness with dynamic ranges"""
    diff = abs(strike - spot)
    diff_pct = diff / spot
    
    if diff_pct < 0.01:  # Within 1%
        return 'ATM'
    elif strike < spot:  # Below current price
        if diff_pct < 0.03:  # 1-3% below
            return 'NTM'  # Near-the-money
        else:
            return 'ITM'
    else:  # Above current price
        if diff_pct < 0.03:  # 1-3% above
            return 'NTM'  # Near-the-money
        else:
            return 'OTM'

def calculate_approximate_greeks(option: dict, spot_price: float) -> Tuple[float, float, float]:
    """Calculate approximate Greeks using simple formulas"""
    moneyness = spot_price / option['strike']
    
    if 'C' in option['contractSymbol']:
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
    
    theta = 0.05 if datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date() == datetime.date.today() else 0.02
    
    return delta, gamma, theta

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    """Validate that option has required data for analysis"""
    required_fields = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
    
    for field in required_fields:
        if field not in option or pd.isna(option[field]):
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
    """Calculate dynamic thresholds with enhanced volatility response"""
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    
    volatility = stock_data.get('ATR_pct', 0.02)
    
    vol_multiplier = 1 + (volatility * 100)
    
    if side == 'call':
        thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_multiplier))
    else:
        thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_multiplier))
    
    thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + thresholds['gamma_vol_multiplier'] * (volatility * 200))
    
    thresholds['volume_multiplier'] = max(0.8, min(2.5, thresholds['volume_multiplier_base'] * (1 + thresholds['volume_vol_multiplier'] * (volatility * 150))))
    
    if is_premarket() or is_early_market():
        if side == 'call':
            thresholds['delta_min'] = 0.35
        else:
            thresholds['delta_max'] = -0.35
        thresholds['volume_multiplier'] *= 0.6
        thresholds['gamma_min'] *= 0.8
    
    if is_0dte:
        thresholds['volume_multiplier'] *= 0.7
        thresholds['gamma_min'] *= 0.7
        if side == 'call':
            thresholds['delta_min'] = max(0.4, thresholds['delta_min'])
        else:
            thresholds['delta_max'] = min(-0.4, thresholds['delta_max'])
    
    return thresholds

def calculate_holding_period(option: pd.Series, spot_price: float) -> str:
    """Determine optimal holding period based on option characteristics"""
    expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - datetime.date.today()).days
    
    if days_to_expiry == 0:
        return "Intraday (Exit before 3:30 PM)"
    
    if 'C' in option['contractSymbol']:
        intrinsic_value = max(0, spot_price - option['strike'])
    else:
        intrinsic_value = max(0, option['strike'] - spot_price)
    
    if intrinsic_value > 0:
        if option['theta'] < -0.1:
            return "1-2 days (Scalp quickly)"
        else:
            return "3-5 days (Swing trade)"
    else:
        if days_to_expiry <= 3:
            return "1 day (Gamma play)"
        else:
            return "3-7 days (Wait for move)"

def calculate_profit_targets(option: pd.Series) -> Tuple[float, float]:
    """Calculate profit targets and stop loss levels"""
    entry_price = option['lastPrice']
    profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['call' if 'C' in option['contractSymbol'] else 'put'])
    stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
    return profit_target, stop_loss

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    """Generate trading signal with detailed analysis using dynamic thresholds"""
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
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume
        
        conditions = []
        
        if side == "call":
            volume_ok = option_volume > thresholds['volume_min']
            
            conditions = [
                (delta >= thresholds['delta_min'], f"Delta >= {thresholds['delta_min']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']:.1f}", rsi),
                (vwap is not None and close > vwap, "Price > VWAP", f"{close:.2f} > {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}")
            ]
        else:
            volume_ok = option_volume > thresholds['volume_min']
            
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']:.1f}", rsi),
                (vwap is not None and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}")
            ]
        
        passed_conditions = [desc for passed, desc, val in conditions if passed]
        failed_conditions = [f"{desc} (got {val})" for passed, desc, val in conditions if not passed]
        
        signal = all(passed for passed, desc, val in conditions)
        
        profit_target = None
        stop_loss = None
        holding_period = None
        est_hourly_decay = 0.0
        est_remaining_decay = 0.0
        
        if signal:
            profit_target, stop_loss = calculate_profit_targets(option)
            holding_period = calculate_holding_period(option, current_price)
            if is_0dte and theta:
                est_hourly_decay = -theta / CONFIG['TRADING_HOURS_PER_DAY']  # Theta is negative, decay is positive loss
                remaining_hours = calculate_remaining_trading_hours()
                est_remaining_decay = est_hourly_decay * remaining_hours
        
        return {
            'signal': signal,
            'passed_conditions': passed_conditions,
            'failed_conditions': failed_conditions,
            'score': len(passed_conditions) / len(conditions),
            'thresholds': thresholds,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period,
            'est_hourly_decay': est_hourly_decay,
            'est_remaining_decay': est_remaining_decay
        }
        
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}'}

def calculate_scanner_score(stock_df: pd.DataFrame, side: str) -> float:
    """Calculate a score for call/put scanner based on technical indicators"""
    if stock_df.empty:
        return 0.0
    
    latest = stock_df.iloc[-1]
    
    score = 0.0
    max_score = 5.0  # Five conditions
    
    try:
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        ema_50 = float(latest['EMA_50']) if not pd.isna(latest['EMA_50']) else None
        ema_200 = float(latest['EMA_200']) if not pd.isna(latest['EMA_200']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
        macd_signal = float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None
        keltner_upper = float(latest['KC_upper']) if not pd.isna(latest['KC_upper']) else None
        keltner_lower = float(latest['KC_lower']) if not pd.isna(latest['KC_lower']) else None
        
        if side == "call":
            if ema_9 and ema_20 and close > ema_9 > ema_20:
                score += 1.0
            if ema_50 and ema_200 and ema_50 > ema_200:
                score += 1.0
            if rsi and rsi > 50:
                score += 1.0
            if macd and macd_signal and macd > macd_signal:
                score += 1.0
            if keltner_upper and close > keltner_upper:
                score += 1.0
        else:
            if ema_9 and ema_20 and close < ema_9 < ema_20:
                score += 1.0
            if ema_50 and ema_200 and ema_50 < ema_200:
                score += 1.0
            if rsi and rsi < 50:
                score += 1.0
            if macd and macd_signal and macd < macd_signal:
                score += 1.0
            if keltner_lower and close < keltner_lower:
                score += 1.0
        
        return (score / max_score) * 100
    except Exception as e:
        st.error(f"Error in scanner score calculation: {str(e)}")
        return 0.0

def create_stock_chart(df: pd.DataFrame):
    """Create TradingView-style chart with indicators using Plotly"""
    if df.empty:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_9'], name='EMA 9', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_20'], name='EMA 20', line=dict(color='orange')), row=1, col=1)
    
    # Keltner Channels
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_upper'], name='KC Upper', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_middle'], name='KC Middle', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_lower'], name='KC Lower', line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df['Datetime'], y=df['Volume'], name='Volume', marker_color='gray'),
        row=1, col=1, secondary_y=True
    )
    
    # RSI
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Bar(x=df['Datetime'], y=df['MACD_hist'], name='Histogram', marker_color='gray'), row=3, col=1)
    
    fig.update_layout(
        height=800,
        title='Stock Price Chart with Indicators',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template='plotly_dark'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

# =============================
# STREAMLIT INTERFACE
# =============================

# Initialize session state
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'refresh_system' not in st.session_state:
    st.session_state.refresh_system = AutoRefreshSystem()

# Rate limit check
if 'rate_limited_until' in st.session_state:
    if time.time() < st.session_state['rate_limited_until']:
        remaining = int(st.session_state['rate_limited_until'] - time.time())
        st.warning(f"Yahoo Finance API rate limited. Please wait {remaining} seconds before retrying.")
        with st.expander("ℹ️ About Rate Limiting"):
            st.markdown("""
            Yahoo Finance may restrict how often data can be retrieved. If you see a "rate limited" warning, please:
            - Wait a few minutes before refreshing again
            - Avoid setting auto-refresh intervals lower than 1 minute
            - Use the app with one ticker at a time to reduce load
            """)
        st.stop()
    else:
        del st.session_state['rate_limited_until']

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Enhanced for volatile markets** with improved signal detection during price moves")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    polygon_api_key = st.text_input("Polygon API Key (for real-time data)", type="password", help="Optional: For live prices including extended hours")
    
    st.subheader("🔄 Auto-Refresh Settings")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    
    if enable_auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[60, 120, 300],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
        st.session_state.refresh_system.start(refresh_interval)
        st.info(f"Data will refresh every {refresh_interval} seconds")
    else:
        st.session_state.refresh_system.stop()
    
    st.subheader("Base Signal Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Calls**")
        SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider("Base Delta", 0.1, 1.0, 0.5, 0.1)
        SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider("Base Gamma", 0.01, 0.2, 0.05, 0.01)
        SIGNAL_THRESHOLDS['call']['rsi_base'] = st.slider("Base RSI", 30, 70, 50, 5)
        SIGNAL_THRESHOLDS['call']['rsi_min'] = st.slider("Min RSI", 30, 70, 50, 5)
        SIGNAL_THRESHOLDS['call']['volume_min'] = st.slider("Min Volume", 100, 5000, 1000, 100)
    
    with col2:
        st.write("**Puts**")
        SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider("Base Delta ", -1.0, -0.1, -0.5, 0.1)
        SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider("Base Gamma ", 0.01, 0.2, 0.05, 0.01)
        SIGNAL_THRESHOLDS['put']['rsi_base'] = st.slider("Base RSI ", 30, 70, 50, 5)
        SIGNAL_THRESHOLDS['put']['rsi_max'] = st.slider("Max RSI", 30, 70, 50, 5)
        SIGNAL_THRESHOLDS['put']['volume_min'] = st.slider("Min Volume ", 100, 5000, 1000, 100)
    
    SIGNAL_THRESHOLDS['call']['theta_base'] = SIGNAL_THRESHOLDS['put']['theta_base'] = st.slider("Max Theta", 0.01, 0.1, 0.05, 0.01)
    SIGNAL_THRESHOLDS['call']['volume_multiplier_base'] = SIGNAL_THRESHOLDS['put']['volume_multiplier_base'] = st.slider("Volume Multiplier", 1.0, 3.0, 1.0, 0.1)
    
    st.subheader("🎯 Profit Targets")
    CONFIG['PROFIT_TARGETS']['call'] = st.slider("Call Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
    CONFIG['PROFIT_TARGETS']['put'] = st.slider("Put Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
    CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01)
    
    st.subheader("📈 Dynamic Threshold Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Call Sensitivities**")
        SIGNAL_THRESHOLDS['call']['delta_vol_multiplier'] = st.slider("Delta Vol Sensitivity", 0.0, 0.5, 0.1, 0.01)
        SIGNAL_THRESHOLDS['call']['gamma_vol_multiplier'] = st.slider("Gamma Vol Sensitivity", 0.0, 0.5, 0.02, 0.01)
        
    with col2:
        st.write("**Put Sensitivities**")
        SIGNAL_THRESHOLDS['put']['delta_vol_multiplier'] = st.slider("Delta Vol Sensitivity ", 0.0, 0.5, 0.1, 0.01)
        SIGNAL_THRESHOLDS['put']['gamma_vol_multiplier'] = st.slider("Gamma Vol Sensitivity ", 0.0, 0.5, 0.02, 0.01)
    
    SIGNAL_THRESHOLDS['call']['volume_vol_multiplier'] = SIGNAL_THRESHOLDS['put']['volume_vol_multiplier'] = st.slider("Volume Vol Multiplier", 0.0, 1.0, 0.3, 0.05)

# Main interface
ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper()

if ticker:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if is_market_open():
            st.success("✅ Market is OPEN")
        elif is_premarket():
            st.warning("⏰ PREMARKET Session")
        else:
            st.info("💤 Market is CLOSED")
    
    with col2:
        current_price = get_current_price(ticker, polygon_api_key)
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

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Signals", "📈 Stock Data & Chart", "⚙️ Analysis Details", "📰 News & Events"])
    
    with tab1:
        try:
            with st.spinner("Fetching and analyzing data..."):
                df = get_stock_data(ticker)
                
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
                    st.caption(f"**Calls:** Δ ≥ {SIGNAL_THRESHOLDS['call']['delta_base']:.2f} | "
                              f"Γ ≥ {SIGNAL_THRESHOLDS['call']['gamma_base']:.3f} | "
                              f"Vol > {SIGNAL_THRESHOLDS['call']['volume_min']}")
                with col2:
                    st.caption(f"**Puts:** Δ ≤ {SIGNAL_THRESHOLDS['put']['delta_base']:.2f} | "
                              f"Γ ≥ {SIGNAL_THRESHOLDS['put']['gamma_base']:.3f} | "
                              f"Vol > {SIGNAL_THRESHOLDS['put']['volume_min']}")
                
                expiries = get_options_expiries(ticker)
                
                if not expiries:
                    st.error("No options expiries available for this ticker. If you recently refreshed, please wait due to Yahoo Finance rate limits.")
                    st.stop()
                
                expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"], index=1)
                
                today = datetime.date.today()
                if expiry_mode == "0DTE Only":
                    expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
                else:
                    expiries_to_use = expiries[:5]
                
                if not expiries_to_use:
                    st.warning("No options expiries available for the selected mode.")
                    st.stop()
                
                st.info(f"Analyzing {len(expiries_to_use)} expiries: {', '.join(expiries_to_use)}")
                
                calls, puts = fetch_options_data(ticker, expiries_to_use)
                
                if calls.empty and puts.empty:
                    st.error("No options data available.")
                    st.stop()
                
                for option_df in [calls, puts]:
                    option_df['is_0dte'] = option_df['expiry'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today)
                
                strike_range = st.slider("Strike Range Around Current Price ($):", -50, 50, (-5, 5), 1)
                min_strike = current_price + strike_range[0]
                max_strike = current_price + strike_range[1]
                
                calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
                puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
                
                if not calls_filtered.empty:
                    calls_filtered['moneyness'] = calls_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                if not puts_filtered.empty:
                    puts_filtered['moneyness'] = puts_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                
                m_filter = st.multiselect("Filter by Moneyness:", options=["ITM", "NTM", "ATM", "OTM"], default=["ITM", "NTM", "ATM"])
                
                if not calls_filtered.empty:
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
                if not puts_filtered.empty:
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]
                
                st.write(f"🔍 Filtered Options: {len(calls_filtered)} calls, {len(puts_filtered)} puts "
                         f"(Strike range: ${min_strike:.2f}-${max_strike:.2f})")
                
                col1, col2 = st.columns(2)
                
                call_signals = []
                put_signals = []
                
                with col1:
                    st.subheader("📈 Call Option Signals")
                    if not calls_filtered.empty:
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
                                row_dict['est_hourly_decay'] = signal_result['est_hourly_decay']
                                row_dict['est_remaining_decay'] = signal_result['est_remaining_decay']
                                call_signals.append(row_dict)
                        
                        if call_signals:
                            signals_df = pd.DataFrame(call_signals)
                            signals_df = signals_df.sort_values('signal_score', ascending=False)
                            
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta', 
                                            'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 
                                            'est_hourly_decay', 'est_remaining_decay', 'is_0dte']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            
                            st.dataframe(
                                signals_df[available_cols].round(4),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            if signals_df.iloc[0]['thresholds']:
                                th = signals_df.iloc[0]['thresholds']
                                st.info(
                                    f"Applied Thresholds: "
                                    f"Δ ≥ {th['delta_min']:.2f} | "
                                    f"Γ ≥ {th['gamma_min']:.3f} | "
                                    f"Θ ≤ {th['theta_base']:.3f} | "
                                    f"RSI > {th['rsi_min']:.1f} | "
                                    f"Vol > {th['volume_min']}"
                                )
                            
                            with st.expander("View Conditions for Top Signal"):
                                if signals_df.iloc[0]['passed_conditions']:
                                    st.write("✅ Passed Conditions:")
                                    for condition in signals_df.iloc[0]['passed_conditions']:
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
                                row_dict['est_hourly_decay'] = signal_result['est_hourly_decay']
                                row_dict['est_remaining_decay'] = signal_result['est_remaining_decay']
                                put_signals.append(row_dict)
                        
                        if put_signals:
                            signals_df = pd.DataFrame(put_signals)
                            signals_df = signals_df.sort_values('signal_score', ascending=False)
                            
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta', 
                                            'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 
                                            'est_hourly_decay', 'est_remaining_decay', 'is_0dte']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            
                            st.dataframe(
                                signals_df[available_cols].round(4),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            if signals_df.iloc[0]['thresholds']:
                                th = signals_df.iloc[0]['thresholds']
                                st.info(
                                    f"Applied Thresholds: "
                                    f"Δ ≤ {th['delta_max']:.2f} | "
                                    f"Γ ≥ {th['gamma_min']:.3f} | "
                                    f"Θ ≤ {th['theta_base']:.3f} | "
                                    f"RSI < {th['rsi_max']:.1f} | "
                                    f"Vol > {th['volume_min']}"
                                )
                            
                            with st.expander("View Conditions for Top Signal"):
                                if signals_df.iloc[0]['passed_conditions']:
                                    st.write("✅ Passed Conditions:")
                                    for condition in signals_df.iloc[0]['passed_conditions']:
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
                
                # Add Call/Put Scanner visual bars
                st.subheader("Call/Put Scanner")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Call Signal Strengths")
                    if call_signals:
                        scores = pd.DataFrame({'Signal Strength': [d['signal_score'] for d in call_signals]})
                        st.bar_chart(scores)
                    else:
                        st.info("No call signals")
                with col2:
                    st.write("Put Signal Strengths")
                    if put_signals:
                        scores = pd.DataFrame({'Signal Strength': [d['signal_score'] for d in put_signals]})
                        st.bar_chart(scores)
                    else:
                        st.info("No put signals")
                
                # Scanner scores
                call_score = calculate_scanner_score(df, 'call')
                put_score = calculate_scanner_score(df, 'put')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Call Scanner Score", f"{call_score:.2f}%")
                with col2:
                    st.metric("Put Scanner Score", f"{put_score:.2f}%")
                
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
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
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
                atr_pct = latest['ATR_pct']
                st.metric("Volatility (ATR%)", f"{atr_pct*100:.2f}%" if not pd.isna(atr_pct) else "N/A")
            
            st.subheader("Recent Data")
            display_df = df.tail(10)[['Datetime', 'Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'ATR_pct', 'Volume', 'avg_vol']].round(2)
            display_df['ATR_pct'] = display_df['ATR_pct'] * 100
            display_df['Volume Ratio'] = display_df['Volume'] / display_df['avg_vol']
            st.dataframe(display_df.rename(columns={'ATR_pct': 'ATR%', 'avg_vol': 'Avg Vol'}), use_container_width=True)
            
            st.subheader("📉 Interactive Chart")
            chart_fig = create_stock_chart(df)
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
    
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
        st.json(SIGNAL_THRESHOLDS)
        
        st.write("**Profit Targets:**")
        st.json(CONFIG['PROFIT_TARGETS'])
        
        st.write("**System Configuration:**")
        st.json(CONFIG)
    
    with tab4:
        st.subheader("📰 News & Events")
        stock = yf.Ticker(ticker)
        
        # News - Fixed KeyError
        news = stock.news
        if news:
            st.subheader("Recent News")
            for item in news[:10]:
                with st.expander(item.get('title', 'Untitled')):
                    st.write(f"Publisher: {item.get('publisher', 'Unknown')}")
                    # Fixed KeyError here - using get() with default value
                    st.write(f"Link: {item.get('link', 'No link available')}")
                    st.write(item.get('summary', 'No summary available'))
        else:
            st.info("No recent news available.")
        
        # Calendar/Events
        calendar = stock.calendar
        if not calendar.empty:
            st.subheader("Upcoming Events/Earnings")
            st.dataframe(calendar)
        else:
            st.info("No upcoming events available.")
        
else:
    st.info("Please enter a stock ticker to begin analysis.")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Steps to analyze options:**
        1. Enter a stock ticker (e.g., SPY, QQQ, AAPL)
        2. Configure auto-refresh settings in the sidebar (optional)
        3. Select expiration filter (0DTE for same-day, or near-term)
        4. Adjust strike range around current price
        5. Filter by moneyness (ITM, ATM, OTM)
        6. Review generated signals
        
        **Key Improvements:**
        - **Seamless Auto-Refresh:** Data updates automatically at your chosen interval
        - **Enhanced Volume Detection:** Fixed volume comparison logic
        - **Profit Targets & Exit Strategy:** Clear profit targets and holding periods
        - **Early Market Detection:** Special thresholds for premarket/early market
        - **Volatility-Based Adjustments:** Thresholds adapt to market conditions
        
        **New Features:**
        - **Profit Targets:** Set custom profit targets and stop losses
        - **Holding Period Suggestions:** Intelligent holding period recommendations
        - **Volume Thresholds:** Minimum volume requirements to filter low-liquidity options
        - **Diagnostic Details:** Clear reasons why signals fail
        """)

with st.expander("ℹ️ About Rate Limiting"):
    st.markdown("""
    Yahoo Finance may restrict how often data can be retrieved. If you see a "rate limited" warning, please:
    - Wait a few minutes before refreshing again
    - Avoid setting auto-refresh intervals lower than 1 minute
    - Use the app with one ticker at a time to reduce load
    """)
