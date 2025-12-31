# =============================
# IMPORTS & SETUP
# =============================
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import pytz
import math
import streamlit as st
import requests
from typing import Optional, Tuple, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, KeltnerChannel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from polygon import RESTClient
from streamlit_autorefresh import st_autorefresh

# Configure yfinance to be less verbose
import logging
logging.getLogger('yfinance').setLevel(logging.ERROR)
yf.pdr_override()

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Support/Resistance analysis will use simplified method.")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

st.set_page_config(
    page_title="Options Greeks Buy Signal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh for real-time updates
refresh_interval = st_autorefresh(interval=5000, limit=None, key="price_refresh")  # Increased to 5 seconds

# =============================
# ENHANCED CONFIGURATION & CONSTANTS
# =============================
CONFIG = {
    'POLYGON_API_KEY': '',  # Will be set from user input
    'ALPHA_VANTAGE_API_KEY': '',
    'FMP_API_KEY': '',
    'IEX_API_KEY': '',
    'MAX_RETRIES': 2,  # Reduced for faster fallback
    'RETRY_DELAY': 1,
    'DATA_TIMEOUT': 10,
    'MIN_DATA_POINTS': 20,
    'CACHE_TTL': 300,
    'STOCK_CACHE_TTL': 180,  # Reduced for more frequent updates
    'RATE_LIMIT_COOLDOWN': 180,
    'MIN_REFRESH_INTERVAL': 120,
    'MARKET_OPEN': datetime.time(9, 30),
    'MARKET_CLOSE': datetime.time(16, 0),
    'PREMARKET_START': datetime.time(4, 0),
    'VOLATILITY_THRESHOLDS': {
        'low': 0.015,
        'medium': 0.03,
        'high': 0.05
    },
    'PROFIT_TARGETS': {
        'call': 0.10,
        'put': 0.10,
        'stop_loss': 0.08
    },
    'TRADING_HOURS_PER_DAY': 6.5,
    'SR_TIME_WINDOWS': {
        'scalping': ['5min', '15min'],  # Reduced timeframes
        'intraday': ['30min', '1h']
    },
    'SR_SENSITIVITY': {
        '5min': 0.002,
        '15min': 0.003,
        '30min': 0.005,
        '1h': 0.008
    },
    'SR_WINDOW_SIZES': {
        '5min': 3,
        '15min': 5,
        '30min': 7,
        '1h': 10
    },
    'LIQUIDITY_THRESHOLDS': {
        'min_open_interest': 500,  # Reduced for better signal finding
        'min_volume': 100,
        'max_bid_ask_spread_pct': 0.15  # Increased for more flexibility
    },
    'MIN_OPTION_PRICE': 0.25,
    'MIN_OPEN_INTEREST': 500,
    'MIN_VOLUME': 100,
    'MAX_BID_ASK_SPREAD_PCT': 0.15,
}

# Initialize API call log in session state
if 'API_CALL_LOG' not in st.session_state:
    st.session_state.API_CALL_LOG = []

# Enhanced signal thresholds with weighted conditions
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
        'volume_min': 300,  # Reduced for scalping
        'condition_weights': {
            'delta': 0.25,
            'gamma': 0.20,
            'theta': 0.15,
            'trend': 0.20,
            'momentum': 0.10,
            'volume': 0.10
        }
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
        'volume_min': 300,  # Reduced for scalping
        'condition_weights': {
            'delta': 0.25,
            'gamma': 0.20,
            'theta': 0.15,
            'trend': 0.20,
            'momentum': 0.10,
            'volume': 0.10
        }
    }
}

# =============================
# UTILITY FUNCTIONS FOR FREE DATA SOURCES
# =============================
def can_make_request(source: str) -> bool:
    """Check if we can make another request without hitting limits"""
    now = time.time()
    
    # Clean up old entries (older than 1 hour)
    st.session_state.API_CALL_LOG = [
        t for t in st.session_state.API_CALL_LOG
        if now - t['timestamp'] < 3600
    ]
    
    # Count recent requests by source
    av_count = len([t for t in st.session_state.API_CALL_LOG
                   if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 60])
    fmp_count = len([t for t in st.session_state.API_CALL_LOG
                    if t['source'] == "FMP" and now - t['timestamp'] < 3600])
    iex_count = len([t for t in st.session_state.API_CALL_LOG
                   if t['source'] == "IEX" and now - t['timestamp'] < 3600])
    
    # Enforce rate limits
    if source == "ALPHA_VANTAGE" and av_count >= 4:
        return False
    if source == "FMP" and fmp_count >= 9:
        return False
    if source == "IEX" and iex_count >= 29:
        return False
    
    return True

def log_api_request(source: str):
    """Log an API request to track usage"""
    st.session_state.API_CALL_LOG.append({
        'source': source,
        'timestamp': time.time()
    })

# =============================
# OPTIMIZED SUPPORT/RESISTANCE FUNCTIONS
# =============================
def find_peaks_valleys_simple(data: np.array, window: int = 5) -> Tuple[List[int], List[int]]:
    """Simple peak and valley detection without scipy dependency"""
    if len(data) < window * 2 + 1:
        return [], []
    
    peaks = []
    valleys = []
    
    for i in range(window, len(data) - window):
        # Check for peak
        if all(data[i] > data[i-j] for j in range(1, window+1)) and \
           all(data[i] > data[i+j] for j in range(1, window+1)):
            peaks.append(i)
        
        # Check for valley
        if all(data[i] < data[i-j] for j in range(1, window+1)) and \
           all(data[i] < data[i+j] for j in range(1, window+1)):
            valleys.append(i)
    
    return peaks, valleys

def calculate_support_resistance_simple(data: pd.DataFrame, timeframe: str, current_price: float) -> dict:
    """Simple support/resistance calculation"""
    if data.empty or len(data) < 20:
        return {
            'support': [],
            'resistance': [],
            'sensitivity': CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005),
            'timeframe': timeframe,
            'data_points': len(data) if not data.empty else 0
        }
    
    try:
        window_size = CONFIG['SR_WINDOW_SIZES'].get(timeframe, 5)
        sensitivity = CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005)
        
        # Get highs and lows
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find peaks and valleys
        peaks, _ = find_peaks_valleys_simple(highs, window_size)
        _, valleys = find_peaks_valleys_simple(lows, window_size)
        
        # Extract levels
        resistance_levels = [float(highs[i]) for i in peaks]
        support_levels = [float(lows[i]) for i in valleys]
        
        # Filter levels close to current price
        min_distance = current_price * 0.005
        
        # Separate support and resistance
        resistance_levels = [level for level in resistance_levels 
                           if level > current_price + min_distance]
        support_levels = [level for level in support_levels 
                         if level < current_price - min_distance]
        
        # Take top 3 levels
        resistance_levels = sorted(resistance_levels)[:3]
        support_levels = sorted(support_levels, reverse=True)[:3]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'sensitivity': sensitivity,
            'timeframe': timeframe,
            'data_points': len(data)
        }
        
    except Exception as e:
        return {
            'support': [],
            'resistance': [],
            'sensitivity': CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005),
            'timeframe': timeframe,
            'data_points': len(data)
        }

@st.cache_data(ttl=300, show_spinner=False)
def get_multi_timeframe_data_simple(ticker: str) -> Tuple[dict, float]:
    """Simplified multi-timeframe data fetching"""
    timeframes = {
        '5min': {'interval': '5m', 'period': '5d'},
        '15min': {'interval': '15m', 'period': '5d'},
        '30min': {'interval': '30m', 'period': '10d'},
    }
    
    data = {}
    current_price = 0
    
    for tf, params in timeframes.items():
        try:
            df = yf.download(
                ticker,
                period=params['period'],
                interval=params['interval'],
                progress=False,
                prepost=True,
                threads=False  # Disable threading for stability
            )
            
            if not df.empty:
                # Clean data
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Get current price from latest close
                if current_price == 0 and not df.empty:
                    current_price = float(df['Close'].iloc[-1])
                
                # Store data
                data[tf] = df
                
        except Exception:
            continue
    
    # Fallback for current price
    if current_price == 0:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            if 'regularMarketPrice' in info:
                current_price = info['regularMarketPrice']
            elif 'previousClose' in info:
                current_price = info['previousClose']
        except:
            current_price = 100.0
    
    return data, current_price

def analyze_support_resistance_simple(ticker: str) -> dict:
    """Simple support/resistance analysis"""
    try:
        tf_data, current_price = get_multi_timeframe_data_simple(ticker)
        
        if not tf_data:
            return {}
        
        results = {}
        
        for timeframe, data in tf_data.items():
            if not data.empty:
                sr_result = calculate_support_resistance_simple(data, timeframe, current_price)
                results[timeframe] = sr_result
        
        return results
        
    except Exception:
        return {}

def plot_sr_levels_simple(data: dict, current_price: float) -> go.Figure:
    """Simple visualization of support/resistance levels"""
    fig = go.Figure()
    
    # Add current price line
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color="blue",
        line_width=3,
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="top right"
    )
    
    # Prepare data for plotting
    for tf, sr in data.items():
        # Add support levels
        for level in sr.get('support', []):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                line_width=2,
                annotation_text=f"S: ${level:.2f}",
                annotation_position="bottom right"
            )
        
        # Add resistance levels
        for level in sr.get('resistance', []):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"R: ${level:.2f}",
                annotation_position="top right"
            )
    
    fig.update_layout(
        title='Support & Resistance Levels',
        xaxis_title='Timeframe',
        yaxis_title='Price ($)',
        height=500,
        template='plotly_dark'
    )
    
    return fig

# =============================
# ENHANCED UTILITY FUNCTIONS
# =============================
def is_market_open() -> bool:
    """Check if market is currently open based on Eastern Time"""
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        now_time = now.time()
        
        if now.weekday() >= 5:
            return False
        
        return CONFIG['MARKET_OPEN'] <= now_time <= CONFIG['MARKET_CLOSE']
    except Exception:
        return False

def is_premarket() -> bool:
    """Check if we're in premarket hours"""
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        now_time = now.time()
        
        if now.weekday() >= 5:
            return False
        
        return CONFIG['PREMARKET_START'] <= now_time < CONFIG['MARKET_OPEN']
    except Exception:
        return False

# UPDATED: Simplified price fetching
@st.cache_data(ttl=30, show_spinner=False)
def get_current_price(ticker: str) -> float:
    """Get current price with fallbacks"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try to get quick price
        try:
            info = stock.info
            if 'regularMarketPrice' in info:
                return float(info['regularMarketPrice'])
            elif 'currentPrice' in info:
                return float(info['currentPrice'])
            elif 'previousClose' in info:
                return float(info['previousClose'])
        except:
            pass
        
        # Fallback to history
        try:
            hist = stock.history(period='1d', interval='1m', prepost=False)
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        
        # Final fallback
        return 0.0
        
    except Exception:
        return 0.0

# UPDATED: Simplified stock data fetching
@st.cache_data(ttl=CONFIG['STOCK_CACHE_TTL'], show_spinner=False)
def get_stock_data_with_indicators(ticker: str) -> pd.DataFrame:
    """Fetch stock data and compute indicators"""
    try:
        # Get data
        df = yf.download(
            ticker,
            period='5d',
            interval='5m',
            progress=False,
            prepost=True,
            threads=False
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Reset index
        df = df.reset_index()
        
        # Calculate EMAs
        close = df['Close'].astype(float)
        for period in [9, 20]:
            if len(close) >= period:
                ema = EMAIndicator(close=close, window=period)
                df[f'EMA_{period}'] = ema.ema_indicator()
        
        # Calculate RSI
        if len(close) >= 14:
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
        
        # Calculate VWAP
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns and 'Volume' in df.columns:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Calculate ATR
        if len(close) >= 14:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=close, window=14)
            df['ATR'] = atr.average_true_range()
            df['ATR_pct'] = df['ATR'] / close
        
        return df
        
    except Exception:
        return pd.DataFrame()

# UPDATED: Better options data fetching
def get_options_data_smart(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Get options data with smart fallback"""
    
    # Check if we should use demo data
    if 'use_demo_data' in st.session_state and st.session_state.use_demo_data:
        return get_fallback_options_data(ticker)
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get expiries
        expiries = stock.options
        if not expiries:
            return get_fallback_options_data(ticker)
        
        # Get only nearest expiry
        nearest = expiries[0]
        
        try:
            chain = stock.option_chain(nearest)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["too many", "rate", "limit", "429"]):
                st.session_state.last_rate_limit = time.time()
                return get_fallback_options_data(ticker)
            else:
                return get_fallback_options_data(ticker)
        
        if chain is None:
            return get_fallback_options_data(ticker)
        
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        
        if calls.empty or puts.empty:
            return get_fallback_options_data(ticker)
        
        # Add expiry
        calls['expiry'] = nearest
        puts['expiry'] = nearest
        
        # Fill missing Greeks
        for df in [calls, puts]:
            for greek in ['delta', 'gamma', 'theta']:
                if greek not in df.columns:
                    df[greek] = np.nan
        
        return [nearest], calls, puts
        
    except Exception:
        return get_fallback_options_data(ticker)

def get_fallback_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Generate realistic fallback options data"""
    
    # Get current price
    current_price = get_current_price(ticker)
    if current_price <= 0:
        current_price = 100.0
    
    # Create strikes
    strike_range = current_price * 0.10  # 10% range
    strikes = []
    increment = 2.5 if current_price > 200 else 1.0 if current_price > 50 else 0.5
    
    start = int((current_price - strike_range) / increment) * increment
    end = int((current_price + strike_range) / increment) * increment
    
    strike = start
    while strike <= end:
        if strike > 0:
            strikes.append(round(strike, 2))
        strike += increment
    
    # Create expiries
    today = datetime.date.today()
    expiries = []
    
    # Add today if weekday
    if today.weekday() < 5:
        expiries.append(today.strftime('%Y-%m-%d'))
    
    # Add next 2 Fridays
    for i in range(1, 3):
        days_to_friday = (4 - today.weekday()) % 7 + (7 * (i-1))
        if days_to_friday == 0:
            days_to_friday = 7
        expiry = today + datetime.timedelta(days=days_to_friday)
        expiries.append(expiry.strftime('%Y-%m-%d'))
    
    # Generate data
    calls_data = []
    puts_data = []
    
    for expiry in expiries[:2]:  # Limit to 2 expiries
        expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
        days_to_expiry = (expiry_date - today).days
        
        for strike in strikes:
            # Calculate moneyness
            moneyness = current_price / strike
            
            # Greeks
            if moneyness > 1.02:  # ITM calls
                call_delta = 0.8
                put_delta = -0.2
            elif moneyness > 0.98:  # ATM
                call_delta = 0.5
                put_delta = -0.5
            else:  # OTM calls
                call_delta = 0.2
                put_delta = -0.8
            
            gamma = 0.05
            theta = -0.10 if days_to_expiry <= 1 else -0.05
            
            # Price
            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = 0.5 if days_to_expiry <= 1 else 1.0
            
            call_price = max(0.25, intrinsic_call + time_value)
            put_price = max(0.25, intrinsic_put + time_value)
            
            # Volume/OI
            volume = 100 if abs(moneyness - 1) < 0.02 else 50
            oi = volume * 2
            
            calls_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}C{int(strike*1000):08d}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(call_price, 2),
                'volume': volume,
                'openInterest': oi,
                'impliedVolatility': 0.30,
                'delta': round(call_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(call_price * 0.99, 2),
                'ask': round(call_price * 1.01, 2)
            })
            
            puts_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}P{int(strike*1000):08d}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(put_price, 2),
                'volume': volume,
                'openInterest': oi,
                'impliedVolatility': 0.30,
                'delta': round(put_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(put_price * 0.99, 2),
                'ask': round(put_price * 1.01, 2)
            })
    
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    
    return expiries[:2], calls_df, puts_df

def classify_moneyness(strike: float, spot: float) -> str:
    """Classify option moneyness"""
    diff_pct = abs(strike - spot) / spot
    
    if diff_pct < 0.01:
        return 'ATM'
    elif strike < spot:
        return 'ITM' if diff_pct > 0.03 else 'NTM'
    else:
        return 'OTM' if diff_pct > 0.03 else 'NTM'

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    """Validate option data"""
    try:
        required = ['strike', 'lastPrice', 'volume', 'openInterest', 'bid', 'ask']
        for field in required:
            if field not in option or pd.isna(option[field]):
                return False
        
        if option['lastPrice'] < CONFIG['MIN_OPTION_PRICE']:
            return False
        
        if option['bid'] <= 0 or option['ask'] <= 0:
            return False
        
        spread_pct = (option['ask'] - option['bid']) / option['ask']
        if spread_pct > CONFIG['MAX_BID_ASK_SPREAD_PCT']:
            return False
        
        return True
        
    except Exception:
        return False

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
    """Calculate dynamic thresholds"""
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    
    volatility = stock_data.get('ATR_pct', 0.02)
    if pd.isna(volatility):
        volatility = 0.02
    
    vol_multiplier = 1 + volatility * 50
    
    if side == 'call':
        thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_multiplier))
    else:
        thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_multiplier))
    
    thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + thresholds['gamma_vol_multiplier'] * volatility * 100)
    
    # Adjust for market conditions
    if is_premarket() or not is_market_open():
        thresholds['volume_multiplier'] = 0.7
        thresholds['gamma_min'] *= 0.8
    
    if is_0dte:
        thresholds['volume_multiplier'] *= 0.8
        if side == 'call':
            thresholds['delta_min'] = max(0.4, thresholds['delta_min'])
        else:
            thresholds['delta_max'] = min(-0.4, thresholds['delta_max'])
    
    return thresholds

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    """Generate trading signal"""
    if stock_df.empty:
        return {'signal': False, 'score': 0.0}
    
    current_price = stock_df.iloc[-1]['Close']
    
    if not validate_option_data(option, current_price):
        return {'signal': False, 'score': 0.0}
    
    latest = stock_df.iloc[-1]
    
    try:
        thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
        weights = thresholds['condition_weights']
        
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        option_volume = float(option['volume'])
        
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        
        score = 0.0
        conditions_met = 0
        total_conditions = 0
        
        # Delta condition
        if side == "call":
            if delta >= thresholds.get('delta_min', 0.5):
                score += weights['delta']
                conditions_met += 1
        else:
            if delta <= thresholds.get('delta_max', -0.5):
                score += weights['delta']
                conditions_met += 1
        total_conditions += 1
        
        # Gamma condition
        if gamma >= thresholds.get('gamma_min', 0.05):
            score += weights['gamma']
            conditions_met += 1
        total_conditions += 1
        
        # Trend condition
        if side == "call":
            if ema_9 and ema_20 and close > ema_9 > ema_20:
                score += weights['trend']
                conditions_met += 1
        else:
            if ema_9 and ema_20 and close < ema_9 < ema_20:
                score += weights['trend']
                conditions_met += 1
        total_conditions += 1
        
        # RSI condition
        if rsi:
            if (side == "call" and rsi > 50) or (side == "put" and rsi < 50):
                score += weights['momentum']
                conditions_met += 1
        total_conditions += 1
        
        # Volume condition
        if option_volume > thresholds['volume_min']:
            score += weights['volume']
            conditions_met += 1
        total_conditions += 1
        
        # VWAP condition (bonus)
        if vwap:
            if (side == "call" and close > vwap) or (side == "put" and close < vwap):
                score += 0.1
                conditions_met += 1
            total_conditions += 1
        
        # Calculate signal
        signal_strength = score
        signal = signal_strength >= 0.6  # 60% threshold
        
        # Calculate profit targets
        profit_target = None
        stop_loss = None
        
        if signal:
            entry_price = option['lastPrice']
            if side == 'call':
                profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['call'])
                stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
            else:
                profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['put'])
                stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
        
        return {
            'signal': signal,
            'score': signal_strength * 100,
            'conditions_met': conditions_met,
            'total_conditions': total_conditions,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'open_interest': option['openInterest'],
            'volume': option['volume']
        }
        
    except Exception:
        return {'signal': False, 'score': 0.0}

def process_options(options_df: pd.DataFrame, side: str, stock_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Process options for signals"""
    if options_df.empty:
        return pd.DataFrame()
    
    options_df = options_df.copy()
    
    # Filter by price
    options_df = options_df[options_df['lastPrice'] >= CONFIG['MIN_OPTION_PRICE']]
    
    if options_df.empty:
        return pd.DataFrame()
    
    # Add moneyness
    options_df['moneyness'] = options_df['strike'].apply(
        lambda x: classify_moneyness(x, current_price)
    )
    
    # Add 0DTE flag
    today = datetime.date.today()
    options_df['is_0dte'] = options_df['expiry'].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today if pd.notna(x) else False
    )
    
    # Process signals
    signals = []
    for idx, row in options_df.iterrows():
        signal_result = generate_signal(row, side, stock_df, row['is_0dte'])
        if signal_result['signal']:
            row_dict = row.to_dict()
            row_dict.update(signal_result)
            signals.append(row_dict)
    
    if signals:
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values(['is_0dte', 'score'], ascending=[False, False])
        return signals_df
    
    return pd.DataFrame()

def create_simple_chart(df: pd.DataFrame):
    """Create simple price chart"""
    if df.empty:
        return None
    
    try:
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['Datetime'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # EMAs
        if 'EMA_9' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'],
                y=df['EMA_9'],
                name='EMA 9',
                line=dict(color='blue')
            ))
        
        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'],
                y=df['EMA_20'],
                name='EMA 20',
                line=dict(color='orange')
            ))
        
        fig.update_layout(
            height=400,
            title='Price Chart',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        return fig
        
    except Exception:
        return None

# =============================
# STREAMLIT INTERFACE
# =============================
# Initialize session state
if 'sr_data' not in st.session_state:
    st.session_state.sr_data = {}
if 'use_demo_data' not in st.session_state:
    st.session_state.use_demo_data = False
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

st.title("📈 Options Signal Analyzer")
st.markdown("**Lightweight** • **Fast** • **Real-time**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    st.subheader("🔑 API Keys (Optional)")
    polygon_key = st.text_input("Polygon API Key:", type="password")
    if polygon_key:
        CONFIG['POLYGON_API_KEY'] = polygon_key
    
    # Data Source
    st.subheader("📊 Data Source")
    use_demo = st.checkbox("Use Demo Data", value=st.session_state.use_demo_data)
    if use_demo != st.session_state.use_demo_data:
        st.session_state.use_demo_data = use_demo
        st.rerun()
    
    # Settings
    st.subheader("⚡ Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        CONFIG['MIN_OPTION_PRICE'] = st.slider("Min Price", 0.05, 1.0, 0.25, 0.05)
        CONFIG['MIN_VOLUME'] = st.slider("Min Volume", 50, 500, 100, 50)
    
    with col2:
        CONFIG['MIN_OPEN_INTEREST'] = st.slider("Min OI", 100, 2000, 500, 100)
        CONFIG['MAX_BID_ASK_SPREAD_PCT'] = st.slider("Max Spread %", 0.05, 0.50, 0.15, 0.05)
    
    # Market Status
    st.subheader("🕐 Market Status")
    if is_market_open():
        st.success("🟢 Market OPEN")
    elif is_premarket():
        st.warning("🟡 PREMARKET")
    else:
        st.info("🔴 Market CLOSED")
    
    # Refresh
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()

# Main interface
ticker = st.text_input("Enter Stock Ticker:", value="SPY").upper()

if ticker:
    # Header with metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = get_current_price(ticker)
        if current_price > 0:
            st.metric("Current Price", f"${current_price:.2f}")
        else:
            st.error("Price Error")
    
    with col2:
        if is_market_open():
            st.success("🟢 OPEN")
        elif is_premarket():
            st.warning("🟡 PRE")
        else:
            st.info("🔴 CLOSED")
    
    with col3:
        cache_age = int(time.time() - st.session_state.last_refresh)
        st.metric("Cache Age", f"{cache_age}s")
    
    with col4:
        if st.button("📊 Update Now"):
            st.cache_data.clear()
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Signals",
        "📊 Technical",
        "📈 S/R Levels",
        "⚙️ Settings"
    ])
    
    with tab1:
        try:
            # Get stock data
            df = get_stock_data_with_indicators(ticker)
            
            if df.empty:
                st.error("Unable to fetch stock data")
                st.stop()
            
            current_price = df.iloc[-1]['Close']
            st.success(f"**{ticker}**: ${current_price:.2f}")
            
            # Get options data
            with st.spinner("Loading options data..."):
                expiries, calls, puts = get_options_data_smart(ticker)
            
            if not expiries:
                st.error("No options data available")
                st.info("Try using demo data or check if market is open")
                st.stop()
            
            # Filter for 0DTE
            today = datetime.date.today()
            calls['is_0dte'] = calls['expiry'].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today
            )
            puts['is_0dte'] = puts['expiry'].apply(
                lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today
            )
            
            # Filter by strike range
            strike_range = st.slider("Strike Range ($)", -10.0, 10.0, (-3.0, 3.0), 0.5)
            min_strike = current_price + strike_range[0]
            max_strike = current_price + strike_range[1]
            
            calls_filtered = calls[
                (calls['strike'] >= min_strike) & 
                (calls['strike'] <= max_strike)
            ].copy()
            
            puts_filtered = puts[
                (puts['strike'] >= min_strike) & 
                (puts['strike'] <= max_strike)
            ].copy()
            
            # Process signals
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Call Signals")
                if not calls_filtered.empty:
                    call_signals = process_options(calls_filtered, "call", df, current_price)
                    
                    if not call_signals.empty:
                        display_cols = ['contractSymbol', 'strike', 'lastPrice', 'score', 
                                      'conditions_met', 'total_conditions', 'is_0dte']
                        display_df = call_signals[display_cols].copy()
                        display_df = display_df.sort_values('score', ascending=False)
                        display_df = display_df.head(10)  # Limit to top 10
                        
                        st.dataframe(
                            display_df.rename(columns={
                                'score': 'Score%',
                                'lastPrice': 'Price',
                                'is_0dte': '0DTE'
                            }),
                            use_container_width=True
                        )
                        
                        st.info(f"Found {len(call_signals)} call signals")
                    else:
                        st.info("No call signals found")
                else:
                    st.info("No call options in range")
            
            with col2:
                st.subheader("📉 Put Signals")
                if not puts_filtered.empty:
                    put_signals = process_options(puts_filtered, "put", df, current_price)
                    
                    if not put_signals.empty:
                        display_cols = ['contractSymbol', 'strike', 'lastPrice', 'score', 
                                      'conditions_met', 'total_conditions', 'is_0dte']
                        display_df = put_signals[display_cols].copy()
                        display_df = display_df.sort_values('score', ascending=False)
                        display_df = display_df.head(10)  # Limit to top 10
                        
                        st.dataframe(
                            display_df.rename(columns={
                                'score': 'Score%',
                                'lastPrice': 'Price',
                                'is_0dte': '0DTE'
                            }),
                            use_container_width=True
                        )
                        
                        st.info(f"Found {len(put_signals)} put signals")
                    else:
                        st.info("No put signals found")
                else:
                    st.info("No put options in range")
            
            # Show top signal details
            st.subheader("🎯 Top Signals")
            
            top_signals = []
            if 'call_signals' in locals() and not call_signals.empty:
                top_call = call_signals.iloc[0]
                top_signals.append(('CALL', top_call))
            
            if 'put_signals' in locals() and not put_signals.empty:
                top_put = put_signals.iloc[0]
                top_signals.append(('PUT', top_put))
            
            if top_signals:
                cols = st.columns(len(top_signals))
                for idx, (option_type, signal_data) in enumerate(top_signals):
                    with cols[idx]:
                        st.metric(
                            f"Top {option_type}",
                            f"{signal_data['score']:.1f}%",
                            f"Strike: ${signal_data['strike']}"
                        )
                        st.caption(f"Price: ${signal_data['lastPrice']:.2f}")
                        st.caption(f"Conditions: {signal_data['conditions_met']}/{signal_data['total_conditions']}")
                        if signal_data.get('profit_target'):
                            st.caption(f"Target: ${signal_data['profit_target']:.2f}")
                
        except Exception as e:
            st.error(f"Error in signal analysis: {str(e)}")
    
    with tab2:
        try:
            if 'df' not in locals():
                df = get_stock_data_with_indicators(ticker)
            
            if not df.empty:
                # Metrics
                latest = df.iloc[-1]
                cols = st.columns(5)
                
                with cols[0]:
                    st.metric("Price", f"${latest['Close']:.2f}")
                
                with cols[1]:
                    if 'EMA_9' in latest and not pd.isna(latest['EMA_9']):
                        st.metric("EMA 9", f"${latest['EMA_9']:.2f}")
                
                with cols[2]:
                    if 'EMA_20' in latest and not pd.isna(latest['EMA_20']):
                        st.metric("EMA 20", f"${latest['EMA_20']:.2f}")
                
                with cols[3]:
                    if 'RSI' in latest and not pd.isna(latest['RSI']):
                        st.metric("RSI", f"{latest['RSI']:.1f}")
                
                with cols[4]:
                    if 'VWAP' in latest and not pd.isna(latest['VWAP']):
                        st.metric("VWAP", f"${latest['VWAP']:.2f}")
                
                # Chart
                chart = create_simple_chart(df)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Recent data
                st.subheader("Recent Data")
                display_df = df.tail(10)[['Datetime', 'Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP']].copy()
                display_df['Datetime'] = display_df['Datetime'].dt.strftime('%H:%M')
                st.dataframe(display_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in technical analysis: {str(e)}")
    
    with tab3:
        st.subheader("Support & Resistance Levels")
        
        # Calculate S/R
        if not st.session_state.sr_data or st.session_state.get('last_ticker') != ticker:
            with st.spinner("Calculating levels..."):
                st.session_state.sr_data = analyze_support_resistance_simple(ticker)
                st.session_state.last_ticker = ticker
        
        if st.session_state.sr_data:
            # Plot
            sr_fig = plot_sr_levels_simple(st.session_state.sr_data, current_price)
            if sr_fig:
                st.plotly_chart(sr_fig, use_container_width=True)
            
            # Display levels
            st.subheader("Key Levels")
            
            for timeframe, sr in st.session_state.sr_data.items():
                with st.expander(f"{timeframe} Timeframe"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Support**")
                        for level in sr.get('support', []):
                            distance = ((level - current_price) / current_price * 100)
                            st.write(f"${level:.2f} ({distance:+.1f}%)")
                    
                    with col2:
                        st.write("**Resistance**")
                        for level in sr.get('resistance', []):
                            distance = ((level - current_price) / current_price * 100)
                            st.write(f"${level:.2f} ({distance:+.1f}%)")
        else:
            st.info("No support/resistance data available")
    
    with tab4:
        st.subheader("Settings & Info")
        
        st.write("**Current Configuration:**")
        st.json({
            'Min Option Price': f"${CONFIG['MIN_OPTION_PRICE']}",
            'Min Volume': CONFIG['MIN_VOLUME'],
            'Min Open Interest': CONFIG['MIN_OPEN_INTEREST'],
            'Max Spread': f"{CONFIG['MAX_BID_ASK_SPREAD_PCT']*100}%",
            'Cache TTL': f"{CONFIG['STOCK_CACHE_TTL']}s",
            'Using Demo Data': st.session_state.use_demo_data
        })
        
        st.write("**Performance Tips:**")
        st.markdown("""
        1. **During market hours** (9:30 AM - 4:00 PM ET) for best data
        2. **Use popular tickers** like SPY, QQQ, AAPL
        3. **Refresh every 2-3 minutes** to avoid rate limits
        4. **Enable demo data** if real data is unavailable
        5. **Clear cache** if data seems stale
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear All Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                
        with col2:
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

else:
    st.info("👋 Enter a stock ticker above to begin analysis")
    
    with st.expander("📚 Quick Start Guide"):
        st.markdown("""
        **Getting Started:**
        1. **Enter a ticker** (e.g., SPY, QQQ, AAPL)
        2. **Check market status** in sidebar
        3. **Adjust settings** as needed
        4. **View signals** in the Signals tab
        
        **Recommended Settings:**
        - **For scalping**: Use tight strike ranges (±$3)
        - **For swing trading**: Use wider ranges (±$10)
        - **Min Option Price**: $0.25 for scalping
        - **Min Volume**: 100+ for liquidity
        
        **Data Sources:**
        - **Real Data**: During market hours
        - **Demo Data**: Always available (toggle in sidebar)
        - **Polygon API**: For premium data (optional)
        """)

# Simple auto-refresh
if time.time() - st.session_state.last_refresh > 180:  # 3 minutes
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()
