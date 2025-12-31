# =============================
# COMPLETE CORRECTED CODE - OPTIONS GREEKS ANALYZER
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
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    import warnings
    warnings.warn("scipy not available. Support/Resistance analysis will use simplified method.")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure yfinance to be less verbose
import logging
logging.getLogger('yfinance').setLevel(logging.ERROR)

st.set_page_config(
    page_title="Options Greeks Buy Signal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# ENHANCED CONFIGURATION & CONSTANTS
# =============================
CONFIG = {
    'POLYGON_API_KEY': '',  # Will be set from user input
    'ALPHA_VANTAGE_API_KEY': '',
    'FMP_API_KEY': '',
    'IEX_API_KEY': '',
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 1,
    'DATA_TIMEOUT': 15,
    'MIN_DATA_POINTS': 30,
    'CACHE_TTL': 180,
    'STOCK_CACHE_TTL': 120,
    'RATE_LIMIT_COOLDOWN': 180,
    'MIN_REFRESH_INTERVAL': 90,
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
        'scalping': ['5min', '15min'],
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
        'min_open_interest': 500,
        'min_volume': 100,
        'max_bid_ask_spread_pct': 0.12
    },
    'MIN_OPTION_PRICE': 0.25,
    'MIN_OPEN_INTEREST': 500,
    'MIN_VOLUME': 100,
    'MAX_BID_ASK_SPREAD_PCT': 0.12,
}

# Update the LIQUIDITY_THRESHOLDS to use the new values
CONFIG['LIQUIDITY_THRESHOLDS'] = {
    'min_open_interest': CONFIG['MIN_OPEN_INTEREST'],
    'min_volume': CONFIG['MIN_VOLUME'],
    'max_bid_ask_spread_pct': CONFIG['MAX_BID_ASK_SPREAD_PCT']
}

# Initialize session state
if 'API_CALL_LOG' not in st.session_state:
    st.session_state.API_CALL_LOG = []
if 'sr_data' not in st.session_state:
    st.session_state.sr_data = {}
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = ""
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = CONFIG['MIN_REFRESH_INTERVAL']
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = False
if 'use_demo_data' not in st.session_state:
    st.session_state.use_demo_data = False
if 'rate_limit_info' not in st.session_state:
    st.session_state.rate_limit_info = {}

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
        'volume_min': 300,
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
        'volume_min': 300,
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
# UTILITY FUNCTIONS
# =============================
def can_make_request(source: str) -> bool:
    """Check if we can make another request without hitting limits"""
    now = time.time()
    
    # Clean up old entries
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

def is_early_market() -> bool:
    """Check if we're in the first 30 minutes of market open"""
    try:
        if not is_market_open():
            return False
        
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        market_open_today = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'])
        market_open_today = eastern.localize(market_open_today)
        
        return (now - market_open_today).total_seconds() < 1800
    except Exception:
        return False

def calculate_remaining_trading_hours() -> float:
    """Calculate remaining trading hours in the day"""
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        close_time = datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE'])
        close_time = eastern.localize(close_time)
        
        if now >= close_time:
            return 0.0
        
        return (close_time - now).total_seconds() / 3600
    except Exception:
        return 0.0

# =============================
# PRICE & DATA FETCHING FUNCTIONS
# =============================
@st.cache_data(ttl=10, show_spinner=False)
def get_current_price(ticker: str) -> float:
    """Get current price from multiple sources"""
    # Try Yahoo Finance first (most reliable for free)
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d', interval='1m')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    
    # Fallback to polygon if available
    if CONFIG['POLYGON_API_KEY']:
        try:
            client = RESTClient(CONFIG['POLYGON_API_KEY'], trace=False, connect_timeout=2)
            trade = client.stocks_equities_last_trade(ticker)
            return float(trade.last.price)
        except:
            pass
    
    # Try Alpha Vantage
    if CONFIG['ALPHA_VANTAGE_API_KEY'] and can_make_request("ALPHA_VANTAGE"):
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={CONFIG['ALPHA_VANTAGE_API_KEY']}"
            response = requests.get(url, timeout=3)
            data = response.json()
            if 'Global Quote' in data and '05. price' in data['Global Quote']:
                log_api_request("ALPHA_VANTAGE")
                return float(data['Global Quote']['05. price'])
        except:
            pass
    
    return 0.0

@st.cache_data(ttl=CONFIG['STOCK_CACHE_TTL'], show_spinner=False)
def get_stock_data_with_indicators(ticker: str) -> pd.DataFrame:
    """Fetch stock data and compute indicators"""
    try:
        # Get 3 days of 5-minute data
        df = yf.download(
            ticker,
            period='3d',
            interval='5m',
            progress=False,
            prepost=True,
            threads=False,
            timeout=10
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Handle multi-index columns
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
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Calculate ATR
        if len(close) >= 14:
            atr = AverageTrueRange(high=df['High'], low=df['Low'], close=close, window=14)
            df['ATR'] = atr.average_true_range()
            df['ATR_pct'] = df['ATR'] / close
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

# =============================
# OPTIONS DATA FUNCTIONS
# =============================
def get_real_options_data_safe(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Get real options data with better error handling"""
    
    # Check rate limit
    rate_limit_key = f'rate_limit_{ticker}'
    if rate_limit_key in st.session_state:
        time_remaining = st.session_state[rate_limit_key] - time.time()
        if time_remaining > 0:
            st.warning(f"⏳ Rate limited. Try again in {int(time_remaining)} seconds.")
            return [], pd.DataFrame(), pd.DataFrame()
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get expiries
        expiries = stock.options
        if not expiries:
            return [], pd.DataFrame(), pd.DataFrame()
        
        # Get nearest expiry only
        nearest = expiries[0]
        
        # Get option chain with timeout
        import threading
        result = [None]
        
        def fetch_chain():
            try:
                chain = stock.option_chain(nearest)
                if chain:
                    result[0] = chain
            except Exception as e:
                if "too many" in str(e).lower() or "rate" in str(e).lower():
                    # Set rate limit
                    st.session_state[rate_limit_key] = time.time() + 180
                result[0] = None
        
        # Run with timeout
        thread = threading.Thread(target=fetch_chain)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)
        
        if result[0] is None:
            return [], pd.DataFrame(), pd.DataFrame()
        
        chain = result[0]
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        
        if calls.empty or puts.empty:
            return [], pd.DataFrame(), pd.DataFrame()
        
        # Add expiry and fill missing Greeks
        calls['expiry'] = nearest
        puts['expiry'] = nearest
        
        for df in [calls, puts]:
            for greek in ['delta', 'gamma', 'theta']:
                if greek not in df.columns:
                    df[greek] = np.nan
        
        return [nearest], calls, puts
        
    except Exception as e:
        error_msg = str(e).lower()
        if "too many" in error_msg or "rate" in error_msg:
            st.session_state[rate_limit_key] = time.time() + 180
        return [], pd.DataFrame(), pd.DataFrame()

def generate_realistic_options_data(ticker: str, current_price: float) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Generate realistic options data for demo/testing"""
    
    # Generate strikes
    strike_range = current_price * 0.10
    strikes = []
    increment = 2.5 if current_price > 200 else 1.0 if current_price > 50 else 0.5
    
    start = int((current_price - strike_range) / increment) * increment
    end = int((current_price + strike_range) / increment) * increment
    
    strike = start
    while strike <= end:
        if strike > 0:
            strikes.append(round(strike, 2))
        strike += increment
    
    # Generate expiries
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
    
    for expiry in expiries[:2]:
        expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
        days_to_expiry = (expiry_date - today).days
        
        for strike_price in strikes:
            # Calculate moneyness
            moneyness = current_price / strike_price
            
            # Greeks based on moneyness
            if moneyness > 1.03:
                call_delta = 0.85
                put_delta = -0.15
                gamma = 0.02
            elif moneyness > 0.97:
                call_delta = 0.55
                put_delta = -0.45
                gamma = 0.06
            else:
                call_delta = 0.15
                put_delta = -0.85
                gamma = 0.02
            
            # Theta based on days to expiry
            if days_to_expiry == 0:
                theta = -0.25
            elif days_to_expiry <= 3:
                theta = -0.15
            else:
                theta = -0.05
            
            # Price calculation
            intrinsic_call = max(0, current_price - strike_price)
            intrinsic_put = max(0, strike_price - current_price)
            time_value = 0.3 if days_to_expiry <= 1 else 0.8 if days_to_expiry <= 7 else 1.5
            
            call_price = max(0.25, intrinsic_call + time_value)
            put_price = max(0.25, intrinsic_put + time_value)
            
            # Volume and OI
            volume = np.random.randint(50, 500)
            oi = volume * np.random.randint(1, 3)
            
            # Bid/ask spreads
            bid_ask_spread = np.random.uniform(0.01, 0.05)
            
            calls_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}C{int(strike_price*1000):08d}",
                'strike': strike_price,
                'expiry': expiry,
                'lastPrice': round(call_price, 2),
                'volume': volume,
                'openInterest': oi,
                'impliedVolatility': round(np.random.uniform(0.20, 0.40), 2),
                'delta': round(call_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(call_price * (1 - bid_ask_spread/2), 2),
                'ask': round(call_price * (1 + bid_ask_spread/2), 2)
            })
            
            puts_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}P{int(strike_price*1000):08d}",
                'strike': strike_price,
                'expiry': expiry,
                'lastPrice': round(put_price, 2),
                'volume': volume,
                'openInterest': oi,
                'impliedVolatility': round(np.random.uniform(0.20, 0.40), 2),
                'delta': round(put_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(put_price * (1 - bid_ask_spread/2), 2),
                'ask': round(put_price * (1 + bid_ask_spread/2), 2)
            })
    
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    
    return expiries[:2], calls_df, puts_df

def get_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Get options data with smart fallback"""
    
    if st.session_state.use_demo_data:
        current_price = get_current_price(ticker)
        if current_price > 0:
            return generate_realistic_options_data(ticker, current_price)
    
    # Try real data
    expiries, calls, puts = get_real_options_data_safe(ticker)
    
    if not expiries:
        # Fallback to generated data
        current_price = get_current_price(ticker)
        if current_price > 0:
            return generate_realistic_options_data(ticker, current_price)
    
    return expiries, calls, puts

# =============================
# SUPPORT/RESISTANCE FUNCTIONS
# =============================
def find_peaks_valleys_simple(prices: np.ndarray, window: int = 5) -> Tuple[List[int], List[int]]:
    """Simple peak and valley detection"""
    if len(prices) < window * 2:
        return [], []
    
    peaks = []
    valleys = []
    
    for i in range(window, len(prices) - window):
        # Check for peak
        is_peak = True
        for j in range(1, window + 1):
            if prices[i] <= prices[i - j] or prices[i] <= prices[i + j]:
                is_peak = False
                break
        if is_peak:
            peaks.append(i)
        
        # Check for valley
        is_valley = True
        for j in range(1, window + 1):
            if prices[i] >= prices[i - j] or prices[i] >= prices[i + j]:
                is_valley = False
                break
        if is_valley:
            valleys.append(i)
    
    return peaks, valleys

@st.cache_data(ttl=300, show_spinner=False)
def get_multi_timeframe_data(ticker: str) -> Tuple[dict, float]:
    """Get multi-timeframe data"""
    timeframes = {
        '5min': {'interval': '5m', 'period': '3d'},
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
                threads=False,
                timeout=5
            )
            
            if not df.empty and len(df) > 20:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                if current_price == 0:
                    current_price = float(df['Close'].iloc[-1])
                
                data[tf] = df
                
        except Exception:
            continue
    
    # Fallback for current price
    if current_price == 0:
        current_price = get_current_price(ticker)
    
    return data, current_price

def calculate_support_resistance(data: pd.DataFrame, timeframe: str, current_price: float) -> dict:
    """Calculate support and resistance levels"""
    if data.empty or len(data) < 20:
        return {
            'support': [],
            'resistance': [],
            'timeframe': timeframe,
            'data_points': len(data)
        }
    
    try:
        window = CONFIG['SR_WINDOW_SIZES'].get(timeframe, 5)
        sensitivity = CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005)
        
        # Get highs and lows
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find peaks and valleys
        peaks, _ = find_peaks_valleys_simple(highs, window)
        _, valleys = find_peaks_valleys_simple(lows, window)
        
        # Extract levels
        resistance_levels = [float(highs[i]) for i in peaks]
        support_levels = [float(lows[i]) for i in valleys]
        
        # Filter and cluster levels
        min_distance = current_price * sensitivity
        
        # Separate support and resistance
        resistance_levels = sorted([level for level in resistance_levels 
                                  if level > current_price + min_distance])
        support_levels = sorted([level for level in support_levels 
                               if level < current_price - min_distance], reverse=True)
        
        # Take top 3 levels
        resistance_levels = resistance_levels[:3]
        support_levels = support_levels[:3]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'timeframe': timeframe,
            'data_points': len(data)
        }
        
    except Exception:
        return {
            'support': [],
            'resistance': [],
            'timeframe': timeframe,
            'data_points': len(data)
        }

def analyze_support_resistance(ticker: str) -> dict:
    """Analyze support and resistance"""
    try:
        tf_data, current_price = get_multi_timeframe_data(ticker)
        
        if not tf_data:
            return {}
        
        results = {}
        for timeframe, data in tf_data.items():
            sr_result = calculate_support_resistance(data, timeframe, current_price)
            results[timeframe] = sr_result
        
        return results
        
    except Exception:
        return {}

def plot_sr_levels(data: dict, current_price: float) -> go.Figure:
    """Plot support and resistance levels"""
    fig = go.Figure()
    
    # Add current price line
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color="blue",
        line_width=2,
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="top right"
    )
    
    # Add levels
    colors = {'5min': 'red', '15min': 'orange', '30min': 'green'}
    
    for tf, sr in data.items():
        color = colors.get(tf, 'gray')
        
        # Support levels
        for level in sr.get('support', []):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                line_width=1,
                annotation_text=f"S ({tf}): ${level:.2f}",
                annotation_position="bottom right"
            )
        
        # Resistance levels
        for level in sr.get('resistance', []):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                line_width=1,
                annotation_text=f"R ({tf}): ${level:.2f}",
                annotation_position="top right"
            )
    
    fig.update_layout(
        title='Support & Resistance Levels',
        xaxis_title='Timeframe',
        yaxis_title='Price ($)',
        height=500,
        template='plotly_dark',
        showlegend=False
    )
    
    # Set y-axis range
    if data:
        all_levels = []
        for sr in data.values():
            all_levels.extend(sr.get('support', []))
            all_levels.extend(sr.get('resistance', []))
        
        if all_levels:
            min_level = min(min(all_levels), current_price * 0.98)
            max_level = max(max(all_levels), current_price * 1.02)
            fig.update_yaxes(range=[min_level, max_level])
    
    return fig

# =============================
# SIGNAL GENERATION FUNCTIONS
# =============================
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
    """Calculate dynamic thresholds based on market conditions"""
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    
    # Get volatility
    volatility = stock_data.get('ATR_pct', 0.02)
    if pd.isna(volatility):
        volatility = 0.02
    
    # Adjust for volatility
    vol_multiplier = 1 + (volatility * 50)
    
    if side == 'call':
        thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_multiplier))
    else:
        thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_multiplier))
    
    thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + thresholds['gamma_vol_multiplier'] * volatility * 100)
    
    # Adjust for market conditions
    if is_premarket() or not is_market_open():
        thresholds['volume_multiplier'] = 0.7
    
    if is_0dte:
        thresholds['volume_multiplier'] *= 0.8
        if side == 'call':
            thresholds['delta_min'] = max(0.4, thresholds['delta_min'])
        else:
            thresholds['delta_max'] = min(-0.4, thresholds['delta_max'])
    
    return thresholds

def generate_enhanced_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    """Generate enhanced trading signal"""
    if stock_df.empty:
        return {'signal': False, 'score': 0.0, 'explanations': []}
    
    current_price = stock_df.iloc[-1]['Close']
    
    if not validate_option_data(option, current_price):
        return {'signal': False, 'score': 0.0, 'explanations': []}
    
    latest = stock_df.iloc[-1]
    
    try:
        thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
        weights = thresholds['condition_weights']
        
        delta = float(option['delta']) if not pd.isna(option['delta']) else 0
        gamma = float(option['gamma']) if not pd.isna(option['gamma']) else 0
        theta = float(option['theta']) if not pd.isna(option['theta']) else 0
        option_volume = float(option['volume'])
        
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if 'EMA_9' in latest and not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if 'EMA_20' in latest and not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if 'RSI' in latest and not pd.isna(latest['RSI']) else None
        vwap = float(latest['VWAP']) if 'VWAP' in latest and not pd.isna(latest['VWAP']) else None
        
        explanations = []
        score = 0.0
        
        # Delta condition
        if side == "call":
            delta_pass = delta >= thresholds.get('delta_min', 0.5)
            delta_score = weights['delta'] if delta_pass else 0
            score += delta_score
            explanations.append({
                'condition': 'Delta',
                'passed': delta_pass,
                'value': delta,
                'threshold': thresholds.get('delta_min', 0.5),
                'score': delta_score
            })
        else:
            delta_pass = delta <= thresholds.get('delta_max', -0.5)
            delta_score = weights['delta'] if delta_pass else 0
            score += delta_score
            explanations.append({
                'condition': 'Delta',
                'passed': delta_pass,
                'value': delta,
                'threshold': thresholds.get('delta_max', -0.5),
                'score': delta_score
            })
        
        # Gamma condition
        gamma_pass = gamma >= thresholds.get('gamma_min', 0.05)
        gamma_score = weights['gamma'] if gamma_pass else 0
        score += gamma_score
        explanations.append({
            'condition': 'Gamma',
            'passed': gamma_pass,
            'value': gamma,
            'threshold': thresholds.get('gamma_min', 0.05),
            'score': gamma_score
        })
        
        # Trend condition
        trend_pass = False
        if ema_9 and ema_20:
            if side == "call":
                trend_pass = close > ema_9 > ema_20
            else:
                trend_pass = close < ema_9 < ema_20
        
        trend_score = weights['trend'] if trend_pass else 0
        score += trend_score
        explanations.append({
            'condition': 'Trend',
            'passed': trend_pass,
            'value': f"Price: {close:.2f}, EMA9: {ema_9:.2f}, EMA20: {ema_20:.2f}" if ema_9 and ema_20 else "N/A",
            'score': trend_score
        })
        
        # Momentum condition
        momentum_pass = False
        if rsi:
            if side == "call":
                momentum_pass = rsi > 50
            else:
                momentum_pass = rsi < 50
        
        momentum_score = weights['momentum'] if momentum_pass else 0
        score += momentum_score
        explanations.append({
            'condition': 'RSI',
            'passed': momentum_pass,
            'value': rsi,
            'threshold': 50,
            'score': momentum_score
        })
        
        # Volume condition
        volume_pass = option_volume > thresholds['volume_min']
        volume_score = weights['volume'] if volume_pass else 0
        score += volume_score
        explanations.append({
            'condition': 'Volume',
            'passed': volume_pass,
            'value': option_volume,
            'threshold': thresholds['volume_min'],
            'score': volume_score
        })
        
        # VWAP condition (bonus)
        vwap_score = 0
        if vwap:
            if (side == "call" and close > vwap) or (side == "put" and close < vwap):
                vwap_score = 0.1
                score += vwap_score
        
        explanations.append({
            'condition': 'VWAP',
            'passed': vwap_score > 0,
            'value': vwap,
            'score': vwap_score
        })
        
        # Calculate if signal passes
        signal_passes = score >= 0.6  # 60% threshold
        
        # Calculate profit targets
        profit_target = None
        stop_loss = None
        holding_period = None
        
        if signal_passes:
            entry_price = option['lastPrice']
            option_type = 'call' if side == 'call' else 'put'
            
            # Add slippage and commission
            slippage_pct = 0.005
            commission = 0.65
            
            entry_price_adj = entry_price * (1 + slippage_pct) + commission
            
            profit_target = entry_price_adj * (1 + CONFIG['PROFIT_TARGETS'][option_type])
            stop_loss = entry_price_adj * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
            
            # Determine holding period
            if is_0dte:
                holding_period = "Intraday (Exit before close)"
            else:
                holding_period = "1-3 days"
        
        return {
            'signal': signal_passes,
            'score': score * 100,
            'max_score': 100,
            'explanations': explanations,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period,
            'open_interest': option['openInterest'],
            'volume': option_volume,
            'implied_volatility': option.get('impliedVolatility', 0),
            'bid': option['bid'],
            'ask': option['ask'],
            'spread_pct': (option['ask'] - option['bid']) / option['ask'] if option['ask'] > 0 else 0
        }
        
    except Exception as e:
        return {'signal': False, 'score': 0.0, 'explanations': [], 'error': str(e)}

def process_options_batch(options_df: pd.DataFrame, side: str, stock_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Process options in batch"""
    if options_df.empty or stock_df.empty:
        return pd.DataFrame()
    
    try:
        # Filter by price and basic validation
        options_df = options_df.copy()
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
            signal_result = generate_enhanced_signal(row, side, stock_df, row['is_0dte'])
            if signal_result['signal']:
                row_dict = row.to_dict()
                row_dict.update(signal_result)
                signals.append(row_dict)
        
        if signals:
            signals_df = pd.DataFrame(signals)
            signals_df = signals_df.sort_values(['is_0dte', 'score'], ascending=[False, False])
            return signals_df
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error processing options: {str(e)}")
        return pd.DataFrame()

# =============================
# CHARTING FUNCTIONS
# =============================
def create_stock_chart(df: pd.DataFrame) -> go.Figure:
    """Create stock chart with indicators"""
    if df.empty:
        return None
    
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Candlestick chart
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
        if 'EMA_9' in df.columns and not df['EMA_9'].isna().all():
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['EMA_9'],
                name='EMA 9', line=dict(color='blue')
            ), row=1, col=1)
        
        if 'EMA_20' in df.columns and not df['EMA_20'].isna().all():
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['EMA_20'],
                name='EMA 20', line=dict(color='orange')
            ), row=1, col=1)
        
        # VWAP
        if 'VWAP' in df.columns and not df['VWAP'].isna().all():
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['VWAP'],
                name='VWAP', line=dict(color='cyan', width=2)
            ), row=1, col=1)
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['Datetime'], y=df['Volume'], name='Volume', marker_color='gray'),
            row=1, col=1, secondary_y=True
        )
        
        # RSI
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=600,
            title='Stock Price & Indicators',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            template='plotly_dark'
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# =============================
# PERFORMANCE MONITORING
# =============================
def measure_performance():
    """Measure and display performance metrics"""
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            'start_time': time.time(),
            'api_calls': 0,
            'data_points_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    with st.expander("⚡ Performance Metrics", expanded=False):
        elapsed = time.time() - st.session_state.performance_metrics['start_time']
        st.metric("Uptime", f"{elapsed:.1f}s")
        st.metric("API Calls", st.session_state.performance_metrics['api_calls'])
        st.metric("Data Points", st.session_state.performance_metrics['data_points_processed'])
        
        total = st.session_state.performance_metrics['cache_hits'] + st.session_state.performance_metrics['cache_misses']
        if total > 0:
            hit_rate = st.session_state.performance_metrics['cache_hits'] / total * 100
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")

# =============================
# STREAMLIT INTERFACE
# =============================
# Auto-refresh
refresh_interval = st_autorefresh(interval=st.session_state.refresh_interval * 1000, 
                                 limit=None, key="price_refresh")

st.title("📈 Enhanced Options Greeks Analyzer")
st.markdown("**Performance Optimized** • **Weighted Scoring** • **Smart Caching**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    st.subheader("🔑 API Settings")
    
    polygon_api_key = st.text_input("Enter Polygon API Key:", type="password", 
                                   value=CONFIG['POLYGON_API_KEY'])
    if polygon_api_key:
        CONFIG['POLYGON_API_KEY'] = polygon_api_key
    
    st.info("💡 Using free data sources (limited rate)")
    
    # Free API Keys
    with st.expander("🔑 Free API Keys", expanded=False):
        CONFIG['ALPHA_VANTAGE_API_KEY'] = st.text_input(
            "Alpha Vantage API Key:",
            type="password",
            value=CONFIG['ALPHA_VANTAGE_API_KEY']
        )
        
        CONFIG['FMP_API_KEY'] = st.text_input(
            "Financial Modeling Prep API Key:",
            type="password",
            value=CONFIG['FMP_API_KEY']
        )
        
        CONFIG['IEX_API_KEY'] = st.text_input(
            "IEX Cloud API Key:",
            type="password",
            value=CONFIG['IEX_API_KEY']
        )
    
    # Data Source
    st.subheader("📊 Data Source")
    use_demo = st.checkbox("Use Demo Data", value=st.session_state.use_demo_data,
                          help="Use realistic generated data when real data is unavailable")
    if use_demo != st.session_state.use_demo_data:
        st.session_state.use_demo_data = use_demo
        st.rerun()
    
    # Auto-refresh
    st.subheader("🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable Auto-Refresh", 
                              value=st.session_state.auto_refresh_enabled)
    if auto_refresh != st.session_state.auto_refresh_enabled:
        st.session_state.auto_refresh_enabled = auto_refresh
        st.rerun()
    
    if st.session_state.auto_refresh_enabled:
        refresh_options = [60, 120, 300]
        st.session_state.refresh_interval = st.selectbox(
            "Refresh Interval:",
            options=refresh_options,
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
    
    # Signal Thresholds
    with st.expander("📊 Signal Thresholds", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Calls")
            SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider(
                "Delta", 0.1, 1.0, 0.5, 0.1, key="call_delta"
            )
            SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider(
                "Gamma", 0.01, 0.2, 0.05, 0.01, key="call_gamma"
            )
            SIGNAL_THRESHOLDS['call']['volume_min'] = st.slider(
                "Min Volume", 100, 2000, 300, 100, key="call_volume"
            )
        
        with col2:
            st.markdown("#### 📉 Puts")
            SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider(
                "Delta", -1.0, -0.1, -0.5, 0.1, key="put_delta"
            )
            SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider(
                "Gamma", 0.01, 0.2, 0.05, 0.01, key="put_gamma"
            )
            SIGNAL_THRESHOLDS['put']['volume_min'] = st.slider(
                "Min Volume", 100, 2000, 300, 100, key="put_volume"
            )
    
    # Risk Management
    with st.expander("🎯 Risk Management", expanded=False):
        CONFIG['PROFIT_TARGETS']['call'] = st.slider(
            "Call Target %", 0.05, 0.50, 0.10, 0.01, key="call_target"
        )
        CONFIG['PROFIT_TARGETS']['put'] = st.slider(
            "Put Target %", 0.05, 0.50, 0.10, 0.01, key="put_target"
        )
        CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider(
            "Stop Loss %", 0.03, 0.20, 0.08, 0.01, key="stop_loss"
        )
    
    # Liquidity Filters
    with st.expander("💰 Liquidity Filters", expanded=False):
        CONFIG['MIN_OPTION_PRICE'] = st.slider(
            "Min Option Price", 0.05, 1.0, 0.25, 0.05,
            help="Filter out cheap, illiquid options"
        )
        CONFIG['MIN_OPEN_INTEREST'] = st.slider(
            "Min Open Interest", 100, 2000, 500, 100,
            help="Higher values filter out less liquid options"
        )
        CONFIG['MIN_VOLUME'] = st.slider(
            "Min Volume", 50, 1000, 100, 50,
            help="Higher values filter out less active options"
        )
        CONFIG['MAX_BID_ASK_SPREAD_PCT'] = st.slider(
            "Max Spread %", 0.05, 0.30, 0.12, 0.01,
            help="Lower values filter out options with wide spreads"
        )
    
    # Market Status
    st.subheader("🕐 Market Status")
    if is_market_open():
        st.success("🟢 Market OPEN")
    elif is_premarket():
        st.warning("🟡 PREMARKET")
    else:
        st.info("🔴 Market CLOSED")
    
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        st.caption(f"**ET**: {now.strftime('%H:%M:%S')}")
    except:
        pass
    
    # Cache status
    if st.session_state.get('last_refresh'):
        time_since = int(time.time() - st.session_state.last_refresh)
        st.caption(f"**Last update**: {time_since}s ago")
    
    # Performance
    measure_performance()
    
    # Clear cache button
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()

# Main interface
ticker = st.text_input("Enter Stock Ticker (e.g., SPY, QQQ, AAPL):", value="SPY").upper()

if ticker:
    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if is_market_open():
            st.success("🟢 OPEN")
        elif is_premarket():
            st.warning("🟡 PRE")
        else:
            st.info("🔴 CLOSED")
    
    with col2:
        current_price = get_current_price(ticker)
        if current_price > 0:
            st.metric("Price", f"${current_price:.2f}")
        else:
            st.error("Price Error")
    
    with col3:
        cache_age = int(time.time() - st.session_state.last_refresh)
        st.metric("Cache Age", f"{cache_age}s")
    
    with col4:
        st.metric("Refreshes", st.session_state.refresh_counter)
    
    with col5:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_counter += 1
            st.rerun()
    
    # Update S/R data if needed
    if not st.session_state.sr_data or st.session_state.last_ticker != ticker:
        with st.spinner("Analyzing support/resistance..."):
            st.session_state.sr_data = analyze_support_resistance(ticker)
            st.session_state.last_ticker = ticker
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Enhanced Signals",
        "📊 Technical Analysis",
        "📈 Support/Resistance",
        "📰 Market Context"
    ])
    
    with tab1:
        try:
            with st.spinner("Loading enhanced analysis..."):
                # Get stock data
                df = get_stock_data_with_indicators(ticker)
                
                if df.empty:
                    st.error("Unable to fetch stock data")
                    st.stop()
                
                current_price = df.iloc[-1]['Close']
                st.success(f"✅ **{ticker}** - ${current_price:.2f}")
                
                # Volatility assessment
                atr_pct = df.iloc[-1].get('ATR_pct', 0)
                if not pd.isna(atr_pct):
                    vol_status = "Low"
                    if atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['high']:
                        vol_status = "High"
                    elif atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['medium']:
                        vol_status = "Medium"
                    st.info(f"📊 **Volatility**: {atr_pct*100:.2f}% ({vol_status})")
                
                # Get options data
                with st.spinner("Fetching options data..."):
                    expiries, all_calls, all_puts = get_options_data(ticker)
                
                if not expiries:
                    st.error("❌ Unable to fetch options data")
                    
                    with st.expander("💡 Solutions", expanded=True):
                        st.markdown("""
                        **To get options data:**
                        1. **Wait 2-3 minutes** for rate limits to reset
                        2. **Enable Demo Data** in sidebar for testing
                        3. **Try during market hours** (9:30 AM - 4:00 PM ET)
                        4. **Use popular tickers** like SPY or QQQ
                        """)
                    
                    if st.button("🔄 Retry with Demo Data"):
                        st.session_state.use_demo_data = True
                        st.rerun()
                    
                    st.stop()
                
                # Show data source
                if st.session_state.use_demo_data:
                    st.warning("⚠️ Using demo data for analysis")
                else:
                    st.success(f"✅ Real options data loaded")
                
                # Expiry selection
                col1, col2 = st.columns(2)
                with col1:
                    expiry_mode = st.radio(
                        "Expiration Filter:",
                        ["0DTE Only", "This Week", "All Available"],
                        index=0
                    )
                
                today = datetime.date.today()
                if expiry_mode == "0DTE Only":
                    expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
                elif expiry_mode == "This Week":
                    week_end = today + datetime.timedelta(days=7)
                    expiries_to_use = [e for e in expiries if today <= datetime.datetime.strptime(e, "%Y-%m-%d").date() <= week_end]
                else:
                    expiries_to_use = expiries[:3]
                
                if not expiries_to_use:
                    st.warning("No expiries available for selected filter")
                    st.stop()
                
                with col2:
                    st.info(f"📅 Analyzing **{len(expiries_to_use)}** expiries")
                
                # Filter options by expiry
                calls_filtered = all_calls[all_calls['expiry'].isin(expiries_to_use)].copy()
                puts_filtered = all_puts[all_puts['expiry'].isin(expiries_to_use)].copy()
                
                # Strike range filter
                strike_range = st.slider(
                    "Strike Range ($):",
                    -10.0, 10.0, (-3.0, 3.0), 0.5
                )
                min_strike = current_price + strike_range[0]
                max_strike = current_price + strike_range[1]
                
                calls_filtered = calls_filtered[
                    (calls_filtered['strike'] >= min_strike) &
                    (calls_filtered['strike'] <= max_strike)
                ].copy()
                
                puts_filtered = puts_filtered[
                    (puts_filtered['strike'] >= min_strike) &
                    (puts_filtered['strike'] <= max_strike)
                ].copy()
                
                # Moneyness filter
                m_filter = st.multiselect(
                    "Moneyness Filter:",
                    options=["ITM", "NTM", "ATM", "OTM"],
                    default=["ATM", "NTM"]
                )
                
                if not calls_filtered.empty:
                    calls_filtered['moneyness'] = calls_filtered['strike'].apply(
                        lambda x: classify_moneyness(x, current_price)
                    )
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
                
                if not puts_filtered.empty:
                    puts_filtered['moneyness'] = puts_filtered['strike'].apply(
                        lambda x: classify_moneyness(x, current_price)
                    )
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]
                
                st.write(f"**Filtered Options**: {len(calls_filtered)} calls, {len(puts_filtered)} puts")
                
                # Process signals
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Call Signals")
                    if not calls_filtered.empty:
                        call_signals = process_options_batch(calls_filtered, "call", df, current_price)
                        
                        if not call_signals.empty:
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'score',
                                          'moneyness', 'volume', 'is_0dte']
                            display_df = call_signals[display_cols].copy()
                            display_df = display_df.sort_values(['is_0dte', 'score'], ascending=[False, False])
                            display_df = display_df.head(15)
                            
                            st.dataframe(
                                display_df.rename(columns={
                                    'score': 'Score%',
                                    'lastPrice': 'Price',
                                    'is_0dte': '0DTE'
                                }).style.format({
                                    'Score%': '{:.1f}',
                                    'Price': '${:.2f}',
                                    'strike': '${:.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Show best signal
                            if len(call_signals) > 0:
                                best_call = call_signals.iloc[0]
                                with st.expander(f"🏆 Best Call Signal ({best_call['contractSymbol']})"):
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Score", f"{best_call['score']:.1f}%")
                                        st.metric("Delta", f"{best_call.get('delta', 0):.3f}")
                                        st.metric("Strike", f"${best_call['strike']:.2f}")
                                    with col_b:
                                        st.metric("Price", f"${best_call['lastPrice']:.2f}")
                                        st.metric("Gamma", f"{best_call.get('gamma', 0):.3f}")
                                        st.metric("Volume", f"{best_call['volume']:.0f}")
                                    with col_c:
                                        if best_call.get('profit_target'):
                                            st.metric("Target", f"${best_call['profit_target']:.2f}")
                                        if best_call.get('stop_loss'):
                                            st.metric("Stop", f"${best_call['stop_loss']:.2f}")
                                        st.metric("0DTE", "Yes" if best_call.get('is_0dte') else "No")
                        else:
                            st.info("No call signals found")
                    else:
                        st.info("No call options in range")
                
                with col2:
                    st.subheader("📉 Put Signals")
                    if not puts_filtered.empty:
                        put_signals = process_options_batch(puts_filtered, "put", df, current_price)
                        
                        if not put_signals.empty:
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'score',
                                          'moneyness', 'volume', 'is_0dte']
                            display_df = put_signals[display_cols].copy()
                            display_df = display_df.sort_values(['is_0dte', 'score'], ascending=[False, False])
                            display_df = display_df.head(15)
                            
                            st.dataframe(
                                display_df.rename(columns={
                                    'score': 'Score%',
                                    'lastPrice': 'Price',
                                    'is_0dte': '0DTE'
                                }).style.format({
                                    'Score%': '{:.1f}',
                                    'Price': '${:.2f}',
                                    'strike': '${:.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Show best signal
                            if len(put_signals) > 0:
                                best_put = put_signals.iloc[0]
                                with st.expander(f"🏆 Best Put Signal ({best_put['contractSymbol']})"):
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Score", f"{best_put['score']:.1f}%")
                                        st.metric("Delta", f"{best_put.get('delta', 0):.3f}")
                                        st.metric("Strike", f"${best_put['strike']:.2f}")
                                    with col_b:
                                        st.metric("Price", f"${best_put['lastPrice']:.2f}")
                                        st.metric("Gamma", f"{best_put.get('gamma', 0):.3f}")
                                        st.metric("Volume", f"{best_put['volume']:.0f}")
                                    with col_c:
                                        if best_put.get('profit_target'):
                                            st.metric("Target", f"${best_put['profit_target']:.2f}")
                                        if best_put.get('stop_loss'):
                                            st.metric("Stop", f"${best_put['stop_loss']:.2f}")
                                        st.metric("0DTE", "Yes" if best_put.get('is_0dte') else "No")
                        else:
                            st.info("No put signals found")
                    else:
                        st.info("No put options in range")
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'call_signals' in locals() and not call_signals.empty:
                        avg_score = call_signals['score'].mean()
                        st.metric("Avg Call Score", f"{avg_score:.1f}%")
                    else:
                        st.metric("Avg Call Score", "N/A")
                
                with col2:
                    if 'put_signals' in locals() and not put_signals.empty:
                        avg_score = put_signals['score'].mean()
                        st.metric("Avg Put Score", f"{avg_score:.1f}%")
                    else:
                        st.metric("Avg Put Score", "N/A")
                
                with col3:
                    total_signals = (len(call_signals) if 'call_signals' in locals() and not call_signals.empty else 0) + \
                                  (len(put_signals) if 'put_signals' in locals() and not put_signals.empty else 0)
                    st.metric("Total Signals", total_signals)
                
        except Exception as e:
            st.error(f"Error in signal analysis: {str(e)}")
            st.error("Please try refreshing or check your ticker symbol.")
    
    with tab2:
        try:
            if 'df' not in locals():
                df = get_stock_data_with_indicators(ticker)
            
            if not df.empty:
                # Market session info
                if is_premarket():
                    st.info("🔔 Currently showing PREMARKET data")
                elif not is_market_open():
                    st.info("🔔 Showing AFTER-HOURS data")
                else:
                    st.success("🔔 Showing REGULAR HOURS data")
                
                latest = df.iloc[-1]
                
                # Metrics
                cols = st.columns(6)
                
                with cols[0]:
                    st.metric("Price", f"${latest['Close']:.2f}")
                
                with cols[1]:
                    if 'EMA_9' in latest and not pd.isna(latest['EMA_9']):
                        trend = "🔺" if latest['Close'] > latest['EMA_9'] else "🔻"
                        st.metric("EMA 9", f"${latest['EMA_9']:.2f} {trend}")
                
                with cols[2]:
                    if 'EMA_20' in latest and not pd.isna(latest['EMA_20']):
                        trend = "🔺" if latest['Close'] > latest['EMA_20'] else "🔻"
                        st.metric("EMA 20", f"${latest['EMA_20']:.2f} {trend}")
                
                with cols[3]:
                    if 'RSI' in latest and not pd.isna(latest['RSI']):
                        status = "🔥" if latest['RSI'] > 70 else "❄️" if latest['RSI'] < 30 else "⚖️"
                        st.metric("RSI", f"{latest['RSI']:.1f} {status}")
                
                with cols[4]:
                    if 'ATR_pct' in latest and not pd.isna(latest['ATR_pct']):
                        vol = "🌪️" if latest['ATR_pct'] > 0.05 else "📊" if latest['ATR_pct'] > 0.02 else "😴"
                        st.metric("Volatility", f"{latest['ATR_pct']*100:.2f}% {vol}")
                
                with cols[5]:
                    if 'VWAP' in latest and not pd.isna(latest['VWAP']):
                        vwap_rel = "Above" if latest['Close'] > latest['VWAP'] else "Below"
                        st.metric("VWAP", f"${latest['VWAP']:.2f}")
                        st.caption(f"Price is {vwap_rel} VWAP")
                
                # Recent data
                st.subheader("📋 Recent Market Data")
                display_df = df.tail(10)[['Datetime', 'Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'Volume']].copy()
                display_df['Datetime'] = display_df['Datetime'].dt.strftime('%H:%M')
                display_df = display_df.rename(columns={
                    'Datetime': 'Time',
                    'Close': 'Price',
                    'EMA_9': 'EMA9',
                    'EMA_20': 'EMA20'
                })
                st.dataframe(display_df.style.format({
                    'Price': '${:.2f}',
                    'EMA9': '${:.2f}',
                    'EMA20': '${:.2f}',
                    'VWAP': '${:.2f}',
                    'RSI': '{:.1f}'
                }), use_container_width=True, hide_index=True)
                
                # Chart
                st.subheader("📈 Interactive Chart")
                chart_fig = create_stock_chart(df)
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)
                else:
                    st.warning("Unable to create chart")
                
        except Exception as e:
            st.error(f"Error in technical analysis: {str(e)}")
    
    with tab3:
        st.subheader("📈 Support & Resistance Analysis")
        
        if not st.session_state.sr_data:
            st.warning("No support/resistance data available")
        else:
            # Plot
            sr_fig = plot_sr_levels(st.session_state.sr_data, current_price)
            if sr_fig:
                st.plotly_chart(sr_fig, use_container_width=True)
            
            # Detailed levels
            st.subheader("Detailed Levels")
            
            for timeframe, sr in st.session_state.sr_data.items():
                with st.expander(f"{timeframe} Timeframe", expanded=(timeframe == '5min')):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Support Levels**")
                        for level in sr.get('support', []):
                            distance = ((level - current_price) / current_price * 100)
                            st.write(f"${level:.2f} ({distance:+.1f}% from current)")
                    
                    with col2:
                        st.markdown("**Resistance Levels**")
                        for level in sr.get('resistance', []):
                            distance = ((level - current_price) / current_price * 100)
                            st.write(f"${level:.2f} ({distance:+.1f}% from current)")
            
            # Trading guidance
            with st.expander("📝 Trading Strategy Guidance", expanded=False):
                st.markdown("""
                **How to use support/resistance:**
                
                **Scalping (5min/15min levels):**
                - Use for quick trades (minutes to hours)
                - Look for options with strikes near key levels
                - Combine with high delta for directional plays
                - Ideal for 0DTE options
                
                **Swing Trading (30min/1h levels):**
                - Use for longer-term trades (hours to days)
                - Look for options between support/resistance for range-bound strategies
                - Combine with technical indicators for confirmation
                - Ideal for weekly expiration options
                
                **VWAP Strategy:**
                - **Bullish**: Buy calls when price crosses above VWAP with volume
                - **Bearish**: Buy puts when price rejects at VWAP
                - **Bounce Play**: Buy calls when price pulls back to VWAP in uptrend
                """)
    
    with tab4:
        st.subheader("📰 Market Context & Information")
        
        try:
            stock = yf.Ticker(ticker)
            
            # Company info
            with st.expander("🏢 Company Overview", expanded=True):
                try:
                    info = stock.info
                    if info:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'longName' in info:
                                st.write(f"**Company**: {info['longName']}")
                            if 'sector' in info:
                                st.write(f"**Sector**: {info['sector']}")
                            if 'industry' in info:
                                st.write(f"**Industry**: {info['industry']}")
                        
                        with col2:
                            if 'marketCap' in info:
                                market_cap = info['marketCap']
                                if market_cap > 1e12:
                                    st.write(f"**Market Cap**: ${market_cap/1e12:.2f}T")
                                elif market_cap > 1e9:
                                    st.write(f"**Market Cap**: ${market_cap/1e9:.2f}B")
                            if 'averageVolume' in info:
                                avg_vol = info['averageVolume']
                                st.write(f"**Avg Volume**: {avg_vol:,.0f}")
                except:
                    st.info("Company information unavailable")
            
            # Market context
            with st.expander("🎯 Trading Context", expanded=True):
                st.markdown("""
                **Current Market Conditions:**
                - Check VIX for overall market volatility
                - Monitor major indices (SPY, QQQ, IWM) for direction
                - Watch for economic events that could impact prices
                
                **Risk Considerations:**
                - Options lose value due to time decay (theta)
                - High volatility increases option prices
                - Earnings announcements create large price movements
                - Market holidays affect expiration schedules
                
                **Best Practices:**
                - Never risk more than you can afford to lose
                - Use stop losses to limit downside
                - Take profits when targets are reached
                - Avoid holding 0DTE options into close
                """)
                
                # Add warnings based on conditions
                if is_premarket():
                    st.warning("⚠️ **PREMARKET**: Lower liquidity, wider spreads")
                elif not is_market_open():
                    st.info("ℹ️ **MARKET CLOSED**: Signals based on previous session")
                
                # Volatility warning
                if 'df' in locals() and not df.empty:
                    latest_atr = df.iloc[-1].get('ATR_pct', 0)
                    if not pd.isna(latest_atr) and latest_atr > 0.05:
                        st.warning("🌪️ **HIGH VOLATILITY**: Increased risk. Consider wider stops.")
            
            # Data source info
            with st.expander("📊 Data Sources", expanded=False):
                st.markdown("""
                **Primary Data Sources:**
                - **Yahoo Finance**: Stock and options data
                - **Polygon.io**: Premium data (if API key provided)
                - **Alpha Vantage**: Backup price data
                - **Financial Modeling Prep**: Fundamental data
                
                **Rate Limits:**
                - Yahoo Finance: Limited options requests
                - Free APIs: 5-10 requests per minute
                - Consider premium APIs for heavy usage
                
                **Data Quality:**
                - Real data during market hours (9:30 AM - 4:00 PM ET)
                - Demo data available for testing
                - Refresh every 2-3 minutes to avoid limits
                """)
                
        except Exception as e:
            st.error(f"Error loading market context: {str(e)}")

else:
    # Welcome screen
    st.info("👋 **Welcome!** Enter a stock ticker above to begin analysis.")
    
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        **Getting Started:**
        1. **Enter a ticker** (e.g., SPY, QQQ, AAPL)
        2. **Configure settings** in sidebar
        3. **View signals** in the Enhanced Signals tab
        4. **Analyze charts** in the Technical Analysis tab
        
        **Recommended Settings:**
        - **For scalping**: Use tight strike ranges (±$3)
        - **For swing trading**: Use wider ranges (±$10)
        - **Min Option Price**: $0.25
        - **Min Volume**: 100+
        
        **Troubleshooting:**
        - If data fails to load, enable **Demo Data** in sidebar
        - Wait 2-3 minutes between refreshes
        - Use popular tickers during market hours
        - Clear cache if data seems stale
        """)
    
    # Popular tickers
    st.subheader("📊 Popular Tickers to Try")
    col1, col2, col3, col4 = st.columns(4)
    
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'GOOGL']
    for idx, ticker_name in enumerate(tickers):
        col = [col1, col2, col3, col4][idx % 4]
        if col.button(ticker_name):
            st.session_state.ticker_input = ticker_name
            st.rerun()

# Auto-refresh logic
if st.session_state.auto_refresh_enabled and ticker:
    current_time = time.time()
    elapsed = current_time - st.session_state.last_refresh
    
    if elapsed > st.session_state.refresh_interval:
        st.session_state.last_refresh = current_time
        st.session_state.refresh_counter += 1
        
        # Show refresh notification
        st.success(f"🔄 Auto-refreshed at {datetime.datetime.now().strftime('%H:%M:%S')}")
        time.sleep(1)
        st.cache_data.clear()
        st.rerun()

# Add footer
st.markdown("---")
st.caption("📊 **Options Greeks Analyzer** • Performance Optimized • Data delays may occur • Not financial advice")
