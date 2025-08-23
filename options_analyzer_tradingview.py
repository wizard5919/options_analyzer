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
from ta.volatility import AverageTrueRange, KeltnerChannel, BollingerBands
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from polygon import RESTClient
from streamlit_autorefresh import st_autorefresh
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Support/Resistance analysis will use simplified method.")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Enhanced page configuration with dark theme
st.set_page_config(
    page_title="📈 TradingView-Style Options Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/wizard5919/options_analyzer',
        'Report a bug': 'https://github.com/wizard5919/options_analyzer/issues',
        'About': "# TradingView-Style Options Analyzer\nProfessional options analysis platform"
    }
)

# Custom CSS for TradingView-style dark theme
st.markdown("""
<style>
    /* Main theme colors */
    .stApp {
        background-color: #0d1421;
        color: #d1d4dc;
    }
    
    /* Navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e222d;
        border-bottom: 1px solid #363a45;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e222d;
        color: #787b86;
        border: none;
        padding: 12px 24px;
        font-weight: 500;
        border-radius: 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a2e39;
        color: #ffffff;
        border-bottom: 2px solid #2962ff;
    }
    
    /* Chart container */
    .chart-container {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid #363a45;
    }
    
    /* Metrics cards */
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        margin: 4px;
        border: 1px solid #363a45;
        text-align: center;
    }
    
    /* Signals panel */
    .signals-panel {
        background-color: #131722;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #363a45;
        min-height: 600px;
    }
    
    /* Timeframe buttons */
    .timeframe-btn {
        background-color: #2a2e39;
        color: #787b86;
        border: 1px solid #363a45;
        padding: 6px 12px;
        margin: 2px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
    }
    
    .timeframe-btn:hover {
        background-color: #363a45;
        color: #ffffff;
    }
    
    .timeframe-btn.active {
        background-color: #2962ff;
        color: #ffffff;
        border-color: #2962ff;
    }
    
    /* Market status indicators */
    .market-open { color: #4caf50; }
    .market-closed { color: #f44336; }
    .market-premarket { color: #ff9800; }
    
    /* Signal strength indicators */
    .signal-strong { color: #4caf50; font-weight: bold; }
    .signal-moderate { color: #ff9800; }
    .signal-weak { color: #f44336; }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e222d;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #363a45;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #434651;
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh for real-time updates
refresh_interval = st_autorefresh(interval=5000, limit=None, key="price_refresh")

# =============================
# ENHANCED CONFIGURATION & CONSTANTS
# =============================
CONFIG = {
    'POLYGON_API_KEY': '',
    'ALPHA_VANTAGE_API_KEY': '',
    'FMP_API_KEY': '',
    'IEX_API_KEY': '',
    'MAX_RETRIES': 5,
    'RETRY_DELAY': 2,
    'DATA_TIMEOUT': 30,
    'MIN_DATA_POINTS': 50,
    'CACHE_TTL': 300,
    'STOCK_CACHE_TTL': 300,
    'RATE_LIMIT_COOLDOWN': 300,
    'MIN_REFRESH_INTERVAL': 60,
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
    'TRADING_HOURS_PER_DAY': 6.5,
    'SR_TIME_WINDOWS': {
        'scalping': ['1min', '5min'],
        'intraday': ['15min', '30min', '1h']
    },
    'SR_SENSITIVITY': {
        '1min': 0.001,
        '5min': 0.002,
        '15min': 0.003,
        '30min': 0.005,
        '1h': 0.008
    },
    'SR_WINDOW_SIZES': {
        '1min': 3,
        '5min': 3,
        '15min': 5,
        '30min': 7,
        '1h': 10
    },
    'LIQUIDITY_THRESHOLDS': {
        'min_open_interest': 100,
        'min_volume': 100,
        'max_bid_ask_spread_pct': 0.1
    },
    'TIMEFRAMES': {
        '5m': {'interval': '5m', 'period': '5d', 'name': '5m'},
        '15m': {'interval': '15m', 'period': '15d', 'name': '15m'},
        '30m': {'interval': '30m', 'period': '30d', 'name': '30m'},
        '1H': {'interval': '1h', 'period': '60d', 'name': '1H'},
        '4H': {'interval': '4h', 'period': '60d', 'name': '4H'},
        '1D': {'interval': '1d', 'period': '1y', 'name': '1D'},
        '1W': {'interval': '1wk', 'period': '2y', 'name': '1W'},
        '1M': {'interval': '1mo', 'period': '5y', 'name': '1M'}
    }
}

# Initialize session state
if 'API_CALL_LOG' not in st.session_state:
    st.session_state.API_CALL_LOG = []
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'selected_timeframe' not in st.session_state:
    st.session_state.selected_timeframe = '5m'
if 'chart_indicators' not in st.session_state:
    st.session_state.chart_indicators = {
        'ema_9': True,
        'ema_20': True,
        'ema_50': True,
        'bollinger': False,
        'volume': True,
        'rsi': True,
        'macd': True
    }
if 'sr_data' not in st.session_state:
    st.session_state.sr_data = {}
if 'signals_data' not in st.session_state:
    st.session_state.signals_data = {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}

# Enhanced signal thresholds
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
        'volume_min': 1000,
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
        'volume_min': 1000,
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
    """Check if market is currently open"""
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

def get_market_status():
    """Get current market status"""
    if is_market_open():
        return "OPEN", "market-open"
    elif is_premarket():
        return "PRE-MARKET", "market-premarket"
    else:
        return "CLOSED", "market-closed"

@st.cache_data(ttl=5, show_spinner=False)
def get_current_price(ticker: str) -> float:
    """Get real-time price from multiple sources"""
    # Try Polygon first if available
    if CONFIG['POLYGON_API_KEY']:
        try:
            client = RESTClient(CONFIG['POLYGON_API_KEY'], trace=False, connect_timeout=0.5)
            trade = client.stocks_equities_last_trade(ticker)
            return float(trade.last.price)
        except Exception:
            pass
    
    # Try Alpha Vantage
    if CONFIG['ALPHA_VANTAGE_API_KEY'] and can_make_request("ALPHA_VANTAGE"):
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={CONFIG['ALPHA_VANTAGE_API_KEY']}"
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            data = response.json()
            if 'Global Quote' in data and '05. price' in data['Global Quote']:
                log_api_request("ALPHA_VANTAGE")
                return float(data['Global Quote']['05. price'])
        except Exception:
            pass
    
    # Yahoo Finance fallback
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception:
        pass
    
    return 0.0

@st.cache_data(ttl=CONFIG['STOCK_CACHE_TTL'], show_spinner=False)
def get_stock_data_with_indicators(ticker: str, timeframe: str = '5m') -> pd.DataFrame:
    """Fetch stock data and compute all indicators"""
    try:
        tf_config = CONFIG['TIMEFRAMES'][timeframe]
        
        data = yf.download(
            ticker,
            period=tf_config['period'],
            interval=tf_config['interval'],
            auto_adjust=True,
            progress=False,
            prepost=True
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return pd.DataFrame()
        
        # Clean data
        data = data.dropna(how='all')
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=required_cols)
        
        if len(data) < CONFIG['MIN_DATA_POINTS']:
            return pd.DataFrame()
        
        # Handle timezone
        eastern = pytz.timezone('US/Eastern')
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc)
        data.index = data.index.tz_convert(eastern)
        
        data = data.reset_index(drop=False)
        data = data.rename(columns={'Date': 'Datetime'})
        
        # Compute indicators
        return compute_all_indicators(data)
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators"""
    if df.empty:
        return df
    
    try:
        df = df.copy()
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()
        
        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=required_cols)
        if df.empty:
            return df
        
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        
        # EMAs
        for period in [9, 20, 50, 200]:
            if len(close) >= period:
                ema = EMAIndicator(close=close, window=period)
                df[f'EMA_{period}'] = ema.ema_indicator()
            else:
                df[f'EMA_{period}'] = np.nan
        
        # Bollinger Bands
        if len(close) >= 20:
            bb = BollingerBands(close=close, window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
        else:
            for col in ['BB_upper', 'BB_middle', 'BB_lower']:
                df[col] = np.nan
        
        # RSI
        if len(close) >= 14:
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
        else:
            df['RSI'] = np.nan
        
        # VWAP
        if 'Datetime' in df.columns:
            df['VWAP'] = calculate_vwap(df)
        else:
            df['VWAP'] = np.nan
        
        # ATR
        if len(close) >= 14:
            atr = AverageTrueRange(high=high, low=low, close=close, window=14)
            df['ATR'] = atr.average_true_range()
            current_price = df['Close'].iloc[-1]
            if current_price > 0:
                df['ATR_pct'] = df['ATR'] / close
            else:
                df['ATR_pct'] = np.nan
        else:
            df['ATR'] = np.nan
            df['ATR_pct'] = np.nan
        
        # MACD
        if len(close) >= 26:
            macd = MACD(close=close)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
        else:
            for col in ['MACD', 'MACD_signal', 'MACD_hist']:
                df[col] = np.nan
        
        # Volume average
        df['avg_vol'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        return df
        
    except Exception as e:
        st.error(f"Error computing indicators: {str(e)}")
        return pd.DataFrame()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP (Volume Weighted Average Price)"""
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Group by date for daily VWAP
        df_with_date = df.copy()
        df_with_date['Date'] = pd.to_datetime(df_with_date['Datetime']).dt.date
        
        vwap_series = pd.Series(index=df.index, dtype=float)
        
        for date, group in df_with_date.groupby('Date'):
            if group.empty:
                continue
            
            tp = typical_price.loc[group.index]
            vol = df['Volume'].loc[group.index]
            
            cumulative_tp = (tp * vol).cumsum()
            cumulative_vol = vol.cumsum()
            
            daily_vwap = np.where(cumulative_vol != 0, cumulative_tp / cumulative_vol, np.nan)
            vwap_series.loc[group.index] = daily_vwap
        
        return vwap_series
        
    except Exception as e:
        st.warning(f"Error calculating VWAP: {str(e)}")
        return pd.Series(index=df.index, dtype=float)

# =============================
# SUPPORT/RESISTANCE FUNCTIONS
# =============================
def find_peaks_valleys_robust(data: np.array, order: int = 5, prominence: float = None) -> Tuple[List[int], List[int]]:
    """Robust peak and valley detection"""
    if len(data) < order * 2 + 1:
        return [], []
    
    try:
        if SCIPY_AVAILABLE and prominence is not None:
            peaks, _ = signal.find_peaks(data, distance=order, prominence=prominence)
            valleys, _ = signal.find_peaks(-data, distance=order, prominence=prominence)
            return peaks.tolist(), valleys.tolist()
        else:
            peaks = []
            valleys = []
            
            for i in range(order, len(data) - order):
                is_peak = True
                for j in range(1, order + 1):
                    if data[i] <= data[i-j] or data[i] <= data[i+j]:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(i)
                
                is_valley = True
                for j in range(1, order + 1):
                    if data[i] >= data[i-j] or data[i] >= data[i+j]:
                        is_valley = False
                        break
                if is_valley:
                    valleys.append(i)
            
            return peaks, valleys
    except Exception as e:
        st.warning(f"Error in peak detection: {str(e)}")
        return [], []

def calculate_support_resistance_levels(df: pd.DataFrame, current_price: float) -> Dict:
    """Calculate support and resistance levels"""
    if df.empty or len(df) < 20:
        return {'support': [], 'resistance': []}
    
    try:
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        # Calculate prominence for peak detection
        price_std = np.std(closes)
        prominence = price_std * 0.3
        
        # Find peaks and valleys
        resistance_indices, support_indices = find_peaks_valleys_robust(highs, order=5, prominence=prominence)
        support_valleys, resistance_peaks = find_peaks_valleys_robust(lows, order=5, prominence=prominence)
        
        # Combine indices
        all_resistance_indices = list(set(resistance_indices + resistance_peaks))
        all_support_indices = list(set(support_indices + support_valleys))
        
        # Extract price levels
        resistance_levels = [float(highs[i]) for i in all_resistance_indices if i < len(highs)]
        support_levels = [float(lows[i]) for i in all_support_indices if i < len(lows)]
        
        # Add VWAP as significant level
        if 'VWAP' in df.columns:
            vwap = df['VWAP'].iloc[-1]
            if not pd.isna(vwap):
                if vwap > current_price:
                    resistance_levels.append(vwap)
                else:
                    support_levels.append(vwap)
        
        # Filter and sort levels
        min_distance = current_price * 0.002  # 0.2% minimum distance
        resistance_levels = sorted([level for level in set(resistance_levels) 
                                  if level > current_price and abs(level - current_price) > min_distance])
        support_levels = sorted([level for level in set(support_levels) 
                               if level < current_price and abs(level - current_price) > min_distance], reverse=True)
        
        return {
            'support': support_levels[:5],  # Top 5 levels
            'resistance': resistance_levels[:5]
        }
        
    except Exception as e:
        st.error(f"Error calculating S/R levels: {str(e)}")
        return {'support': [], 'resistance': []}

# =============================
# OPTIONS DATA FUNCTIONS
# =============================
@st.cache_data(ttl=1800, show_spinner=False)
def get_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Get options data with proper error handling"""
    try:
        stock = yf.Ticker(ticker)
        expiries = list(stock.options) if stock.options else []
        
        if not expiries:
            return [], pd.DataFrame(), pd.DataFrame()
        
        # Get nearest expiry to minimize API calls
        nearest_expiry = expiries[0]
        time.sleep(1)  # Rate limiting
        
        chain = stock.option_chain(nearest_expiry)
        if chain is None:
            return [], pd.DataFrame(), pd.DataFrame()
        
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        
        if calls.empty and puts.empty:
            return [], pd.DataFrame(), pd.DataFrame()
        
        # Add expiry column
        calls['expiry'] = nearest_expiry
        puts['expiry'] = nearest_expiry
        
        # Validate essential columns
        required_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'bid', 'ask']
        if not all(col in calls.columns for col in required_cols):
            return [], pd.DataFrame(), pd.DataFrame()
        if not all(col in puts.columns for col in required_cols):
            return [], pd.DataFrame(), pd.DataFrame()
        
        # Add Greeks if missing
        for df_name, df in [('calls', calls), ('puts', puts)]:
            if 'delta' not in df.columns:
                df['delta'] = np.nan
            if 'gamma' not in df.columns:
                df['gamma'] = np.nan
            if 'theta' not in df.columns:
                df['theta'] = np.nan
        
        return [nearest_expiry], calls, puts
        
    except Exception as e:
        return [], pd.DataFrame(), pd.DataFrame()

def generate_demo_options_data(ticker: str, current_price: float) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Generate realistic demo options data"""
    # Create strike range
    strikes = []
    strike_range = max(5, current_price * 0.1)
    
    if current_price < 50:
        increment = 1
    elif current_price < 200:
        increment = 5
    else:
        increment = 10
    
    start_strike = int((current_price - strike_range) / increment) * increment
    end_strike = int((current_price + strike_range) / increment) * increment
    
    for strike in range(start_strike, end_strike + increment, increment):
        if strike > 0:
            strikes.append(strike)
    
    # Generate expiry dates
    today = datetime.date.today()
    expiries = []
    
    # Add today if weekday (0DTE)
    if today.weekday() < 5:
        expiries.append(today.strftime('%Y-%m-%d'))
    
    # Add next Friday
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + datetime.timedelta(days=days_until_friday)
    expiries.append(next_friday.strftime('%Y-%m-%d'))
    
    # Create options data
    calls_data = []
    puts_data = []
    
    for expiry in expiries:
        expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
        days_to_expiry = (expiry_date - today).days
        is_0dte = days_to_expiry == 0
        
        for strike in strikes:
            # Calculate moneyness
            moneyness = current_price / strike
            
            # Realistic Greeks
            if moneyness > 1.05:  # ITM calls
                call_delta = 0.7 + (moneyness - 1) * 0.2
                put_delta = call_delta - 1
                gamma = 0.02
            elif moneyness > 0.95:  # ATM
                call_delta = 0.5
                put_delta = -0.5
                gamma = 0.08 if is_0dte else 0.05
            else:  # OTM calls
                call_delta = 0.3 - (1 - moneyness) * 0.2
                put_delta = call_delta - 1
                gamma = 0.02
            
            theta = -0.1 if is_0dte else -0.05 if days_to_expiry <= 7 else -0.02
            
            # Realistic pricing
            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = 5 if is_0dte else 10 if days_to_expiry <= 7 else 15
            
            call_price = intrinsic_call + time_value * gamma
            put_price = intrinsic_put + time_value * gamma
            
            volume = 1000 if abs(moneyness - 1) < 0.05 else 500
            
            calls_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}C{strike*1000:08.0f}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(call_price, 2),
                'volume': volume,
                'openInterest': volume // 2,
                'impliedVolatility': 0.25,
                'delta': round(call_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(call_price * 0.98, 2),
                'ask': round(call_price * 1.02, 2)
            })
            
            puts_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}P{strike*1000:08.0f}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(put_price, 2),
                'volume': volume,
                'openInterest': volume // 2,
                'impliedVolatility': 0.25,
                'delta': round(put_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(put_price * 0.98, 2),
                'ask': round(put_price * 1.02, 2)
            })
    
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    
    return expiries, calls_df, puts_df

# =============================
# SIGNAL GENERATION FUNCTIONS
# =============================
def generate_enhanced_signal(option: pd.Series, side: str, stock_df: pd.DataFrame) -> Dict:
    """Generate enhanced trading signal with explanations"""
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available', 'score': 0.0, 'explanations': []}
    
    current_price = stock_df.iloc[-1]['Close']
    latest = stock_df.iloc[-1]
    
    try:
        thresholds = SIGNAL_THRESHOLDS[side]
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
        
        explanations = []
        weighted_score = 0.0
        
        if side == "call":
            # Delta condition
            delta_pass = delta >= thresholds['delta_base']
            delta_score = weights['delta'] if delta_pass else 0
            weighted_score += delta_score
            explanations.append({
                'condition': 'Delta',
                'passed': delta_pass,
                'value': delta,
                'threshold': thresholds['delta_base'],
                'weight': weights['delta'],
                'score': delta_score,
                'explanation': f"Delta {delta:.3f} {'✓' if delta_pass else '✗'} threshold {thresholds['delta_base']:.2f}. Higher delta = more price sensitivity."
            })
            
            # Gamma condition
            gamma_pass = gamma >= thresholds['gamma_base']
            gamma_score = weights['gamma'] if gamma_pass else 0
            weighted_score += gamma_score
            explanations.append({
                'condition': 'Gamma',
                'passed': gamma_pass,
                'value': gamma,
                'threshold': thresholds['gamma_base'],
                'weight': weights['gamma'],
                'score': gamma_score,
                'explanation': f"Gamma {gamma:.3f} {'✓' if gamma_pass else '✗'} threshold {thresholds['gamma_base']:.3f}. Higher gamma = faster delta changes."
            })
            
            # Trend condition
            trend_pass = ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20
            trend_score = weights['trend'] if trend_pass else 0
            weighted_score += trend_score
            explanations.append({
                'condition': 'Trend',
                'passed': trend_pass,
                'value': f"Price: {close:.2f}, EMA9: {ema_9:.2f}, EMA20: {ema_20:.2f}" if ema_9 and ema_20 else "N/A",
                'threshold': "Price > EMA9 > EMA20",
                'weight': weights['trend'],
                'score': trend_score,
                'explanation': f"Price above short-term EMAs {'✓' if trend_pass else '✗'}. Bullish trend alignment."
            })
            
        else:  # put side
            # Delta condition
            delta_pass = delta <= thresholds['delta_base']
            delta_score = weights['delta'] if delta_pass else 0
            weighted_score += delta_score
            explanations.append({
                'condition': 'Delta',
                'passed': delta_pass,
                'value': delta,
                'threshold': thresholds['delta_base'],
                'weight': weights['delta'],
                'score': delta_score,
                'explanation': f"Delta {delta:.3f} {'✓' if delta_pass else '✗'} threshold {thresholds['delta_base']:.2f}. More negative delta = higher put sensitivity."
            })
            
            # Gamma condition
            gamma_pass = gamma >= thresholds['gamma_base']
            gamma_score = weights['gamma'] if gamma_pass else 0
            weighted_score += gamma_score
            explanations.append({
                'condition': 'Gamma',
                'passed': gamma_pass,
                'value': gamma,
                'threshold': thresholds['gamma_base'],
                'weight': weights['gamma'],
                'score': gamma_score,
                'explanation': f"Gamma {gamma:.3f} {'✓' if gamma_pass else '✗'} threshold {thresholds['gamma_base']:.3f}. Higher gamma = faster delta changes."
            })
            
            # Trend condition
            trend_pass = ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20
            trend_score = weights['trend'] if trend_pass else 0
            weighted_score += trend_score
            explanations.append({
                'condition': 'Trend',
                'passed': trend_pass,
                'value': f"Price: {close:.2f}, EMA9: {ema_9:.2f}, EMA20: {ema_20:.2f}" if ema_9 and ema_20 else "N/A",
                'threshold': "Price < EMA9 < EMA20",
                'weight': weights['trend'],
                'score': trend_score,
                'explanation': f"Price below short-term EMAs {'✓' if trend_pass else '✗'}. Bearish trend alignment."
            })
        
        # Momentum condition (RSI)
        if side == "call":
            momentum_pass = rsi is not None and rsi > thresholds['rsi_min']
        else:
            momentum_pass = rsi is not None and rsi < thresholds['rsi_max']
        
        momentum_score = weights['momentum'] if momentum_pass else 0
        weighted_score += momentum_score
        explanations.append({
            'condition': 'Momentum (RSI)',
            'passed': momentum_pass,
            'value': rsi,
            'threshold': thresholds['rsi_min'] if side == "call" else thresholds['rsi_max'],
            'weight': weights['momentum'],
            'score': momentum_score,
            'explanation': f"RSI {rsi:.1f} {'✓' if momentum_pass else '✗'} indicates {'bullish' if side == 'call' else 'bearish'} momentum." if rsi else "RSI N/A"
        })
        
        # Volume condition
        volume_pass = option_volume > thresholds['volume_min']
        volume_score = weights['volume'] if volume_pass else 0
        weighted_score += volume_score
        explanations.append({
            'condition': 'Volume',
            'passed': volume_pass,
            'value': option_volume,
            'threshold': thresholds['volume_min'],
            'weight': weights['volume'],
            'score': volume_score,
            'explanation': f"Option volume {option_volume:.0f} {'✓' if volume_pass else '✗'} minimum {thresholds['volume_min']:.0f}. Higher volume = better liquidity."
        })
        
        # VWAP condition
        vwap_pass = False
        vwap_score = 0
        if vwap is not None:
            if side == "call":
                vwap_pass = close > vwap
            else:
                vwap_pass = close < vwap
            
            vwap_score = 0.15 if vwap_pass else 0
            weighted_score += vwap_score
            explanations.append({
                'condition': 'VWAP',
                'passed': vwap_pass,
                'value': vwap,
                'threshold': f"Price {'>' if side == 'call' else '<'} VWAP",
                'weight': 0.15,
                'score': vwap_score,
                'explanation': f"Price ${close:.2f} {'above' if close > vwap else 'below'} VWAP ${vwap:.2f}"
            })
        
        signal = all(exp['passed'] for exp in explanations)
        
        return {
            'signal': signal,
            'score': weighted_score,
            'max_score': 1.0,
            'score_percentage': weighted_score * 100,
            'explanations': explanations,
            'profit_target': option['lastPrice'] * (1 + CONFIG['PROFIT_TARGETS'][side]),
            'stop_loss': option['lastPrice'] * (1 - CONFIG['PROFIT_TARGETS']['stop_loss']),
            'open_interest': option['openInterest'],
            'volume': option['volume'],
            'implied_volatility': option['impliedVolatility']
        }
        
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}', 'score': 0.0, 'explanations': []}

def process_options_signals(options_df: pd.DataFrame, side: str, stock_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Process options for signals"""
    if options_df.empty or stock_df.empty:
        return pd.DataFrame()
    
    try:
        # Filter options
        options_df = options_df.copy()
        options_df = options_df[options_df['lastPrice'] > 0]
        options_df = options_df.dropna(subset=['strike', 'lastPrice', 'volume', 'openInterest'])
        
        if options_df.empty:
            return pd.DataFrame()
        
        # Add moneyness classification
        def classify_moneyness(strike, spot):
            diff_pct = abs(strike - spot) / spot
            if diff_pct < 0.01:
                return 'ATM'
            elif strike < spot:
                return 'ITM' if diff_pct > 0.03 else 'NTM'
            else:
                return 'OTM' if diff_pct > 0.03 else 'NTM'
        
        options_df['moneyness'] = options_df['strike'].apply(lambda x: classify_moneyness(x, current_price))
        
        # Fill missing Greeks with approximations
        for idx, row in options_df.iterrows():
            if pd.isna(row.get('delta')) or pd.isna(row.get('gamma')) or pd.isna(row.get('theta')):
                moneyness = current_price / row['strike']
                
                if side == 'call':
                    if moneyness > 1.03:
                        delta, gamma, theta = 0.7, 0.02, -0.05
                    elif moneyness > 0.97:
                        delta, gamma, theta = 0.5, 0.08, -0.1
                    else:
                        delta, gamma, theta = 0.3, 0.02, -0.02
                else:
                    if moneyness < 0.97:
                        delta, gamma, theta = -0.7, 0.02, -0.05
                    elif moneyness < 1.03:
                        delta, gamma, theta = -0.5, 0.08, -0.1
                    else:
                        delta, gamma, theta = -0.3, 0.02, -0.02
                
                options_df.loc[idx, 'delta'] = delta
                options_df.loc[idx, 'gamma'] = gamma
                options_df.loc[idx, 'theta'] = theta
        
        # Generate signals
        signals = []
        for idx, row in options_df.iterrows():
            signal_result = generate_enhanced_signal(row, side, stock_df)
            if signal_result['signal']:
                row_dict = row.to_dict()
                row_dict.update(signal_result)
                signals.append(row_dict)
        
        if signals:
            signals_df = pd.DataFrame(signals)
            return signals_df.sort_values('score_percentage', ascending=False)
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error processing options signals: {str(e)}")
        return pd.DataFrame()

# =============================
# CHART CREATION FUNCTIONS
# =============================
def create_tradingview_chart(df: pd.DataFrame, timeframe: str, sr_levels: Dict = None):
    """Create TradingView-style interactive chart"""
    if df.empty:
        return None
    
    try:
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            specs=[[{"secondary_y": True}], [{}], [{}], [{}]],
            subplot_titles=('Price Chart', 'Volume', 'RSI', 'MACD')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add indicators based on session state
        if st.session_state.chart_indicators.get('ema_9', True) and 'EMA_9' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['EMA_9'], 
                name='EMA 9', 
                line=dict(color='#2196f3', width=1)
            ), row=1, col=1)
        
        if st.session_state.chart_indicators.get('ema_20', True) and 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['EMA_20'], 
                name='EMA 20', 
                line=dict(color='#ff9800', width=1)
            ), row=1, col=1)
        
        if st.session_state.chart_indicators.get('ema_50', True) and 'EMA_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['EMA_50'], 
                name='EMA 50', 
                line=dict(color='#9c27b0', width=1)
            ), row=1, col=1)
        
        # Bollinger Bands
        if st.session_state.chart_indicators.get('bollinger', False) and 'BB_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['BB_upper'], 
                name='BB Upper', 
                line=dict(color='#607d8b', width=1, dash='dash')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['BB_lower'], 
                name='BB Lower', 
                line=dict(color='#607d8b', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(96, 125, 139, 0.1)'
            ), row=1, col=1)
        
        # VWAP
        if 'VWAP' in df.columns and not df['VWAP'].isna().all():
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['VWAP'], 
                name='VWAP', 
                line=dict(color='#00bcd4', width=2)
            ), row=1, col=1)
        
        # Add support/resistance levels
        if sr_levels:
            for level in sr_levels.get('support', []):
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="#4caf50",
                    annotation_text=f"S: ${level:.2f}",
                    row=1, col=1
                )
            
            for level in sr_levels.get('resistance', []):
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="#f44336",
                    annotation_text=f"R: ${level:.2f}",
                    row=1, col=1
                )
        
        # Volume bars
        if st.session_state.chart_indicators.get('volume', True):
            colors = ['#26a69a' if close >= open else '#ef5350' 
                     for close, open in zip(df['Close'], df['Open'])]
            fig.add_trace(
                go.Bar(
                    x=df['Datetime'], 
                    y=df['Volume'], 
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        # RSI
        if st.session_state.chart_indicators.get('rsi', True) and 'RSI' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['RSI'], 
                name='RSI', 
                line=dict(color='#9c27b0')
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#f44336", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#4caf50", row=3, col=1)
            fig.add_hline(y=50, line_dash="solid", line_color="#607d8b", row=3, col=1)
        
        # MACD
        if st.session_state.chart_indicators.get('macd', True) and 'MACD' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['MACD'], 
                name='MACD', 
                line=dict(color='#2196f3')
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=df['Datetime'], 
                y=df['MACD_signal'], 
                name='Signal', 
                line=dict(color='#ff9800')
            ), row=4, col=1)
            
            # MACD Histogram
            colors = ['#4caf50' if val >= 0 else '#f44336' for val in df['MACD_hist']]
            fig.add_trace(go.Bar(
                x=df['Datetime'], 
                y=df['MACD_hist'], 
                name='Histogram',
                marker_color=colors,
                opacity=0.7
            ), row=4, col=1)
        
        # Update layout with TradingView styling
        fig.update_layout(
            height=800,
            title=dict(
                text=f'Chart - {timeframe.upper()}',
                font=dict(color='#d1d4dc', size=18),
                x=0.02
            ),
            paper_bgcolor='#0d1421',
            plot_bgcolor='#0d1421',
            font=dict(color='#d1d4dc'),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis4=dict(
                showgrid=True,
                gridcolor='#363a45',
                zeroline=False
            )
        )
        
        # Update all y-axes
        for i in range(1, 5):
            fig.update_yaxes(
                showgrid=True,
                gridcolor='#363a45',
                zeroline=False,
                row=i, col=1
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# =============================
# UI COMPONENTS
# =============================
def create_timeframe_buttons():
    """Create timeframe selector buttons"""
    st.markdown("### 📊 Timeframe Selection")
    
    timeframes = list(CONFIG['TIMEFRAMES'].keys())
    cols = st.columns(len(timeframes))
    
    for i, tf in enumerate(timeframes):
        with cols[i]:
            if st.button(
                tf,
                key=f"tf_{tf}",
                help=f"Switch to {CONFIG['TIMEFRAMES'][tf]['name']} timeframe",
                type="primary" if st.session_state.selected_timeframe == tf else "secondary"
            ):
                st.session_state.selected_timeframe = tf
                st.rerun()

def create_market_status_header():
    """Create market status header"""
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    
    with col1:
        status, css_class = get_market_status()
        st.markdown(f'<div class="{css_class}">🕐 Market: {status}</div>', unsafe_allow_html=True)
    
    with col2:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        st.markdown(f"🕐 ET: {now.strftime('%H:%M:%S')}")
    
    with col3:
        cache_age = int(time.time() - st.session_state.get('last_refresh', 0))
        st.markdown(f"⚡ Cache: {cache_age}s")
    
    with col4:
        st.markdown(f"🔄 Refreshes: {st.session_state.refresh_counter}")
    
    with col5:
        if st.button("🔄 Refresh", key="manual_refresh"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_counter += 1
            st.rerun()

def create_enhanced_signals_panel(ticker: str, current_price: float):
    """Create enhanced signals panel"""
    st.markdown("### 🎯 Enhanced Signals")
    
    # Get options data
    expiries, calls, puts = get_options_data(ticker)
    
    if not expiries:
        st.warning("⚠️ Unable to fetch real options data. Using demo data.")
        expiries, calls, puts = generate_demo_options_data(ticker, current_price)
        st.info("📊 **DEMO DATA** - For interface demonstration only")
    
    if expiries:
        # Get stock data for signals
        stock_df = get_stock_data_with_indicators(ticker, st.session_state.selected_timeframe)
        
        if not stock_df.empty:
            # Process signals
            call_signals = process_options_signals(calls, 'call', stock_df, current_price)
            put_signals = process_options_signals(puts, 'put', stock_df, current_price)
            
            # Store in session state
            st.session_state.signals_data = {
                'calls': call_signals,
                'puts': put_signals
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Call Signals")
                if not call_signals.empty:
                    for idx, signal in call_signals.head(3).iterrows():
                        with st.expander(f"🟢 {signal['contractSymbol']} ({signal['score_percentage']:.1f}%)", expanded=idx==call_signals.index[0]):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Strike", f"${signal['strike']:.2f}")
                                st.metric("Delta", f"{signal['delta']:.3f}")
                                st.metric("Volume", f"{signal['volume']:.0f}")
                            with col_b:
                                st.metric("Last Price", f"${signal['lastPrice']:.2f}")
                                st.metric("Gamma", f"{signal['gamma']:.3f}")
                                st.metric("Open Interest", f"{signal['open_interest']:.0f}")
                            
                            # Signal explanations
                            st.markdown("**Signal Breakdown:**")
                            for exp in signal['explanations']:
                                status = "✅" if exp['passed'] else "❌"
                                st.write(f"{status} {exp['condition']}: {exp['explanation']}")
                else:
                    st.info("No call signals found")
            
            with col2:
                st.markdown("#### 📉 Put Signals")
                if not put_signals.empty:
                    for idx, signal in put_signals.head(3).iterrows():
                        with st.expander(f"🔴 {signal['contractSymbol']} ({signal['score_percentage']:.1f}%)", expanded=idx==put_signals.index[0]):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Strike", f"${signal['strike']:.2f}")
                                st.metric("Delta", f"{signal['delta']:.3f}")
                                st.metric("Volume", f"{signal['volume']:.0f}")
                            with col_b:
                                st.metric("Last Price", f"${signal['lastPrice']:.2f}")
                                st.metric("Gamma", f"{signal['gamma']:.3f}")
                                st.metric("Open Interest", f"{signal['open_interest']:.0f}")
                            
                            # Signal explanations
                            st.markdown("**Signal Breakdown:**")
                            for exp in signal['explanations']:
                                status = "✅" if exp['passed'] else "❌"
                                st.write(f"{status} {exp['condition']}: {exp['explanation']}")
                else:
                    st.info("No put signals found")

def create_technical_analysis_summary(df: pd.DataFrame):
    """Create technical analysis summary"""
    if df.empty:
        st.warning("No data available for technical analysis")
        return
    
    latest = df.iloc[-1]
    
    st.markdown("### 📊 Technical Analysis Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${latest['Close']:.2f}",
            delta=f"{((latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100):.2f}%" if len(df) > 1 else None
        )
    
    with col2:
        if not pd.isna(latest['RSI']):
            rsi_status = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"
            st.metric("RSI", f"{latest['RSI']:.1f}", delta=rsi_status)
        else:
            st.metric("RSI", "N/A")
    
    with col3:
        if not pd.isna(latest['ATR_pct']):
            vol_status = "High" if latest['ATR_pct'] > 0.05 else "Normal"
            st.metric("Volatility", f"{latest['ATR_pct']*100:.2f}%", delta=vol_status)
        else:
            st.metric("Volatility", "N/A")
    
    with col4:
        volume_ratio = latest['Volume'] / latest['avg_vol'] if not pd.isna(latest['avg_vol']) else 1
        vol_status = "High" if volume_ratio > 1.5 else "Normal"
        st.metric("Volume Ratio", f"{volume_ratio:.1f}x", delta=vol_status)
    
    # Trend analysis
    st.markdown("#### 📈 Trend Analysis")
    trend_signals = []
    
    if not pd.isna(latest['EMA_9']) and not pd.isna(latest['EMA_20']):
        if latest['Close'] > latest['EMA_9'] > latest['EMA_20']:
            trend_signals.append("🟢 **Bullish** - Price above short-term EMAs")
        elif latest['Close'] < latest['EMA_9'] < latest['EMA_20']:
            trend_signals.append("🔴 **Bearish** - Price below short-term EMAs")
        else:
            trend_signals.append("🟡 **Mixed** - Conflicting EMA signals")
    
    if not pd.isna(latest['VWAP']):
        if latest['Close'] > latest['VWAP']:
            trend_signals.append(f"🟢 **Above VWAP** - ${latest['VWAP']:.2f}")
        else:
            trend_signals.append(f"🔴 **Below VWAP** - ${latest['VWAP']:.2f}")
    
    if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal']:
            trend_signals.append("🟢 **MACD Bullish** - Above signal line")
        else:
            trend_signals.append("🔴 **MACD Bearish** - Below signal line")
    
    for signal in trend_signals:
        st.markdown(signal)

def create_usage_tracker():
    """Create free tier usage tracker"""
    st.markdown("### 📊 Free Tier Usage")
    
    if not st.session_state.API_CALL_LOG:
        st.info("No API usage recorded yet")
        return
    
    now = time.time()
    
    # Calculate usage
    av_usage_1min = len([t for t in st.session_state.API_CALL_LOG
                        if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 60])
    fmp_usage_1hr = len([t for t in st.session_state.API_CALL_LOG
                        if t['source'] == "FMP" and now - t['timestamp'] < 3600])
    iex_usage_1hr = len([t for t in st.session_state.API_CALL_LOG
                        if t['source'] == "IEX" and now - t['timestamp'] < 3600])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Alpha Vantage**")
        progress = min(1.0, av_usage_1min / 5)
        st.progress(progress, text=f"{av_usage_1min}/5 per minute")
        if progress > 0.8:
            st.warning("⚠️ Approaching limit")
    
    with col2:
        st.markdown("**FMP**")
        progress = min(1.0, fmp_usage_1hr / 10)
        st.progress(progress, text=f"{fmp_usage_1hr}/10 per hour")
        if progress > 0.8:
            st.warning("⚠️ Approaching limit")
    
    with col3:
        st.markdown("**IEX Cloud**")
        progress = min(1.0, iex_usage_1hr / 30)
        st.progress(progress, text=f"{iex_usage_1hr}/30 per hour")
        if progress > 0.8:
            st.warning("⚠️ Approaching limit")

# =============================
# MAIN APPLICATION
# =============================
def main():
    # App header
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #2962ff; margin: 0;'>📈 TradingView-Style Options Analyzer</h1>
        <p style='color: #787b86; margin: 5px 0;'>Professional Options Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market status header
    create_market_status_header()
    
    st.markdown("---")
    
    # Ticker input
    ticker = st.text_input(
        "Enter Stock Ticker:",
        value="SPY",
        placeholder="e.g., SPY, QQQ, AAPL",
        key="ticker_input"
    ).upper()
    
    if not ticker:
        st.info("👋 Enter a ticker symbol to begin analysis")
        return
    
    # Get current price
    current_price = get_current_price(ticker)
    if current_price > 0:
        st.success(f"✅ **{ticker}** - ${current_price:.2f}")
    else:
        st.error(f"❌ Unable to fetch price for {ticker}")
        return
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 General",
        "📈 Chart",
        "📰 News & Analysis",
        "💰 Financials", 
        "🔧 Technical",
        "💬 Forum"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get stock data
            stock_df = get_stock_data_with_indicators(ticker, st.session_state.selected_timeframe)
            
            if not stock_df.empty:
                # Calculate support/resistance
                sr_levels = calculate_support_resistance_levels(stock_df, current_price)
                st.session_state.sr_data = sr_levels
                
                # Create mini chart for overview
                mini_fig = create_tradingview_chart(stock_df.tail(50), st.session_state.selected_timeframe, sr_levels)
                if mini_fig:
                    mini_fig.update_layout(height=400)
                    st.plotly_chart(mini_fig, use_container_width=True)
                
                # Technical analysis summary
                create_technical_analysis_summary(stock_df)
            else:
                st.error("Unable to fetch stock data")
        
        with col2:
            # Enhanced signals panel
            create_enhanced_signals_panel(ticker, current_price)
    
    with tab2:
        # Timeframe selector
        create_timeframe_buttons()
        
        # Chart indicators toggle
        with st.expander("🔧 Chart Indicators", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.session_state.chart_indicators['ema_9'] = st.checkbox("EMA 9", value=True)
                st.session_state.chart_indicators['ema_20'] = st.checkbox("EMA 20", value=True)
            with col2:
                st.session_state.chart_indicators['ema_50'] = st.checkbox("EMA 50", value=True)
                st.session_state.chart_indicators['bollinger'] = st.checkbox("Bollinger Bands", value=False)
            with col3:
                st.session_state.chart_indicators['volume'] = st.checkbox("Volume", value=True)
                st.session_state.chart_indicators['rsi'] = st.checkbox("RSI", value=True)
            with col4:
                st.session_state.chart_indicators['macd'] = st.checkbox("MACD", value=True)
        
        # Main chart
        stock_df = get_stock_data_with_indicators(ticker, st.session_state.selected_timeframe)
        
        if not stock_df.empty:
            sr_levels = st.session_state.sr_data
            chart_fig = create_tradingview_chart(stock_df, st.session_state.selected_timeframe, sr_levels)
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
            
            # Support/Resistance summary
            st.markdown("### 📈 Support & Resistance Levels")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Support Levels**")
                for level in sr_levels.get('support', [])[:5]:
                    distance = abs(level - current_price) / current_price * 100
                    st.write(f"${level:.2f} ({distance:.1f}% away)")
            
            with col2:
                st.markdown("**🔴 Resistance Levels**")
                for level in sr_levels.get('resistance', [])[:5]:
                    distance = abs(level - current_price) / current_price * 100
                    st.write(f"${level:.2f} ({distance:.1f}% away)")
        else:
            st.error("Unable to fetch chart data")
    
    with tab3:
        st.markdown("### 📰 News & Market Analysis")
        
        # Company info
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'longName' in info:
                        st.markdown(f"**Company:** {info['longName']}")
                    if 'sector' in info:
                        st.markdown(f"**Sector:** {info['sector']}")
                
                with col2:
                    if 'marketCap' in info and info['marketCap']:
                        market_cap = info['marketCap']
                        if market_cap > 1e12:
                            st.markdown(f"**Market Cap:** ${market_cap/1e12:.2f}T")
                        elif market_cap > 1e9:
                            st.markdown(f"**Market Cap:** ${market_cap/1e9:.2f}B")
                        else:
                            st.markdown(f"**Market Cap:** ${market_cap/1e6:.2f}M")
                
                with col3:
                    if 'beta' in info and info['beta']:
                        st.markdown(f"**Beta:** {info['beta']:.2f}")
                    if 'trailingPE' in info and info['trailingPE']:
                        st.markdown(f"**P/E Ratio:** {info['trailingPE']:.2f}")
        except Exception as e:
            st.warning("Company information unavailable")
        
        # Recent news
        st.markdown("### 📰 Recent News")
        try:
            news = stock.news
            if news:
                for i, item in enumerate(news[:5]):
                    title = item.get('title', 'Untitled')
                    publisher = item.get('publisher', 'Unknown')
                    link = item.get('link', '#')
                    
                    st.markdown(f"**{i+1}. {title}**")
                    st.write(f"📰 {publisher}")
                    if link != '#':
                        st.markdown(f"🔗 [Read Article]({link})")
                    st.markdown("---")
            else:
                st.info("No recent news available")
        except Exception:
            st.warning("News unavailable")
    
    with tab4:
        st.markdown("### 💰 Financial Data")
        
        try:
            stock = yf.Ticker(ticker)
            
            # Key statistics
            st.markdown("#### 📊 Key Statistics")
            info = stock.info
            
            if info:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'trailingPE' in info and info['trailingPE']:
                        st.metric("P/E Ratio", f"{info['trailingPE']:.2f}")
                    if 'forwardPE' in info and info['forwardPE']:
                        st.metric("Forward P/E", f"{info['forwardPE']:.2f}")
                
                with col2:
                    if 'priceToBook' in info and info['priceToBook']:
                        st.metric("P/B Ratio", f"{info['priceToBook']:.2f}")
                    if 'returnOnEquity' in info and info['returnOnEquity']:
                        st.metric("ROE", f"{info['returnOnEquity']*100:.1f}%")
                
                with col3:
                    if 'debtToEquity' in info and info['debtToEquity']:
                        st.metric("Debt/Equity", f"{info['debtToEquity']:.2f}")
                    if 'currentRatio' in info and info['currentRatio']:
                        st.metric("Current Ratio", f"{info['currentRatio']:.2f}")
                
                with col4:
                    if 'dividendYield' in info and info['dividendYield']:
                        st.metric("Dividend Yield", f"{info['dividendYield']*100:.2f}%")
                    if 'payoutRatio' in info and info['payoutRatio']:
                        st.metric("Payout Ratio", f"{info['payoutRatio']*100:.1f}%")
            
        except Exception:
            st.warning("Financial data unavailable")
    
    with tab5:
        st.markdown("### 🔧 Technical Settings")
        
        # Signal thresholds
        st.markdown("#### ⚙️ Signal Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Call Signals**")
            SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider(
                "Call Delta Threshold", 0.1, 1.0, 0.5, 0.1, key="call_delta"
            )
            SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider(
                "Call Gamma Threshold", 0.01, 0.2, 0.05, 0.01, key="call_gamma"
            )
            SIGNAL_THRESHOLDS['call']['volume_min'] = st.slider(
                "Call Min Volume", 100, 5000, 1000, 100, key="call_volume"
            )
        
        with col2:
            st.markdown("**📉 Put Signals**")
            SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider(
                "Put Delta Threshold", -1.0, -0.1, -0.5, 0.1, key="put_delta"
            )
            SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider(
                "Put Gamma Threshold", 0.01, 0.2, 0.05, 0.01, key="put_gamma"
            )
            SIGNAL_THRESHOLDS['put']['volume_min'] = st.slider(
                "Put Min Volume", 100, 5000, 1000, 100, key="put_volume"
            )
        
        # API Configuration
        st.markdown("#### 🔑 API Configuration")
        CONFIG['POLYGON_API_KEY'] = st.text_input("Polygon API Key:", type="password", value=CONFIG['POLYGON_API_KEY'])
        CONFIG['ALPHA_VANTAGE_API_KEY'] = st.text_input("Alpha Vantage API Key:", type="password", value=CONFIG['ALPHA_VANTAGE_API_KEY'])
        CONFIG['FMP_API_KEY'] = st.text_input("FMP API Key:", type="password", value=CONFIG['FMP_API_KEY'])
        CONFIG['IEX_API_KEY'] = st.text_input("IEX API Key:", type="password", value=CONFIG['IEX_API_KEY'])
        
        # Usage tracker
        create_usage_tracker()
    
    with tab6:
        st.markdown("### 💬 Community Forum")
        st.info("🚧 Forum feature coming soon!")
        
        # Placeholder for forum functionality
        st.markdown("""
        **Planned Features:**
        - 💬 Trade discussions
        - 📊 Signal sharing
        - 🎓 Educational content
        - 👥 Community insights
        """)

if __name__ == "__main__":
    main()