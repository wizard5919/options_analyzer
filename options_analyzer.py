import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import pytz
import math
from typing import Optional, Tuple, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, KeltnerChannel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import aiohttp
import asyncio
import requests
import unittest  # For unit tests
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

st.set_page_config(page_title="Options Greeks Buy Signal Analyzer", layout="wide", initial_sidebar_state="expanded")

# =============================
# CONFIGURATION & CONSTANTS
# =============================
CONFIG = {
    'MAX_RETRIES': 5,
    'RETRY_DELAY': 2,
    'DATA_TIMEOUT': 30,
    'MIN_DATA_POINTS': 50,
    'CACHE_TTL': 300,  # 5 minutes
    'RATE_LIMIT_COOLDOWN': 300,  # 5 minutes
    'MIN_REFRESH_INTERVAL': 60,
    'MARKET_OPEN': datetime.time(9, 30),
    'MARKET_CLOSE': datetime.time(16, 0),
    'PREMARKET_START': datetime.time(4, 0),
    'VOLATILITY_THRESHOLDS': {'low': 0.015, 'medium': 0.03, 'high': 0.05},
    'PROFIT_TARGETS': {'call': 0.15, 'put': 0.15, 'stop_loss': 0.08},
    'TRADING_HOURS_PER_DAY': 6.5,
    'SR_TIME_WINDOWS': {'scalping': ['1min'], 'intraday': ['15min', '1h']},  # Reduced for speed
    'SR_SENSITIVITY': {'1min': 0.001, '15min': 0.003, '1h': 0.008},
    'SR_WINDOW_SIZES': {'1min': 3, '15min': 5, '1h': 10},
    'POLYGON_API_KEY': '',
    'ALPHA_VANTAGE_API_KEY': '',
    'FMP_API_KEY': '',
    'IEX_API_KEY': '',
    'FINNHUB_API_KEY': '',
}

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_base': 0.5, 'delta_vol_multiplier': 0.1,
        'gamma_base': 0.05, 'gamma_vol_multiplier': 0.02,
        'theta_base': 0.05,
        'rsi_min': 50,
        'volume_multiplier_base': 1.0, 'volume_vol_multiplier': 0.3, 'volume_min': 1000,
        'condition_weights': {
            'delta': 0.20, 'gamma': 0.15, 'theta': 0.10, 'trend': 0.15,
            'momentum': 0.05, 'volume': 0.05, 'vega': 0.10, 'oi': 0.10,
            'sr_proximity': 0.15, 'sentiment': 0.05
        }
    },
    'put': {
        'delta_base': -0.5, 'delta_vol_multiplier': 0.1,
        'gamma_base': 0.05, 'gamma_vol_multiplier': 0.02,
        'theta_base': 0.05,
        'rsi_max': 50,
        'volume_multiplier_base': 1.0, 'volume_vol_multiplier': 0.3, 'volume_min': 1000,
        'condition_weights': {
            'delta': 0.20, 'gamma': 0.15, 'theta': 0.10, 'trend': 0.15,
            'momentum': 0.05, 'volume': 0.05, 'vega': 0.10, 'oi': 0.10,
            'sr_proximity': 0.15, 'sentiment': 0.05
        }
    }
}

# Session state initialization
if 'API_CALL_LOG' not in st.session_state:
    st.session_state.API_CALL_LOG = []
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = CONFIG['MIN_REFRESH_INTERVAL']
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = False
if 'sr_data' not in st.session_state:
    st.session_state.sr_data = {}
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = ""
if 'yf_rate_limited_until' not in st.session_state:
    st.session_state.yf_rate_limited_until = 0
if 'session_start' not in st.session_state:
    st.session_state.session_start = time.time()
if 'force_demo' not in st.session_state:
    st.session_state.force_demo = False

# =============================
# UTILITY FUNCTIONS
# =============================
def can_make_request(source: str) -> bool:
    now = time.time()
    st.session_state.API_CALL_LOG = [t for t in st.session_state.API_CALL_LOG if now - t['timestamp'] < 3600]
    counts = {
        "ALPHA_VANTAGE": len([t for t in st.session_state.API_CALL_LOG if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 60]),
        "FMP": len([t for t in st.session_state.API_CALL_LOG if t['source'] == "FMP" and now - t['timestamp'] < 3600]),
        "IEX": len([t for t in st.session_state.API_CALL_LOG if t['source'] == "IEX" and now - t['timestamp'] < 3600]),
    }
    limits = {"ALPHA_VANTAGE": 4, "FMP": 9, "IEX": 29}
    return counts.get(source, 0) < limits.get(source, 0)

def log_api_request(source: str):
    st.session_state.API_CALL_LOG.append({'source': source, 'timestamp': time.time()})

def is_market_open() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    return CONFIG['MARKET_OPEN'] <= now.time() <= CONFIG['MARKET_CLOSE']

def is_premarket() -> bool:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    return CONFIG['PREMARKET_START'] <= now.time() < CONFIG['MARKET_OPEN']

def is_early_market() -> bool:
    if not is_market_open():
        return False
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open_today = eastern.localize(datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN']))
    return (now - market_open_today).total_seconds() < 1800

def calculate_remaining_trading_hours() -> float:
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    close_time = eastern.localize(datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE']))
    if now >= close_time:
        return 0.0
    return (close_time - now).total_seconds() / 3600

async def async_get_current_price(ticker: str) -> float:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        tasks = []
        if CONFIG['POLYGON_API_KEY']:
            tasks.append(fetch_polygon(session, ticker))
        if CONFIG['ALPHA_VANTAGE_API_KEY'] and can_make_request("ALPHA_VANTAGE"):
            tasks.append(fetch_alpha_vantage(session, ticker))
        if CONFIG['FMP_API_KEY'] and can_make_request("FMP"):
            tasks.append(fetch_fmp(session, ticker))
        if CONFIG['IEX_API_KEY'] and can_make_request("IEX"):
            tasks.append(fetch_iex(session, ticker))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if not isinstance(result, Exception) and result > 0:
                return result
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        pass
    return 0.0

async def fetch_polygon(session, ticker):
    try:
        url = f"https://api.polygon.io/v2/last/trades/{ticker}?apiKey={CONFIG['POLYGON_API_KEY']}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if 'results' in data and data['results']:
                    return float(data['results'][0]['p'])
    except:
        pass
    return 0.0

async def fetch_alpha_vantage(session, ticker):
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={CONFIG['ALPHA_VANTAGE_API_KEY']}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if 'Global Quote' in data and '05. price' in data['Global Quote']:
                    log_api_request("ALPHA_VANTAGE")
                    return float(data['Global Quote']['05. price'])
    except:
        pass
    return 0.0

async def fetch_fmp(session, ticker):
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{ticker}?apikey={CONFIG['FMP_API_KEY']}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if data and 'price' in data[0]:
                    log_api_request("FMP")
                    return float(data[0]['price'])
    except:
        pass
    return 0.0

async def fetch_iex(session, ticker):
    try:
        url = f"https://cloud.iexapis.com/stable/stock/{ticker}/quote?token={CONFIG['IEX_API_KEY']}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if 'latestPrice' in data:
                    log_api_request("IEX")
                    return float(data['latestPrice'])
    except:
        pass
    return 0.0

@st.cache_data(ttl=5, show_spinner=False)
def get_current_price(ticker: str) -> float:
    return asyncio.run(async_get_current_price(ticker))

# NEW: Combined stock data and indicators function for better caching
@st.cache_data(ttl=CONFIG['STOCK_CACHE_TTL'], show_spinner=False)
def get_stock_data_with_indicators(ticker: str) -> pd.DataFrame:
    """Fetch stock data and compute all indicators in one cached function"""
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
            prepost=True
        )
        if data.empty:
            return pd.DataFrame()
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
      
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return pd.DataFrame()
        # Clean and validate data
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
      
        # Add premarket indicator
        data['premarket'] = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
      
        data = data.reset_index(drop=False)
      
        # Compute all indicators in one go
        data = compute_all_indicators(data)
      
        return data
      
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators with comprehensive error handling"""
    if df.empty:
        return df
  
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
      
        # Validate required columns exist
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
      
        # Ensure data types are correct
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
      
        # Remove any remaining NaN values
        df = df.dropna(subset=required_cols)
      
        if df.empty:
            return df
      
        # Extract series for calculations
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
          
        # RSI
        if len(close) >= 14:
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
        else:
            df['RSI'] = np.nan
        # VWAP calculation by session
        df['VWAP'] = np.nan
        for session, group in df.groupby(pd.Grouper(key='Datetime', freq='D')):
            if group.empty:
                continue
          
            # Calculate VWAP for regular hours
            regular = group[~group['premarket']]
            if not regular.empty:
                typical_price = (regular['High'] + regular['Low'] + regular['Close']) / 3
                vwap_cumsum = (regular['Volume'] * typical_price).cumsum()
                volume_cumsum = regular['Volume'].cumsum()
                regular_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                df.loc[regular.index, 'VWAP'] = regular_vwap
          
            # Calculate VWAP for premarket
            premarket = group[group['premarket']]
            if not premarket.empty:
                typical_price = (premarket['High'] + premarket['Low'] + premarket['Close']) / 3
                vwap_cumsum = (premarket['Volume'] * typical_price).cumsum()
                volume_cumsum = premarket['Volume'].cumsum()
                premarket_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                df.loc[premarket.index, 'VWAP'] = premarket_vwap
      
        # ATR
        if len(close) >= 14:
            atr = AverageTrueRange(high=high, low=low, close=close, window=14)
            df['ATR'] = atr.average_true_range()
            df['ATR_pct'] = df['ATR'] / close
        else:
            df['ATR'] = np.nan
            df['ATR_pct'] = np.nan
      
        # MACD and Keltner Channels
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
            for col in ['MACD', 'MACD_signal', 'MACD_hist', 'KC_upper', 'KC_middle', 'KC_lower']:
                df[col] = np.nan
      
        # Calculate volume averages
        df = calculate_volume_averages(df)
      
        return df
      
    except Exception as e:
        st.error(f"Error in compute_all_indicators: {str(e)}")
        return pd.DataFrame()
def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume averages with separate premarket handling"""
    if df.empty:
        return df
  
    df = df.copy()
    df['avg_vol'] = np.nan
  
    try:
        # Group by date and calculate averages
        for date, group in df.groupby(df['Datetime'].dt.date):
            regular = group[~group['premarket']]
            if not regular.empty:
                regular_avg_vol = regular['Volume'].expanding(min_periods=1).mean()
                df.loc[regular.index, 'avg_vol'] = regular_avg_vol
          
            premarket = group[group['premarket']]
            if not premarket.empty:
                premarket_avg_vol = premarket['Volume'].expanding(min_periods=1).mean()
                df.loc[premarket.index, 'avg_vol'] = premarket_avg_vol
      
        # Fill any remaining NaN values with overall average
        overall_avg = df['Volume'].mean()
        df['avg_vol'] = df['avg_vol'].fillna(overall_avg)
      
    except Exception as e:
        st.warning(f"Error calculating volume averages: {str(e)}")
        df['avg_vol'] = df['Volume'].mean()
  
    return df
# NEW: Real data fetching with fixed session handling
@st.cache_data(ttl=1800, show_spinner=False) # 30-minute cache for real data
def get_real_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Get real options data with proper yfinance handling"""
  
    # Check if we can clear the rate limit status
    if 'yf_rate_limited_until' in st.session_state:
        time_remaining = st.session_state['yf_rate_limited_until'] - time.time()
        if time_remaining <= 0:
            # Rate limit expired, try again
            del st.session_state['yf_rate_limited_until']
        else:
            return [], pd.DataFrame(), pd.DataFrame()
  
    try:
        # Don't use custom session - let yfinance handle it
        stock = yf.Ticker(ticker)
      
        # Single attempt with minimal delay
        try:
            expiries = list(stock.options) if stock.options else []
          
            if not expiries:
                return [], pd.DataFrame(), pd.DataFrame()
          
            # Get only the nearest expiry to minimize API calls
            nearest_expiry = expiries[0]
          
            # Add small delay
            time.sleep(1)
          
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
          
            # Validate we have essential columns
            required_cols = ['strike', 'lastPrice', 'volume', 'openInterest']
            calls_valid = all(col in calls.columns for col in required_cols)
            puts_valid = all(col in puts.columns for col in required_cols)
          
            if not (calls_valid and puts_valid):
                return [], pd.DataFrame(), pd.DataFrame()
          
            # Add Greeks columns if missing
            for df_name, df in [('calls', calls), ('puts', puts)]:
                if 'delta' not in df.columns:
                    df['delta'] = np.nan
                if 'gamma' not in df.columns:
                    df['gamma'] = np.nan
                if 'theta' not in df.columns:
                    df['theta'] = np.nan
                if 'vega' not in df.columns: # NEW: Add vega if missing
                    df['vega'] = np.nan
          
            return [nearest_expiry], calls, puts
          
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["too many requests", "rate limit", "429", "quota"]):
                # Set a shorter cooldown for real data attempts
                st.session_state['yf_rate_limited_until'] = time.time() + 180 # 3 minutes
                return [], pd.DataFrame(), pd.DataFrame()
            else:
                return [], pd.DataFrame(), pd.DataFrame()
              
    except Exception as e:
        return [], pd.DataFrame(), pd.DataFrame()
def clear_rate_limit():
    """Allow user to manually clear rate limit"""
    if 'yf_rate_limited_until' in st.session_state:
        del st.session_state['yf_rate_limited_until']
        st.success("✅ Rate limit status cleared - try fetching data again")
        st.rerun()
# NEW: Non-cached options data fetching (no widgets in cached functions)
def get_full_options_chain(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Get options data - prioritize real data, handle UI separately"""
  
    # Try to get real data
    expiries, calls, puts = get_real_options_data(ticker)
  
    return expiries, calls, puts
def get_fallback_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Enhanced fallback method with realistic options data"""
  
    # Get current price for realistic strikes
    try:
        current_price = get_current_price(ticker)
        if current_price <= 0:
            # Default prices for common tickers
            default_prices = {
                'SPY': 550, 'QQQ': 480, 'IWM': 215, 'AAPL': 230,
                'TSLA': 250, 'NVDA': 125, 'MSFT': 420, 'GOOGL': 175,
                'AMZN': 185, 'META': 520
            }
            current_price = default_prices.get(ticker, 100)
    except:
        current_price = 100
  
    # Create realistic strike ranges around current price
    strike_range = max(5, current_price * 0.1) # 10% range or minimum $5
    strikes = []
  
    # Generate strikes in reasonable increments
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
  
    # Generate expiry dates with proper handling for weekends/holidays
    today = datetime.date.today()
    expiries = []
  
    # Add today if it's a weekday and market is open (0DTE)
    if today.weekday() < 5 and is_market_open():
        expiries.append(today.strftime('%Y-%m-%d'))
  
    # Add next Friday
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_friday = today + datetime.timedelta(days=days_until_friday)
    expiries.append(next_friday.strftime('%Y-%m-%d'))
  
    # Add week after
    week_after = next_friday + datetime.timedelta(days=7)
    expiries.append(week_after.strftime('%Y-%m-%d'))
  
    st.info(f"📊 Generated {len(strikes)} strikes around ${current_price:.2f} for {ticker}")
  
    # Create realistic options data with better Greeks approximation
    calls_data = []
    puts_data = []
  
    for expiry in expiries:
        expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
        days_to_expiry = (expiry_date - today).days
        is_0dte = days_to_expiry == 0
      
        for strike in strikes:
            # Calculate moneyness
            moneyness = current_price / strike
          
            # Enhanced Greeks calculation using Black-Scholes approximation
            # This is a simplified version for demo purposes
            t = days_to_expiry / 365.0
            iv = 0.25 # Assume 25% implied volatility
          
            # Calculate d1 for Black-Scholes
            d1 = (math.log(current_price / strike) + (0.5 * iv**2) * t) / (iv * math.sqrt(t))
          
            # Delta calculation
            if moneyness > 1: # ITM call
                call_delta = 0.7 + (moneyness - 1) * 0.2
                put_delta = call_delta - 1
            else: # OTM call
                call_delta = 0.3 - (1 - moneyness) * 0.2
                put_delta = call_delta - 1
              
            # Gamma calculation (same for calls and puts)
            gamma = (1 / (current_price * iv * math.sqrt(t))) * (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2)
          
            # Theta increases as expiry approaches
            theta = -0.1 if is_0dte else -0.05 if days_to_expiry <= 7 else -0.02
          
            # Vega calculation
            vega = current_price * math.sqrt(t) * (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1**2) * 0.01
          
            # Realistic pricing (very rough estimate)
            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = 5 if is_0dte else 10 if days_to_expiry <= 7 else 15
          
            call_price = intrinsic_call + time_value * gamma
            put_price = intrinsic_put + time_value * gamma
          
            # Volume estimates
            volume = 1000 if abs(moneyness - 1) < 0.05 else 500 # Higher volume near ATM
            open_interest = volume * 2 # OI typically higher than volume
          
            calls_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}C{strike*1000:08.0f}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(call_price, 2),
                'volume': volume,
                'openInterest': open_interest,
                'impliedVolatility': iv,
                'delta': round(call_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'vega': round(vega, 3)
            })
          
            puts_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}P{strike*1000:08.0f}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(put_price, 2),
                'volume': volume,
                'openInterest': open_interest,
                'impliedVolatility': iv,
                'delta': round(put_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'vega': round(vega, 3)
            })
  
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
  
    st.success(f"✅ Generated realistic demo data: {len(calls_df)} calls, {len(puts_df)} puts")
    st.warning("⚠️ **DEMO DATA**: Realistic structure but not real market data. Do not use for actual trading!")
    return expiries, calls_df, puts_df
def classify_moneyness(strike: float, spot: float) -> str:
    """Classify option moneyness with dynamic ranges"""
    try:
        diff = abs(strike - spot)
        diff_pct = diff / spot
      
        if diff_pct < 0.01: # Within 1%
            return 'ATM'
        elif strike < spot: # Below current price
            if diff_pct < 0.03: # 1-3% below
                return 'NTM' # Near-the-money
            else:
                return 'ITM'
        else: # Above current price
            if diff_pct < 0.03: # 1-3% above
                return 'NTM' # Near-the-money
            else:
                return 'OTM'
    except Exception:
        return 'Unknown'
def calculate_approximate_greeks(option: dict, spot_price: float) -> Tuple[float, float, float]:
    """Calculate approximate Greeks using simple formulas"""
    try:
        moneyness = spot_price / option['strike']
      
        if 'C' in option.get('contractSymbol', ''):
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
    except Exception:
        return 0.5, 0.05, 0.02
def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    """Validate that option has required data for analysis"""
    try:
        required_fields = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
      
        for field in required_fields:
            if field not in option or pd.isna(option[field]):
                return False
      
        if option['lastPrice'] <= 0:
            return False
      
        # Fill in Greeks if missing
        if pd.isna(option.get('delta')) or pd.isna(option.get('gamma')) or pd.isna(option.get('theta')):
            delta, gamma, theta = calculate_approximate_greeks(option.to_dict(), spot_price)
            option['delta'] = delta
            option['gamma'] = gamma
            option['theta'] = theta
      
        # Fill in Vega if missing
        if pd.isna(option.get('vega')):
            try:
                expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
                today = datetime.date.today()
                dte = max(1, (expiry_date - today).days)
                # Approximate vega: vega = price * sqrt(dte/365) * 0.01
                option['vega'] = float(option['lastPrice']) * math.sqrt(dte / 365) * 0.01
            except:
                option['vega'] = 0.1
      
        return True
    except Exception:
        return False
def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, option: dict, current_price: float) -> Dict[str, float]:
    """Calculate dynamic thresholds with enhanced volatility response"""
    try:
        thresholds = SIGNAL_THRESHOLDS[side].copy()
      
        volatility = stock_data.get('ATR_pct', 0.02)
      
        # Handle NaN volatility
        if pd.isna(volatility):
            volatility = 0.02
      
        vol_multiplier = 1 + (volatility * 100)
      
        if side == 'call':
            thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_multiplier))
        else:
            thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_multiplier))
      
        thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + thresholds['gamma_vol_multiplier'] * (volatility * 200))
      
        thresholds['volume_multiplier'] = max(0.8, min(2.5, thresholds['volume_multiplier_base'] * (1 + thresholds['volume_vol_multiplier'] * (volatility * 150))))
      
        # Adjust for market conditions
        if is_premarket() or is_early_market():
            if side == 'call':
                thresholds['delta_min'] = 0.35
            else:
                thresholds['delta_max'] = -0.35
            thresholds['volume_multiplier'] *= 0.6
            thresholds['gamma_min'] *= 0.8
      
        # DTE and moneyness adjustments
        try:
            expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
            today = datetime.date.today()
            dte = (expiry_date - today).days
            if dte < 0:
                dte = 0
        except:
            dte = 1 # fallback
      
        moneyness = abs(float(option['strike']) - current_price) / current_price
      
        # Adjust thresholds by DTE
        if dte <= 1: # 0DTE/1DTE
            thresholds['gamma_min'] *= 1.5
            thresholds['theta_base'] *= 0.8 # less tolerance for decay
        elif dte > 7: # longer-term
            thresholds['gamma_min'] *= 0.7
            thresholds['theta_base'] *= 1.2 # more tolerance for decay
      
        # Adjust by moneyness
        if moneyness < 0.01: # ATM
            thresholds['gamma_min'] *= 1.2
        elif moneyness > 0.05: # OTM
            thresholds['volume_min'] *= 1.5 # require more volume for conviction
      
        return thresholds
    except Exception:
        return SIGNAL_THRESHOLDS[side].copy()
# NEW: Enhanced signal generation with precision improvements
def generate_enhanced_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    """Generate trading signal with weighted scoring and detailed explanations"""
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available', 'score': 0.0, 'explanations': []}
  
    current_price = stock_df.iloc[-1]['Close']
  
    if not validate_option_data(option, current_price):
        return {'signal': False, 'reason': 'Insufficient option data', 'score': 0.0, 'explanations': []}
  
    latest = stock_df.iloc[-1]
  
    try:
        # Calculate dynamic thresholds with option data
        thresholds = calculate_dynamic_thresholds(latest, side, option.to_dict(), current_price)
        weights = thresholds['condition_weights']
      
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        option_volume = float(option['volume'])
        open_interest = float(option['openInterest'])
        iv = float(option.get('impliedVolatility', 0.25)) # Default to 25% if missing
        vega = float(option.get('vega', 0.1)) # Default to 0.1 if missing
      
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume
      
        # Get sentiment
        sentiment = get_sentiment_score(option.get('contractSymbol', '').split()[0] if 'contractSymbol' in option else ticker)
      
        conditions = []
        explanations = []
        weighted_score = 0.0
      
        if side == "call":
            # Delta condition
            delta_pass = delta >= thresholds.get('delta_min', 0.5)
            delta_score = weights['delta'] if delta_pass else 0
            weighted_score += delta_score
            conditions.append((delta_pass, f"Delta >= {thresholds.get('delta_min', 0.5):.2f}", delta))
            explanations.append({
                'condition': 'Delta',
                'passed': delta_pass,
                'value': delta,
                'threshold': thresholds.get('delta_min', 0.5),
                'weight': weights['delta'],
                'score': delta_score,
                'explanation': f"Delta {delta:.3f} {'✓' if delta_pass else '✗'} threshold {thresholds.get('delta_min', 0.5):.2f}. Higher delta = more price sensitivity."
            })
          
            # Gamma condition
            gamma_pass = gamma >= thresholds.get('gamma_min', 0.05)
            gamma_score = weights['gamma'] if gamma_pass else 0
            weighted_score += gamma_score
            conditions.append((gamma_pass, f"Gamma >= {thresholds.get('gamma_min', 0.05):.3f}", gamma))
            explanations.append({
                'condition': 'Gamma',
                'passed': gamma_pass,
                'value': gamma,
                'threshold': thresholds.get('gamma_min', 0.05),
                'weight': weights['gamma'],
                'score': gamma_score,
                'explanation': f"Gamma {gamma:.3f} {'✓' if gamma_pass else '✗'} threshold {thresholds.get('gamma_min', 0.05):.3f}. Higher gamma = faster delta changes."
            })
          
            # Theta condition
            theta_pass = theta <= thresholds['theta_base']
            theta_score = weights['theta'] if theta_pass else 0
            weighted_score += theta_score
            conditions.append((theta_pass, f"Theta <= {thresholds['theta_base']:.3f}", theta))
            explanations.append({
                'condition': 'Theta',
                'passed': theta_pass,
                'value': theta,
                'threshold': thresholds['theta_base'],
                'weight': weights['theta'],
                'score': theta_score,
                'explanation': f"Theta {theta:.3f} {'✓' if theta_pass else '✗'} threshold {thresholds['theta_base']:.3f}. Lower theta = less time decay."
            })
          
            # Trend condition
            trend_pass = ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20
            trend_score = weights['trend'] if trend_pass else 0
            weighted_score += trend_score
            conditions.append((trend_pass, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"))
            explanations.append({
                'condition': 'Trend',
                'passed': trend_pass,
                'value': f"Price: {close:.2f}, EMA9: {ema_9:.2f}, EMA20: {ema_20:.2f}" if ema_9 and ema_20 else "N/A",
                'threshold': "Price > EMA9 > EMA20",
                'weight': weights['trend'],
                'score': trend_score,
                'explanation': f"Price above short-term EMAs {'✓' if trend_pass else '✗'}. Bullish trend alignment needed for calls."
            })
          
            # Sentiment condition
            sentiment_pass = sentiment > 0
            sentiment_score = weights['sentiment'] if sentiment_pass else 0
            weighted_score += sentiment_score
            explanations.append({
                'condition': 'Sentiment',
                'passed': sentiment_pass,
                'value': sentiment,
                'threshold': 0,
                'weight': weights['sentiment'],
                'score': sentiment_score,
                'explanation': f"Sentiment {sentiment:.2f} {'✓' if sentiment_pass else '✗'} positive for calls."
            })
        else: # put side
            # Delta condition
            delta_pass = delta <= thresholds.get('delta_max', -0.5)
            delta_score = weights['delta'] if delta_pass else 0
            weighted_score += delta_score
            conditions.append((delta_pass, f"Delta <= {thresholds.get('delta_max', -0.5):.2f}", delta))
            explanations.append({
                'condition': 'Delta',
                'passed': delta_pass,
                'value': delta,
                'threshold': thresholds.get('delta_max', -0.5),
                'weight': weights['delta'],
                'score': delta_score,
                'explanation': f"Delta {delta:.3f} {'✓' if delta_pass else '✗'} threshold {thresholds.get('delta_max', -0.5):.2f}. More negative delta = higher put sensitivity."
            })
          
            # Gamma condition (same as calls)
            gamma_pass = gamma >= thresholds.get('gamma_min', 0.05)
            gamma_score = weights['gamma'] if gamma_pass else 0
            weighted_score += gamma_score
            conditions.append((gamma_pass, f"Gamma >= {thresholds.get('gamma_min', 0.05):.3f}", gamma))
            explanations.append({
                'condition': 'Gamma',
                'passed': gamma_pass,
                'value': gamma,
                'threshold': thresholds.get('gamma_min', 0.05),
                'weight': weights['gamma'],
                'score': gamma_score,
                'explanation': f"Gamma {gamma:.3f} {'✓' if gamma_pass else '✗'} threshold {thresholds.get('gamma_min', 0.05):.3f}. Higher gamma = faster delta changes."
            })
          
            # Theta condition (same as calls)
            theta_pass = theta <= thresholds['theta_base']
            theta_score = weights['theta'] if theta_pass else 0
            weighted_score += theta_score
            conditions.append((theta_pass, f"Theta <= {thresholds['theta_base']:.3f}", theta))
            explanations.append({
                'condition': 'Theta',
                'passed': theta_pass,
                'value': theta,
                'threshold': thresholds['theta_base'],
                'weight': weights['theta'],
                'score': theta_score,
                'explanation': f"Theta {theta:.3f} {'✓' if theta_pass else '✗'} threshold {thresholds['theta_base']:.3f}. Lower theta = less time decay."
            })
          
            # Trend condition (inverted for puts)
            trend_pass = ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20
            trend_score = weights['trend'] if trend_pass else 0
            weighted_score += trend_score
            conditions.append((trend_pass, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"))
            explanations.append({
                'condition': 'Trend',
                'passed': trend_pass,
                'value': f"Price: {close:.2f}, EMA9: {ema_9:.2f}, EMA20: {ema_20:.2f}" if ema_9 and ema_20 else "N/A",
                'threshold': "Price < EMA9 < EMA20",
                'weight': weights['trend'],
                'score': trend_score,
                'explanation': f"Price below short-term EMAs {'✓' if trend_pass else '✗'}. Bearish trend alignment needed for puts."
            })
          
            # Sentiment condition (inverted for puts)
            sentiment_pass = sentiment < 0
            sentiment_score = weights['sentiment'] if sentiment_pass else 0
            weighted_score += sentiment_score
            explanations.append({
                'condition': 'Sentiment',
                'passed': sentiment_pass,
                'value': sentiment,
                'threshold': 0,
                'weight': weights['sentiment'],
                'score': sentiment_score,
                'explanation': f"Sentiment {sentiment:.2f} {'✓' if sentiment_pass else '✗'} negative for puts."
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
      
        # NEW: Vega/IV condition
        vega_min = 0.1
        iv_threshold = 0.35 # 35% IV threshold
        vega_pass = vega >= vega_min and iv < iv_threshold
        vega_score = weights['vega'] if vega_pass else 0
        weighted_score += vega_score
        explanations.append({
            'condition': 'Vega/IV',
            'passed': vega_pass,
            'value': f"Vega: {vega:.3f}, IV: {iv:.2f}",
            'threshold': f"Vega >= {vega_min}, IV < {iv_threshold}",
            'weight': weights['vega'],
            'score': vega_score,
            'explanation': f"Vega/IV {'✓' if vega_pass else '✗'}. Ensures volatility alignment."
        })
      
        # NEW: Open Interest condition
        oi_min = thresholds['volume_min'] * 2 # OI should be at least twice volume min
        oi_pass = open_interest >= oi_min
        oi_score = weights['oi'] if oi_pass else 0
        weighted_score += oi_score
        explanations.append({
            'condition': 'Open Interest',
            'passed': oi_pass,
            'value': open_interest,
            'threshold': oi_min,
            'weight': weights['oi'],
            'score': oi_score,
            'explanation': f"OI {open_interest:.0f} {'✓' if oi_pass else '✗'} min {oi_min:.0f}. High OI = better liquidity/conviction."
        })
      
        # NEW: S/R Proximity condition
        sr_proximity_pass = False
        if st.session_state.sr_data:
            # Get S/R levels from the 5min timeframe (most relevant for options trading)
            sr = st.session_state.sr_data.get('5min', {'support': [], 'resistance': []})
          
            # Find closest support/resistance
            closest_support = max(sr['support']) if sr['support'] else current_price * 0.98
            closest_resistance = min(sr['resistance']) if sr['resistance'] else current_price * 1.02
          
            strike = float(option['strike'])
            distance_threshold = 0.02 # 2% of current_price
          
            if side == "call":
                # For calls, we want near support for a bounce
                distance_to_level = abs(strike - closest_support)
                sr_proximity_pass = distance_to_level / current_price <= distance_threshold
            else:
                # For puts, near resistance for reversal
                distance_to_level = abs(strike - closest_resistance)
                sr_proximity_pass = distance_to_level / current_price <= distance_threshold
          
        sr_score = weights['sr_proximity'] if sr_proximity_pass else 0
        weighted_score += sr_score
        explanations.append({
            'condition': 'S/R Proximity',
            'passed': sr_proximity_pass,
            'value': f"{distance_to_level/current_price*100:.1f}%",
            'threshold': f"{distance_threshold*100:.1f}%",
            'weight': weights['sr_proximity'],
            'score': sr_score,
            'explanation': f"Strike near {'support' if side == 'call' else 'resistance'} {'✓' if sr_proximity_pass else '✗'}. Improves breakout odds."
        })
      
        signal = all(passed for passed, desc, val in conditions)
      
        # Calculate profit targets and other metrics
        profit_target = None
        stop_loss = None
        holding_period = None
        est_hourly_decay = 0.0
        est_remaining_decay = 0.0
      
        if signal:
            entry_price = option['lastPrice']
            option_type = 'call' if side == 'call' else 'put'
            profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS'][option_type])
            stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
          
            # Calculate holding period
            expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
            days_to_expiry = (expiry_date - datetime.date.today()).days
          
            if days_to_expiry == 0:
                holding_period = "Intraday (Exit before 3:30 PM)"
            elif days_to_expiry <= 3:
                holding_period = "1-2 days (Quick scalp)"
            else:
                holding_period = "3-7 days (Swing trade)"
          
            if is_0dte and theta:
                est_hourly_decay = -theta / CONFIG['TRADING_HOURS_PER_DAY']
                remaining_hours = calculate_remaining_trading_hours()
                est_remaining_decay = est_hourly_decay * remaining_hours
      
        return {
            'signal': signal,
            'score': weighted_score,
            'max_score': 1.0,
            'score_percentage': weighted_score * 100,
            'explanations': explanations,
            'thresholds': thresholds,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period,
            'est_hourly_decay': est_hourly_decay,
            'est_remaining_decay': est_remaining_decay,
            'passed_conditions': [exp['condition'] for exp in explanations if exp['passed']],
            'failed_conditions': [exp['condition'] for exp in explanations if not exp['passed']]
        }
      
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}', 'score': 0.0, 'explanations': []}
# =============================
# ENHANCED SENTIMENT ANALYSIS
# =============================
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_sentiment_score(ticker: str) -> float:
    """Get sentiment score for a ticker using Finnhub API if available, fallback to price action"""
    if CONFIG['FINNHUB_API_KEY']:
        try:
            from_date = (datetime.date.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = datetime.date.today().strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={ticker}&from={from_date}&to={to_date}&token={CONFIG['FINNHUB_API_KEY']}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            sentiment = 0.0
            sources = ['reddit', 'twitter']
            total_mention = 0
            total_score = 0.0
            for source in sources:
                if source in data and data[source]:
                    for day in data[source]:
                        mention = day.get('mention', 0)
                        score = day.get('score', 0)
                        total_mention += mention
                        total_score += score * mention
            if total_mention > 0:
                sentiment = total_score / total_mention
            return sentiment
        except Exception as e:
            st.warning(f"Finnhub sentiment error: {str(e)}. Falling back to price action.")
   
    # Fallback to price action
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if len(hist) < 2:
            return 0.0
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        five_day_return = (end_price - start_price) / start_price
        sentiment = np.tanh(five_day_return * 5)
        return float(sentiment)
    except:
        return 0.0
def process_options_batch(options_df: pd.DataFrame, side: str, stock_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Process options in batches for better performance"""
    if options_df.empty or stock_df.empty:
        return pd.DataFrame()
  
    try:
        # Add basic validation
        options_df = options_df.copy()
        options_df = options_df[options_df['lastPrice'] > 0]
        options_df = options_df.dropna(subset=['strike', 'lastPrice', 'volume', 'openInterest'])
      
        if options_df.empty:
            return pd.DataFrame()
      
        # Add 0DTE flag
        today = datetime.date.today()
        options_df['is_0dte'] = options_df['expiry'].apply(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today
        )
      
        # Add moneyness
        options_df['moneyness'] = options_df['strike'].apply(
            lambda x: classify_moneyness(x, current_price)
        )
      
        # Fill missing Greeks
        for idx, row in options_df.iterrows():
            if pd.isna(row.get('delta')) or pd.isna(row.get('gamma')) or pd.isna(row.get('theta')):
                delta, gamma, theta = calculate_approximate_greeks(row.to_dict(), current_price)
                options_df.loc[idx, 'delta'] = delta
                options_df.loc[idx, 'gamma'] = gamma
                options_df.loc[idx, 'theta'] = theta
          
            # Fill in Vega if missing
            if pd.isna(row.get('vega')):
                try:
                    expiry_date = datetime.datetime.strptime(row['expiry'], "%Y-%m-%d").date()
                    dte = max(1, (expiry_date - today).days)
                    # Approximate vega: vega = price * sqrt(dte/365) * 0.01
                    options_df.loc[idx, 'vega'] = float(row['lastPrice']) * math.sqrt(dte / 365) * 0.01
                except:
                    options_df.loc[idx, 'vega'] = 0.1
      
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
            return signals_df.sort_values('score_percentage', ascending=False)
      
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing options batch: {str(e)}")
        return pd.DataFrame()
def calculate_scanner_score(stock_df: pd.DataFrame, side: str) -> float:
    """Calculate a score for call/put scanner based on technical indicators"""
    if stock_df.empty:
        return 0.0
  
    latest = stock_df.iloc[-1]
  
    score = 0.0
    max_score = 5.0 # Five conditions
  
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
def create_stock_chart(df: pd.DataFrame, sr_levels: dict = None):
    """Create TradingView-style chart with indicators using Plotly"""
    if df.empty:
        return None
  
    try:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
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
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_9'], name='EMA 9', line=dict(color='blue')), row=1, col=1)
        if 'EMA_20' in df.columns and not df['EMA_20'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_20'], name='EMA 20', line=dict(color='orange')), row=1, col=1)
      
        # Keltner Channels
        if 'KC_upper' in df.columns and not df['KC_upper'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_upper'], name='KC Upper', line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_middle'], name='KC Middle', line=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_lower'], name='KC Lower', line=dict(color='red', dash='dash')), row=1, col=1)
      
        # Volume
        fig.add_trace(
            go.Bar(x=df['Datetime'], y=df['Volume'], name='Volume', marker_color='gray'),
            row=1, col=1, secondary_y=True
        )
      
        # RSI
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
      
        # MACD
        if 'MACD' in df.columns and not df['MACD'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Bar(x=df['Datetime'], y=df['MACD_hist'], name='Histogram', marker_color='gray'), row=3, col=1)
      
        # Add support and resistance levels if available
        if sr_levels:
            # Add support levels
            for level in sr_levels.get('5min', {}).get('support', []):
                if isinstance(level, (int, float)) and not math.isnan(level):
                    fig.add_hline(y=level, line_dash="dash", line_color="green", row=1, col=1,
                                 annotation_text=f"S: {level:.2f}", annotation_position="bottom right")
          
            # Add resistance levels
            for level in sr_levels.get('5min', {}).get('resistance', []):
                if isinstance(level, (int, float)) and not math.isnan(level):
                    fig.add_hline(y=level, line_dash="dash", line_color="red", row=1, col=1,
                                 annotation_text=f"R: {level:.2f}", annotation_position="top right")
      
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
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None
# =============================
# ENHANCED STREAMLIT INTERFACE
# =============================
# Initialize session state for enhanced auto-refresh
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = CONFIG['MIN_REFRESH_INTERVAL']
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = False
if 'sr_data' not in st.session_state:
    st.session_state.sr_data = {}
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = ""
# Enhanced rate limit check
if 'rate_limited_until' in st.session_state:
    if time.time() < st.session_state['rate_limited_until']:
        remaining = int(st.session_state['rate_limited_until'] - time.time())
        st.error(f"⚠️ API rate limited. Please wait {remaining} seconds before retrying.")
        st.stop()
    else:
        del st.session_state['rate_limited_until']
st.title("📈 Enhanced Options Greeks Analyzer")
st.markdown("**Performance Optimized** • Weighted Scoring • Smart Caching • Rate Limit Protection")
# Enhanced sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
  
    # API Key Section
    st.subheader("🔑 API Settings")
  
    # Polygon API Key Input
    polygon_api_key = st.text_input("Enter Polygon API Key:", type="password", value=CONFIG['POLYGON_API_KEY'])
    if polygon_api_key:
        CONFIG['POLYGON_API_KEY'] = polygon_api_key
        st.success("✅ Polygon API key saved!")
        st.info("💡 **Tip**: Polygon Premium provides higher rate limits and real-time Greeks")
    else:
        st.warning("⚠️ Using free data sources (limited rate)")
  
    # NEW: Free API Key Inputs
    st.subheader("🔑 Free API Keys")
    st.info("Use these free alternatives to reduce rate limits")
  
    CONFIG['ALPHA_VANTAGE_API_KEY'] = st.text_input(
        "Alpha Vantage API Key (free):",
        type="password",
        value=CONFIG['ALPHA_VANTAGE_API_KEY']
    )
  
    CONFIG['FMP_API_KEY'] = st.text_input(
        "Financial Modeling Prep API Key (free):",
        type="password",
        value=CONFIG['FMP_API_KEY']
    )
  
    CONFIG['IEX_API_KEY'] = st.text_input(
        "IEX Cloud API Key (free):",
        type="password",
        value=CONFIG['IEX_API_KEY']
    )
   
    CONFIG['FINNHUB_API_KEY'] = st.text_input(
        "Finnhub API Key (free for sentiment):",
        type="password",
        value=CONFIG['FINNHUB_API_KEY']
    )
  
    with st.expander("💡 How to get free keys"):
        st.markdown("""
        **1. Alpha Vantage:**
        - Visit [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
        - Free tier: 5 requests/minute, 500/day
      
        **2. Financial Modeling Prep:**
        - Visit [https://site.financialmodelingprep.com/developer](https://site.financialmodelingprep.com/developer)
        - Free tier: 250 requests/day
      
        **3. IEX Cloud:**
        - Visit [https://iexcloud.io/cloud-login#/register](https://iexcloud.io/cloud-login#/register)
        - Free tier: 50,000 credits/month
      
        **4. Finnhub:**
        - Visit [https://finnhub.io/register](https://finnhub.io/register)
        - Free tier for social sentiment
       
        **Pro Tip:** Use all three for maximum free requests!
        """)
  
    # Enhanced auto-refresh with minimum interval enforcement
    with st.container():
        st.subheader("🔄 Smart Auto-Refresh")
        enable_auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.auto_refresh_enabled,
            key='auto_refresh_enabled'
        )
      
        if enable_auto_refresh:
            refresh_options = [60, 120, 300, 600] # Enforced minimum intervals
            refresh_interval = st.selectbox(
                "Refresh Interval (Rate-Limit Safe)",
                options=refresh_options,
                index=1, # Default to 120 seconds
                format_func=lambda x: f"{x} seconds" if x < 60 else f"{x//60} minute{'s' if x > 60 else ''}",
                key='refresh_interval_selector'
            )
            st.session_state.refresh_interval = refresh_interval
          
            if refresh_interval >= 300:
                st.success(f"✅ Conservative: {refresh_interval}s interval")
            elif refresh_interval >= 120:
                st.info(f"⚖️ Balanced: {refresh_interval}s interval")
            else:
                st.warning(f"⚠️ Aggressive: {refresh_interval}s interval (may hit limits)")
  
    # Enhanced thresholds with tooltips
    with st.expander("📊 Signal Thresholds & Weights", expanded=False):
        st.markdown("**🏋️ Condition Weights** (How much each factor matters)")
      
        col1, col2 = st.columns(2)
      
        with col1:
            st.markdown("#### 📈 Calls")
            SIGNAL_THRESHOLDS['call']['condition_weights']['delta'] = st.slider(
                "Delta Weight", 0.1, 0.4, 0.20, 0.05,
                help="Higher delta = more price sensitivity",
                key="call_delta_weight"
            )
            SIGNAL_THRESHOLDS['call']['condition_weights']['gamma'] = st.slider(
                "Gamma Weight", 0.1, 0.3, 0.15, 0.05,
                help="Higher gamma = faster delta acceleration",
                key="call_gamma_weight"
            )
            SIGNAL_THRESHOLDS['call']['condition_weights']['trend'] = st.slider(
                "Trend Weight", 0.1, 0.3, 0.15, 0.05,
                help="EMA alignment strength",
                key="call_trend_weight"
            )
            SIGNAL_THRESHOLDS['call']['condition_weights']['vega'] = st.slider(
                "Vega Weight", 0.05, 0.2, 0.10, 0.05,
                help="Sensitivity to implied volatility",
                key="call_vega_weight"
            )
            SIGNAL_THRESHOLDS['call']['condition_weights']['oi'] = st.slider(
                "OI Weight", 0.05, 0.2, 0.10, 0.05,
                help="Open interest importance",
                key="call_oi_weight"
            )
            SIGNAL_THRESHOLDS['call']['condition_weights']['sr_proximity'] = st.slider(
                "S/R Proximity Weight", 0.05, 0.2, 0.15, 0.05,
                help="Importance of support/resistance levels",
                key="call_sr_proximity_weight"
            )
            SIGNAL_THRESHOLDS['call']['condition_weights']['sentiment'] = st.slider(
                "Sentiment Weight", 0.0, 0.1, 0.05, 0.01,
                help="Market sentiment from social media",
                key="call_sentiment_weight"
            )
      
        with col2:
            st.markdown("#### 📉 Puts")
            SIGNAL_THRESHOLDS['put']['condition_weights']['delta'] = st.slider(
                "Delta Weight", 0.1, 0.4, 0.20, 0.05,
                help="More negative delta = higher put sensitivity",
                key="put_delta_weight"
            )
            SIGNAL_THRESHOLDS['put']['condition_weights']['gamma'] = st.slider(
                "Gamma Weight", 0.1, 0.3, 0.15, 0.05,
                help="Higher gamma = faster delta acceleration",
                key="put_gamma_weight"
            )
            SIGNAL_THRESHOLDS['put']['condition_weights']['trend'] = st.slider(
                "Trend Weight", 0.1, 0.3, 0.15, 0.05,
                help="Bearish EMA alignment strength",
                key="put_trend_weight"
            )
            SIGNAL_THRESHOLDS['put']['condition_weights']['vega'] = st.slider(
                "Vega Weight", 0.05, 0.2, 0.10, 0.05,
                help="Sensitivity to implied volatility",
                key="put_vega_weight"
            )
            SIGNAL_THRESHOLDS['put']['condition_weights']['oi'] = st.slider(
                "OI Weight", 0.05, 0.2, 0.10, 0.05,
                help="Open interest importance",
                key="put_oi_weight"
            )
            SIGNAL_THRESHOLDS['put']['condition_weights']['sr_proximity'] = st.slider(
                "S/R Proximity Weight", 0.05, 0.2, 0.15, 0.05,
                help="Importance of support/resistance levels",
                key="put_sr_proximity_weight"
            )
            SIGNAL_THRESHOLDS['put']['condition_weights']['sentiment'] = st.slider(
                "Sentiment Weight", 0.0, 0.1, 0.05, 0.01,
                help="Market sentiment from social media",
                key="put_sentiment_weight"
            )
      
        st.markdown("---")
        st.markdown("**⚙️ Base Thresholds**")
      
        col1, col2 = st.columns(2)
        with col1:
            SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider("Call Delta", 0.1, 1.0, 0.5, 0.1, key="call_delta_base")
            SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider("Call Gamma", 0.01, 0.2, 0.05, 0.01, key="call_gamma_base")
            SIGNAL_THRESHOLDS['call']['volume_min'] = st.slider("Call Min Volume", 100, 5000, 1000, 100, key="call_vol_min")
      
        with col2:
            SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider("Put Delta", -1.0, -0.1, -0.5, 0.1, key="put_delta_base")
            SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider("Put Gamma", 0.01, 0.2, 0.05, 0.01, key="put_gamma_base")
            SIGNAL_THRESHOLDS['put']['volume_min'] = st.slider("Put Min Volume", 100, 5000, 1000, 100, key="put_vol_min")
  
    # Enhanced profit targets
    with st.expander("🎯 Risk Management", expanded=False):
        CONFIG['PROFIT_TARGETS']['call'] = st.slider("Call Profit Target (%)", 0.05, 0.50, 0.15, 0.01, key="call_profit")
        CONFIG['PROFIT_TARGETS']['put'] = st.slider("Put Profit Target (%)", 0.05, 0.50, 0.15, 0.01, key="put_profit")
        CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01, key="stop_loss")
      
        st.info("💡 **Tip**: Higher volatility may require wider targets")
  
    # Enhanced market status
    with st.container():
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
        except Exception:
            st.caption("**ET**: N/A")
      
        # Cache status
        if st.session_state.get('last_refresh'):
            last_update = datetime.datetime.fromtimestamp(st.session_state.last_refresh)
            time_since = int(time.time() - st.session_state.last_refresh)
            st.caption(f"**Cache**: {time_since}s ago")
  
    # Performance tips
    with st.expander("⚡ Performance Tips"):
        st.markdown("""
        **🚀 Speed Optimizations:**
        - Data cached for 5 minutes (options) / 5 minutes (stocks)
        - Vectorized signal processing (no slow loops)
        - Async API calls for parallel fetching
      
        **💰 Cost Reduction:**
        - Wait a few minutes before refreshing again
        - Avoid setting auto-refresh intervals lower than 1 minute
        - Use the app with one ticker at a time to reduce load
        """)
    # NEW: Sentiment analysis toggle
    st.subheader("🧠 Sentiment Analysis")
    enable_sentiment = st.checkbox("Enable Market Sentiment Scoring", value=False)
    if enable_sentiment:
        st.info("✅ Sentiment analysis will be factored into signals")
    else:
        st.info("ℹ️ Sentiment analysis disabled")
# NEW: Create placeholders for real-time metrics
if 'price_placeholder' not in st.session_state:
    st.session_state.price_placeholder = st.empty()
if 'status_placeholder' not in st.session_state:
    st.session_state.status_placeholder = st.empty()
if 'cache_placeholder' not in st.session_state:
    st.session_state.cache_placeholder = st.empty()
if 'refresh_placeholder' not in st.session_state:
    st.session_state.refresh_placeholder = st.empty()
# Main interface
ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper()
if ticker:
    # Enhanced header with metrics
    col1, col2, col3, col4, col5 = st.columns(5)
  
    with col1:
        st.session_state.status_placeholder = st.empty()
    with col2:
        st.session_state.price_placeholder = st.empty()
    with col3:
        st.session_state.cache_placeholder = st.empty()
    with col4:
        st.session_state.refresh_placeholder = st.empty()
    with col5:
        manual_refresh = st.button("🔄 Refresh", key="manual_refresh")
  
    # Update real-time metrics
    current_price = get_current_price(ticker)
    cache_age = int(time.time() - st.session_state.get('last_refresh', 0))
  
    # Update placeholders
    if is_market_open():
        st.session_state.status_placeholder.success("🟢 OPEN")
    elif is_premarket():
        st.session_state.status_placeholder.warning("🟡 PRE")
    else:
        st.session_state.status_placeholder.info("🔴 CLOSED")
  
    if current_price > 0:
        st.session_state.price_placeholder.metric("Price", f"${current_price:.2f}")
    else:
        st.session_state.price_placeholder.error("❌ Price Error")
  
    st.session_state.cache_placeholder.metric("Cache Age", f"{cache_age}s")
    st.session_state.refresh_placeholder.metric("Refreshes", st.session_state.refresh_counter)
  
    if manual_refresh:
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.rerun()
    # UPDATED: Enhanced Support/Resistance Analysis with better error handling
    if not st.session_state.sr_data or st.session_state.last_ticker != ticker:
        with st.spinner("🔍 Analyzing support/resistance levels..."):
            try:
                st.session_state.sr_data = analyze_support_resistance_enhanced(ticker)
                st.session_state.last_ticker = ticker
            except Exception as e:
                st.error(f"Error in S/R analysis: {str(e)}")
                st.session_state.sr_data = {}
  
    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Enhanced Signals",
        "📊 Technical Analysis",
        "📈 Support/Resistance",
        "🔍 Signal Explanations",
        "📰 Market Context",
        "📊 Free Tier Usage"
    ])
  
    with tab1:
        try:
            with st.spinner("🔄 Loading enhanced analysis..."):
                # Get stock data with indicators (cached)
                df = get_stock_data_with_indicators(ticker)
              
                if df.empty:
                    st.error("❌ Unable to fetch stock data. Please check ticker or wait for rate limits.")
                    # st.stop()  -- Removed to prevent stopping
                    st.warning("Continuing with limited functionality.")
                    return  # Exit early but don't stop the app
              
                current_price = df.iloc[-1]['Close']
                st.success(f"✅ **{ticker}** - ${current_price:.2f}")
              
                # Volatility assessment
                atr_pct = df.iloc[-1].get('ATR_pct', 0)
                if not pd.isna(atr_pct):
                    vol_status = "Low"
                    vol_color = "🟢"
                    if atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['high']:
                        vol_status = "Extreme"
                        vol_color = "🔴"
                    elif atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['medium']:
                        vol_status = "High"
                        vol_color = "🟡"
                    elif atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['low']:
                        vol_status = "Medium"
                        vol_color = "🟠"
                  
                    st.info(f"{vol_color} **Volatility**: {atr_pct*100:.2f}% ({vol_status}) - Thresholds auto-adjust")
              
                # Get full options chain with real data priority and proper UI handling
                with st.spinner("📥 Fetching REAL options data..."):
                    expiries, all_calls, all_puts = get_full_options_chain(ticker)
              
                # Handle the results and show UI controls outside of cached functions
                if not expiries:
                    st.error("❌ Unable to fetch real options data")
                  
                    # Check rate limit status
                    rate_limited = False
                    remaining_time = 0
                    if 'yf_rate_limited_until' in st.session_state:
                        remaining_time = max(0, int(st.session_state['yf_rate_limited_until'] - time.time()))
                        rate_limited = remaining_time > 0
                  
                    with st.expander("💡 Solutions for Real Data", expanded=True):
                        st.markdown("""
                        **🔧 To get real options data:**
                      
                        1. **Wait and Retry**: Rate limits typically reset in 3-5 minutes
                        2. **Try Different Time**: Options data is more available during market hours
                        3. **Use Popular Tickers**: SPY, QQQ, AAPL often have better access
                        4. **Premium Data Sources**: Consider paid APIs for reliable access
                      
                        **⏰ Rate Limit Management:**
                        - Yahoo Finance limits options requests heavily
                        - Limits are per IP address and reset periodically
                        - Try again in a few minutes
                        """)
                      
                        if rate_limited:
                            st.warning(f"⏳ Currently rate limited for {remaining_time} more seconds")
                        else:
                            st.info("✅ No active rate limits detected")
                      
                        col1, col2, col3 = st.columns(3)
                      
                        with col1:
                            if st.button("🔄 Clear Rate Limit & Retry", help="Clear rate limit status and try again"):
                                clear_rate_limit()
                      
                        with col2:
                            if st.button("⏰ Force Retry Now", help="Attempt to fetch data regardless of rate limit"):
                                if 'yf_rate_limited_until' in st.session_state:
                                    del st.session_state['yf_rate_limited_until']
                                st.cache_data.clear()
                                st.rerun()
                      
                        with col3:
                            show_demo = st.button("📊 Show Demo Data", help="Use demo data for testing interface")
                  
                    if show_demo:
                        st.session_state.force_demo = True
                        st.warning("⚠️ **DEMO DATA ONLY** - For testing the app interface")
                        expiries, calls, puts = get_fallback_options_data(ticker)
                    else:
                        # Suggest using other tabs
                        st.info("💡 **Alternative**: Use Technical Analysis or Support/Resistance tabs (work without options data)")
                        # st.stop() -- Removed
                        return  # Exit early
              
                # Only proceed if we have data (real or explicitly chosen demo)
                if expiries:
                    if st.session_state.get('force_demo', False):
                        st.warning("⚠️ Using demo data for interface testing only")
                    else:
                        st.success(f"✅ **REAL OPTIONS DATA** loaded: {len(all_calls)} calls, {len(all_puts)} puts")
                else:
                    # st.stop() -- Removed
                    return
      
                # Expiry selection
                col1, col2 = st.columns(2)
                with col1:
                    expiry_mode = st.radio(
                        "📅 Expiration Filter:",
                        ["0DTE Only", "This Week", "All Near-Term"],
                        index=1,
                        help="0DTE = Same day expiry, This Week = Within 7 days"
                    )
              
                today = datetime.date.today()
                if expiry_mode == "0DTE Only":
                    expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
                elif expiry_mode = "This Week":
                    week_end = today + datetime.timedelta(days=7)
                    expiries_to_use = [e for e in expiries if today <= datetime.datetime.strptime(e, "%Y-%m-%d").date() <= week_end]
                else:
                    expiries_to_use = expiries[:5] # Reduced from 8 to 5 expiries
              
                if not expiries_to_use:
                    st.warning(f"⚠️ No expiries available for {expiry_mode} mode.")
                    # st.stop() -- Removed
                    return
      
                with col2:
                    st.info(f"📊 Analyzing **{len(expiries_to_use)}** expiries")
                    if expiries_to_use:
                        st.caption(f"Range: {expiries_to_use[0]} to {expiries_to_use[-1]}")
              
                # Filter options by expiry
                calls_filtered = all_calls[all_calls['expiry'].isin(expiries_to_use)].copy()
                puts_filtered = all_puts[all_puts['expiry'].isin(expiries_to_use)].copy()
              
                # Strike range filter
                strike_range = st.slider(
                    "🎯 Strike Range Around Current Price ($):",
                    -50, 50, (-10, 10), 1,
                    help="Narrow range for focused analysis, wide range for comprehensive scan"
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
                    "💰 Moneyness Filter:",
                    options=["ITM", "NTM", "ATM", "OTM"],
                    default=["NTM", "ATM"],
                    help="ATM=At-the-money, NTM=Near-the-money, ITM=In-the-money, OTM=Out-of-the-money"
                )
              
                if not calls_filtered.empty:
                    calls_filtered['moneyness'] = calls_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
              
                if not puts_filtered.empty:
                    puts_filtered['moneyness'] = puts_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]
              
                st.write(f"🔍 **Filtered Options**: {len(calls_filtered)} calls, {len(puts_filtered)} puts")
              
                # Process signals using enhanced batch processing
                col1, col2 = st.columns(2)
              
                with col1:
                    st.subheader("📈 Enhanced Call Signals")
                    if not calls_filtered.empty:
                        call_signals_df = process_options_batch(calls_filtered, "call", df, current_price)
                      
                        if not call_signals_df.empty:
                            # Display top signals with enhanced info
                            display_cols = [
                                'contractSymbol', 'strike', 'lastPrice', 'volume',
                                'delta', 'gamma', 'theta', 'moneyness',
                                'score_percentage', 'profit_target', 'stop_loss',
                                'holding_period', 'is_0dte'
                            ]
                            available_cols = [col for col in display_cols if col in call_signals_df.columns]
                          
                            # Rename columns for better display
                            display_df = call_signals_df[available_cols].copy()
                            display_df = display_df.rename(columns={
                                'score_percentage': 'Score%',
                                'profit_target': 'Target',
                                'stop_loss': 'Stop',
                                'holding_period': 'Hold Period',
                                'is_0dte': '0DTE'
                            })
                          
                            st.dataframe(
                                display_df.round(3),
                                use_container_width=True,
                                hide_index=True
                            )
                          
                            # Add CSV export
                            csv = call_signals_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Call Signals CSV",
                                data=csv,
                                file_name=f"{ticker}_call_signals.csv",
                                mime="text/csv"
                            )
                          
                            # Enhanced success message with stats
                            avg_score = call_signals_df['score_percentage'].mean()
                            top_score = call_signals_df['score_percentage'].max()
                            st.success(f"✅ **{len(call_signals_df)} call signals** | Avg: {avg_score:.1f}% | Best: {top_score:.1f}%")
                          
                            # Show best signal details
                            if len(call_signals_df) > 0:
                                best_call = call_signals_df.iloc[0]
                                with st.expander(f"🏆 Best Call Signal Details ({best_call['contractSymbol']})"):
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Score", f"{best_call['score_percentage']:.1f}%")
                                        st.metric("Delta", f"{best_call['delta']:.3f}")
                                        st.metric("Vega", f"{best_call['vega']:.3f}")
                                    with col_b:
                                        st.metric("Profit Target", f"${best_call['profit_target']:.2f}")
                                        st.metric("Gamma", f"{best_call['gamma']:.3f}")
                                        st.metric("IV", f"{best_call['impliedVolatility']:.2%}")
                                    with col_c:
                                        st.metric("Stop Loss", f"${best_call['stop_loss']:.2f}")
                                        st.metric("Volume", f"{best_call['volume']:,.0f}")
                                        st.metric("OI", f"{best_call['openInterest']:,.0f}")
                        else:
                            st.info("ℹ️ No call signals found matching current criteria.")
                            st.caption("💡 Try adjusting strike range, moneyness filter, or threshold weights")
                    else:
                        st.info("ℹ️ No call options available for selected filters.")
              
                with col2:
                    st.subheader("📉 Enhanced Put Signals")
                    if not puts_filtered.empty:
                        put_signals_df = process_options_batch(puts_filtered, "put", df, current_price)
                      
                        if not put_signals_df.empty:
                            # Display top signals with enhanced info
                            display_cols = [
                                'contractSymbol', 'strike', 'lastPrice', 'volume',
                                'delta', 'gamma', 'theta', 'moneyness',
                                'score_percentage', 'profit_target', 'stop_loss',
                                'holding_period', 'is_0dte'
                            ]
                            available_cols = [col for col in display_cols if col in put_signals_df.columns]
                          
                            # Rename columns for better display
                            display_df = put_signals_df[available_cols].copy()
                            display_df = display_df.rename(columns={
                                'score_percentage': 'Score%',
                                'profit_target': 'Target',
                                'stop_loss': 'Stop',
                                'holding_period': 'Hold Period',
                                'is_0dte': '0DTE'
                            })
                          
                            st.dataframe(
                                display_df.round(3),
                                use_container_width=True,
                                hide_index=True
                            )
                          
                            # Add CSV export
                            csv = put_signals_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Put Signals CSV",
                                data=csv,
                                file_name=f"{ticker}_put_signals.csv",
                                mime="text/csv"
                            )
                          
                            # Enhanced success message with stats
                            avg_score = put_signals_df['score_percentage'].mean()
                            top_score = put_signals_df['score_percentage'].max()
                            st.success(f"✅ **{len(put_signals_df)} put signals** | Avg: {avg_score:.1f}% | Best: {top_score:.1f}%")
                          
                            # Show best signal details
                            if len(put_signals_df) > 0:
                                best_put = put_signals_df.iloc[0]
                                with st.expander(f"🏆 Best Put Signal Details ({best_put['contractSymbol']})"):
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Score", f"{best_put['score_percentage']:.1f}%")
                                        st.metric("Delta", f"{best_put['delta']:.3f}")
                                        st.metric("Vega", f"{best_put['vega']:.3f}")
                                    with col_b:
                                        st.metric("Profit Target", f"${best_put['profit_target']:.2f}")
                                        st.metric("Gamma", f"{best_put['gamma']:.3f}")
                                        st.metric("IV", f"{best_put['impliedVolatility']:.2%}")
                                    with col_c:
                                        st.metric("Stop Loss", f"${best_put['stop_loss']:.2f}")
                                        st.metric("Volume", f"{best_put['volume']:,.0f}")
                                        st.metric("OI", f"{best_put['openInterest']:,.0f}")
                        else:
                            st.info("ℹ️ No put signals found matching current criteria.")
                            st.caption("💡 Try adjusting strike range, moneyness filter, or threshold weights")
                    else:
                        st.info("ℹ️ No put options available for selected filters.")
              
                # Enhanced scanner scores
                call_score = calculate_scanner_score(df, 'call')
                put_score = calculate_scanner_score(df, 'put')
              
                st.markdown("---")
                st.subheader("🧠 Technical Scanner Scores")
              
                col1, col2, col3 = st.columns(3)
                with col1:
                    score_color = "🟢" if call_score >= 70 else "🟡" if call_score >= 40 else "🔴"
                    st.metric("📈 Call Scanner", f"{call_score:.1f}%", help="Based on bullish technical indicators")
                    st.caption(f"{score_color} {'Strong' if call_score >= 70 else 'Moderate' if call_score >= 40 else 'Weak'} bullish setup")
              
                with col2:
                    score_color = "🟢" if put_score >= 70 else "🟡" if put_score >= 40 else "🔴"
                    st.metric("📉 Put Scanner", f"{put_score:.1f}%", help="Based on bearish technical indicators")
                    st.caption(f"{score_color} {'Strong' if put_score >= 70 else 'Moderate' if put_score >= 40 else 'Weak'} bearish setup")
              
                with col3:
                    directional_bias = "Bullish" if call_score > put_score else "Bearish" if put_score > call_score else "Neutral"
                    bias_strength = abs(call_score - put_score)
                    st.metric("🎯 Directional Bias", directional_bias)
                    st.caption(f"Strength: {bias_strength:.1f}% difference")
              
        except Exception as e:
            st.error(f"❌ Error in signal analysis: {str(e)}")
            st.error("Please try refreshing or check your ticker symbol.")
  
    with tab2:
        try:
            if 'df' not in locals():
                df = get_stock_data_with_indicators(ticker)
          
            if not df.empty:
                st.subheader("📊 Technical Analysis Dashboard")
              
                # Market session indicator
                if is_premarket():
                    st.info("🔔 Currently showing PREMARKET data")
                elif not is_market_open():
                    st.info("🔔 Showing AFTER-HOURS data")
                else:
                    st.success("🔔 Showing REGULAR HOURS data")
              
                latest = df.iloc[-1]
              
                # Enhanced metrics display
                col1, col2, col3, col4, col5, col6 = st.columns(6)
              
                with col1:
                    st.metric("Current Price", f"${latest['Close']:.2f}")
              
                with col2:
                    ema_9 = latest['EMA_9']
                    if not pd.isna(ema_9):
                        trend_9 = "🔺" if latest['Close'] > ema_9 else "🔻"
                        st.metric("EMA 9", f"${ema_9:.2f} {trend_9}")
                    else:
                        st.metric("EMA 9", "N/A")
              
                with col3:
                    ema_20 = latest['EMA_20']
                    if not pd.isna(ema_20):
                        trend_20 = "🔺" if latest['Close'] > ema_20 else "🔻"
                        st.metric("EMA 20", f"${ema_20:.2f} {trend_20}")
                    else:
                        st.metric("EMA 20", "N/A")
              
                with col4:
                    rsi = latest['RSI']
                    if not pd.isna(rsi):
                        rsi_status = "🔥" if rsi > 70 else "❄️" if rsi < 30 else "⚖️"
                        st.metric("RSI", f"{rsi:.1f} {rsi_status}")
                    else:
                        st.metric("RSI", "N/A")
              
                with col5:
                    atr_pct = latest['ATR_pct']
                    if not pd.isna(atr_pct):
                        vol_emoji = "🌪️" if atr_pct > 0.05 else "📊" if atr_pct > 0.02 else "😴"
                        st.metric("Volatility", f"{atr_pct*100:.2f}% {vol_emoji}")
                    else:
                        st.metric("Volatility", "N/A")
              
                with col6:
                    volume_ratio = latest['Volume'] / latest['avg_vol'] if not pd.isna(latest['avg_vol']) else 1
                    vol_emoji = "🚀" if volume_ratio > 2 else "📈" if volume_ratio > 1.5 else "📊"
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x {vol_emoji}")
              
                # Recent data table with enhanced formatting
                st.subheader("📋 Recent Market Data")
                display_df = df.tail(10)[['Datetime', 'Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'ATR_pct', 'Volume', 'avg_vol']].copy()
              
                if 'ATR_pct' in display_df.columns:
                    display_df['ATR_pct'] = display_df['ATR_pct'] * 100
              
                display_df['Volume Ratio'] = display_df['Volume'] / display_df['avg_vol']
                display_df = display_df.round(2)
              
                # Format datetime for better readability
                display_df['Time'] = display_df['Datetime'].dt.strftime('%H:%M')
              
                final_cols = ['Time', 'Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'ATR_pct', 'Volume Ratio']
                available_final_cols = [col for col in final_cols if col in display_df.columns]
              
                st.dataframe(
                    display_df[available_final_cols].rename(columns={'ATR_pct': 'ATR%'}),
                    use_container_width=True,
                    hide_index=True
                )
              
                # Enhanced interactive chart
                st.subheader("📈 Interactive Price Chart")
                chart_fig = create_stock_chart(df, st.session_state.sr_data)
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)
                else:
                    st.warning("⚠️ Unable to create chart. Chart data may be insufficient.")
              
        except Exception as e:
            st.error(f"❌ Error in Technical Analysis: {str(e)}")
  
    with tab3:
        st.subheader("📈 Multi-Timeframe Support/Resistance Analysis")
        st.info("Key levels for options trading strategies. Scalping: 1min/5min | Intraday: 15min/30min/1h")
      
        if not st.session_state.sr_data:
            st.warning("No support/resistance data available. Please try refreshing.")
        else:
            # Display visualization using enhanced function
            sr_fig = plot_sr_levels_enhanced(st.session_state.sr_data, current_price)
            if sr_fig:
                st.plotly_chart(sr_fig, use_container_width=True)
          
            # Display detailed levels
            st.subheader("Detailed Levels by Timeframe")
          
            # Scalping timeframes
            st.markdown("#### 🚀 Scalping Timeframes (Short-Term Trades)")
            col1, col2 = st.columns(2)
            with col1:
                if '1min' in st.session_state.sr_data:
                    sr = st.session_state.sr_data['1min']
                    st.markdown("**1 Minute**")
                    st.markdown(f"Sensitivity: {sr['sensitivity']*100:.2f}%")
                  
                    st.markdown("**Support Levels**")
                    for level in sr['support']:
                        distance = abs(level - current_price) / current_price * 100
                        st.markdown(f"- ${level:.2f} ({distance:.1f}% away)")
                  
                    st.markdown("**Resistance Levels**")
                    for level in sr['resistance']:
                        distance = abs(level - current_price) / current_price * 100
                        st.markdown(f"- ${level:.2f} ({distance:.1f}% away)")
          
            # Trading strategy guidance
            st.subheader("📝 Trading Strategy Guidance")
            with st.expander("How to use support/resistance for options trading", expanded=True):
                st.markdown("""
                **Scalping Strategies (1min/5min levels):**
                - Use for quick, short-term trades (minutes to hours)
                - Look for options with strikes near key levels for breakout plays
                - Combine with high delta options for directional plays
                - Ideal for 0DTE or same-day expiration options
              
                **Intraday Strategies (15min/1h levels):**
                - Use for swing trades (hours to days)
                - Look for options with strikes between support/resistance levels for range-bound strategies
                - Combine with technical indicators for confirmation
                - Ideal for weekly expiration options
              
                **General Tips:**
                1. **Breakout Trading**: Buy calls when price breaks above resistance, puts when below support
                2. **Bounce Trading**: Buy calls near support, puts near resistance
                3. **Range Trading**: Sell options when price is between support/resistance
                4. **Straddles/Strangles**: Use when expecting volatility breakout
                """)
  
    with tab4:
        st.subheader("🔍 Signal Explanations & Methodology")
      
        # Show current configuration
        st.markdown("### ⚙️ Current Configuration")
      
        col1, col2 = st.columns(2)
      
        with col1:
            st.markdown("**📈 Call Signal Weights**")
            call_weights = SIGNAL_THRESHOLDS['call']['condition_weights']
            for condition, weight in call_weights.items():
                st.write(f"• {condition.title()}: {weight:.1%}")
          
            st.markdown("**🎯 Profit Targets**")
            st.write(f"• Call Target: {CONFIG['PROFIT_TARGETS']['call']:.1%}")
            st.write(f"• Put Target: {CONFIG['PROFIT_TARGETS']['put']:.1%}")
            st.write(f"• Stop Loss: {CONFIG['PROFIT_TARGETS']['stop_loss']:.1%}")
      
        with col2:
            st.markdown("**📉 Put Signal Weights**")
            put_weights = SIGNAL_THRESHOLDS['put']['condition_weights']
            for condition, weight in put_weights.items():
                st.write(f"• {condition.title()}: {weight:.1%}")
          
            st.markdown("**⏱️ Cache Settings**")
            st.write(f"• Options Cache: {CONFIG['CACHE_TTL']}s")
            st.write(f"• Stock Cache: {CONFIG['STOCK_CACHE_TTL']}s")
            st.write(f"• Min Refresh: {CONFIG['MIN_REFRESH_INTERVAL']}s")
      
        # Methodology explanation
        st.markdown("### 🧠 Signal Methodology")
      
        with st.expander("📊 How Signals Are Generated", expanded=True):
            st.markdown("""
            **🏋️ Weighted Scoring System:**
            - Each condition gets a weight (importance factor)
            - Final score = sum of (condition_passed × weight)
            - Scores range from 0-100%
          
            **📈 Call Signal Conditions:**
            1. **Delta** ≥ threshold (price sensitivity)
            2. **Gamma** ≥ threshold (acceleration potential)
            3. **Theta** ≤ threshold (time decay acceptable)
            4. **Trend**: Price > EMA9 > EMA20 (bullish alignment)
            5. **Momentum**: RSI > 50 (bullish momentum)
            6. **Volume** > minimum (sufficient liquidity)
            7. **Vega/IV**: Vega > 0.1 and IV < 35% (volatility alignment)
            8. **Open Interest**: OI > 2× volume min (liquidity/conviction)
            9. **S/R Proximity**: Strike near support (improves breakout odds)
            10. **Sentiment**: Positive social sentiment
          
            **📉 Put Signal Conditions:**
            1. **Delta** ≤ threshold (negative price sensitivity)
            2. **Gamma** ≥ threshold (acceleration potential)
            3. **Theta** ≤ threshold (time decay acceptable)
            4. **Trend**: Price < EMA9 < EMA20 (bearish alignment)
            5. **Momentum**: RSI < 50 (bearish momentum)
            6. **Volume** > minimum (sufficient liquidity)
            7. **Vega/IV**: Vega > 0.1 and IV < 35% (volatility alignment)
            8. **Open Interest**: OI > 2× volume min (liquidity/conviction)
            9. **S/R Proximity**: Strike near resistance (improves reversal odds)
            10. **Sentiment**: Negative social sentiment
            """)
      
        with st.expander("🎯 Dynamic Threshold Adjustments", expanded=False):
            st.markdown("""
            **📊 Volatility Adjustments:**
            - Higher volatility → Higher delta requirements
            - Higher volatility → Higher gamma requirements
            - Volatility measured by ATR% (Average True Range)
          
            **🕐 Market Condition Adjustments:**
            - **Premarket/Early Market**: Lower volume requirements, higher delta requirements
            - **0DTE Options**: Higher delta requirements, lower gamma requirements
            - **High Volatility**: All thresholds scale up proportionally
          
            **⏳ DTE & Moneyness Adjustments:**
            - **0DTE/1DTE**: Boost gamma requirements, tighten theta tolerance
            - **Longer-term**: Lower gamma requirements, more theta tolerance
            - **ATM Options**: Require higher gamma
            - **OTM Options**: Require higher volume/OI for conviction
          
            **💡 Why Dynamic Thresholds:**
            - Static thresholds fail in changing market conditions
            - Volatile markets need higher Greeks for same profit potential
            - Different market sessions have different liquidity characteristics
            """)
      
        with st.expander("⚡ Performance Optimizations", expanded=False):
            st.markdown("""
            **🚀 Speed Improvements:**
            - **SmartCaching**: Options cached for 5 min, stocks for 5 min
            - Batch processing: Vectorized operations instead of slow loops
            - Async API calls: Parallel fetching from multiple sources
          
            **💰 Cost Reduction:**
            - Full chain caching: Fetch all expiries once, filter locally
            - Conservative defaults: 120s refresh intervals prevent overuse
            - Fallback logic: Yahoo Finance backup when Polygon unavailable
          
            **📊 Better Analysis:**
            - Weighted scoring: Most important factors weighted highest
            - Detailed explanations: See exactly why signals pass/fail
            - Multiple timeframes: 0DTE, weekly, monthly analysis
            """)
      
        # Performance metrics
        if st.session_state.get('refresh_counter', 0) > 0:
            st.markdown("### 📈 Session Performance")
          
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Refreshes", st.session_state.refresh_counter)
            with col2:
                avg_interval = (time.time() - st.session_state.get('session_start', time.time())) / max(st.session_state.refresh_counter, 1)
                st.metric("Avg Refresh Interval", f"{avg_interval:.0f}s")
            with col3:
                cache_hit_rate = 85 # Estimated based on caching strategy
                st.metric("Est. Cache Hit Rate", f"{cache_hit_rate}%")
  
    with tab5:
        st.subheader("📰 Market Context & News")
      
        try:
            # Company info section
            stock = yf.Ticker(ticker)
          
            # Basic company information
            with st.expander("🏢 Company Overview", expanded=True):
                try:
                    info = stock.info
                    if info:
                        col1, col2, col3, col4 = st.columns(4)
                      
                        with col1:
                            if 'longName' in info:
                                st.write(f"**Company**: {info['longName']}")
                            if 'sector' in info:
                                st.write(f"**Sector**: {info['sector']}")
                      
                        with col2:
                            if 'marketCap' in info and info['marketCap']:
                                market_cap = info['marketCap']
                                if market_cap > 1e12:
                                    st.write(f"**Market Cap**: ${market_cap/1e12:.2f}T")
                                elif market_cap > 1e9:
                                    st.write(f"**Market Cap**: ${market_cap/1e9:.2f}B")
                                else:
                                    st.write(f"**Market Cap**: ${market_cap/1e6:.2f}M")
                      
                        with col3:
                            if 'beta' in info and info['beta']:
                                st.write(f"**Beta**: {info['beta']:.2f}")
                            if 'trailingPE' in info and info['trailingPE']:
                                st.write(f"**P/E Ratio**: {info['trailingPE']:.2f}")
                      
                        with col4:
                            if 'averageVolume' in info:
                                avg_vol = info['averageVolume']
                                if avg_vol > 1e6:
                                    st.write(f"**Avg Volume**: {avg_vol/1e6:.1f}M")
                                else:
                                    st.write(f"**Avg Volume**: {avg_vol/1e3:.0f}K")
                except Exception as e:
                    st.warning(f"⚠️ Company info unavailable: {str(e)}")
          
            # Recent news
            with st.expander("📰 Recent News", expanded=False):
                try:
                    news = stock.news
                    if news:
                        for i, item in enumerate(news[:5]): # Limit to 5 most recent
                            title = item.get('title', 'Untitled')
                            publisher = item.get('publisher', 'Unknown')
                            link = item.get('link', '#')
                            summary = item.get('summary', 'No summary available')
                          
                            # Format publish time
                            publish_time = item.get('providerPublishTime', 'Unknown')
                            if isinstance(publish_time, (int, float)):
                                try:
                                    publish_time = datetime.datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                                except:
                                    publish_time = 'Unknown'
                          
                            st.markdown(f"**{i+1}. {title}**")
                            st.write(f"📅 {publish_time} | 📰 {publisher}")
                            if link != '#':
                                st.markdown(f"🔗 [Read Article]({link})")
                            st.write(summary[:200] + "..." if len(summary) > 200 else summary)
                            st.markdown("---")
                    else:
                        st.info("ℹ️ No recent news available")
                except Exception as e:
                    st.warning(f"⚠️ News unavailable: {str(e)}")
          
            # Upcoming events/earnings
            with st.expander("📅 Upcoming Events", expanded=False):
                try:
                    calendar = stock.calendar
                    if calendar is not None and not calendar.empty:
                        st.dataframe(calendar, use_container_width=True)
                    else:
                        st.info("ℹ️ No upcoming events scheduled")
                except Exception as e:
                    st.warning(f"⚠️ Calendar unavailable: {str(e)}")
          
            # Market context
            with st.expander("🎯 Trading Context", expanded=True):
                st.markdown("""
                **📊 Current Market Conditions:**
                - Check VIX levels for overall market fear/greed
                - Monitor major indices (SPY, QQQ, IWM) for directional bias
                - Watch for economic events that could impact volatility
              
                **⚠️ Risk Considerations:**
                - Options lose value due to time decay (theta)
                - High volatility can increase option prices rapidly
                - Earnings announcements create significant price movements
                - Market holidays affect option expiration schedules
              
                **💡 Best Practices:**
                - Never risk more than you can afford to lose
                - Use stop losses to limit downside
                - Take profits when targets are reached
                - Avoid holding 0DTE options into close
                """)
              
                # Add market warnings based on conditions
                if is_premarket():
                    st.warning("⚠️ **PREMARKET TRADING**: Lower liquidity, wider spreads expected")
                elif not is_market_open():
                    st.info("ℹ️ **MARKET CLOSED**: Signals based on last session data")
              
                # Add volatility warnings
                if 'df' in locals() and not df.empty:
                    latest_atr = df.iloc[-1].get('ATR_pct', 0)
                    if not pd.isna(latest_atr) and latest_atr > CONFIG['VOLATILITY_THRESHOLDS']['high']:
                        st.warning("🌪️ **HIGH VOLATILITY**: Increased risk and opportunity. Use wider stops.")
      
        except Exception as e:
            st.error(f"❌ Error loading market context: {str(e)}")
  
    with tab6:
        st.subheader("📰 Free Tier Usage Dashboard")
      
        if not st.session_state.API_CALL_LOG:
            st.info("No API calls recorded yet")
        else:
            now = time.time()
          
            # Calculate usage
            av_usage_1min = len([t for t in st.session_state.API_CALL_LOG
                                if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 60])
            av_usage_1hr = len([t for t in st.session_state.API_CALL_LOG
                               if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 3600])
          
            fmp_usage_1hr = len([t for t in st.session_state.API_CALL_LOG
                                if t['source'] == "FMP" and now - t['timestamp'] < 3600])
            fmp_usage_24hr = len([t for t in st.session_state.API_CALL_LOG
                                 if t['source'] == "FMP" and now - t['timestamp'] < 86400])
          
            iex_usage_1hr = len([t for t in st.session_state.API_CALL_LOG
                                if t['source'] == "IEX" and now - t['timestamp'] < 3600])
            iex_usage_24hr = len([t for t in st.session_state.API_CALL_LOG
                                 if t['source'] == "IEX" and now - t['timestamp'] < 86400])
         
            # Display gauges
            col1, col2, col3 = st.columns(3)
         
            with col1:
                st.subheader("Alpha Vantage")
                st.metric("Last Minute", f"{av_usage_1min}/5", "per minute")
                st.metric("Last Hour", f"{av_usage_1hr}/300", "per hour")
                st.progress(min(1.0, av_usage_1min/5), text=f"{min(100, av_usage_1min/5*100):.0f}% of minute limit")
         
            with col2:
                st.subheader("Financial Modeling Prep")
                st.metric("Last Hour", f"{fmp_usage_1hr}/10", "per hour")
                st.metric("Last 24 Hours", f"{fmp_usage_24hr}/250", "per day")
                st.progress(min(1.0, fmp_usage_1hr/10), text=f"{min(100, fmp_usage_1hr/10*100):.0f}% of hourly limit")
         
            with col3:
                st.subheader("IEX Cloud")
                st.metric("Last Hour", f"{iex_usage_1hr}/69", "per hour")
                st.metric("Last 24 Hours", f"{iex_usage_24hr}/1667", "per day")
                st.progress(min(1.0, iex_usage_1hr/69), text=f"{min(100, iex_usage_1hr/69*100):.0f}% of hourly limit")
         
            # Usage history chart
            st.subheader("Usage History")
         
            # Create a DataFrame for visualization
            log_df = pd.DataFrame(st.session_state.API_CALL_LOG)
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], unit='s')
            log_df['time'] = log_df['timestamp'].dt.floor('min')
         
            # Group by source and time
            usage_df = log_df.groupby(['source', pd.Grouper(key='time', freq='5min')]).size().unstack(fill_value=0)
         
            # Fill missing time periods
            if not usage_df.empty:
                all_times = pd.date_range(
                    start=log_df['timestamp'].min().floor('5min'),
                    end=log_df['timestamp'].max().ceil('5min'),
                    freq='5min'
                )
                usage_df = usage_df.reindex(all_times, axis=1, fill_value=0)
             
                # Plot
                fig = go.Figure()
                for source in usage_df.index:
                    fig.add_trace(go.Scatter(
                        x=usage_df.columns,
                        y=usage_df.loc[source],
                        mode='lines+markers',
                        name=source,
                        stackgroup='one'
                    ))
             
                fig.update_layout(
                    title='API Calls Over Time',
                    xaxis_title='Time',
                    yaxis_title='API Calls',
                    hovermode='x unified',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No API calls recorded in the selected time range")
         
            st.info("💡 Usage resets over time. Add more free API keys to increase capacity")
else:
    st.info("Please enter a stock ticker to begin analysis.")
    
    # Display help information
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

# Auto-refresh logic
if st.session_state.auto_refresh_enabled and ticker:
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed > st.session_state.refresh_interval:
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.cache_data.clear()
        st.rerun()

# Unit Tests
class TestOptionsAnalyzer(unittest.TestCase):
    def test_generate_enhanced_signal_call(self):
        # Mock option data for call
        option = pd.Series({
            'delta': 0.6,
            'gamma': 0.06,
            'theta': 0.04,
            'volume': 1500,
            'openInterest': 3000,
            'vega': 0.15,
            'impliedVolatility': 0.30,
            'strike': 100,
            'expiry': (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        })
        # Mock stock data
        stock_df = pd.DataFrame([{
            'Close': 101,
            'EMA_9': 100.5,
            'EMA_20': 100,
            'RSI': 55,
            'ATR_pct': 0.02,
            'Volume': 100000,
            'avg_vol': 50000
        }])
        # Assume sr_data with support at 99
        st.session_state.sr_data = {
            '5min': {
                'support': [99.0],
                'resistance': [105.0]
            }
        }
        result = generate_enhanced_signal(option, 'call', stock_df, False)
        self.assertTrue(result['signal'])
        self.assertGreater(result['score'], 0.7)
        self.assertEqual(len(result['explanations']), 10) # Updated number of conditions
    def test_get_sentiment_score(self):
        # Test fallback
        sentiment = get_sentiment_score('AAPL')
        self.assertGreaterEqual(sentiment, -1.0)
        self.assertLessEqual(sentiment, 1.0)
    def test_calculate_dynamic_thresholds(self):
        stock_data = pd.Series({'ATR_pct': 0.03})
        option = {'expiry': datetime.date.today().strftime('%Y-%m-%d')}
        current_price = 100
        thresholds = calculate_dynamic_thresholds(stock_data, 'call', option, current_price)
        self.assertGreater(thresholds['delta_min'], 0.5)
        self.assertGreater(thresholds['gamma_min'], 0.05)

if __name__ == '__main__':
    unittest.main()
