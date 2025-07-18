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
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

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
    }
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

def get_current_price(ticker: str) -> float:
    """Get the most current price including premarket"""
    try:
        stock = yf.Ticker(ticker)
        # Get today's data including premarket
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
            # Check for rate limit
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
        
        # Use auto_adjust=True to suppress warnings
        # Include pre/post market data
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
        
        # Convert to numeric and handle any nested structures
        for col in required_cols:
            if col in data.columns:
                # Handle nested data structures
                if hasattr(data[col].iloc[0], '__len__') and not isinstance(data[col].iloc[0], str):
                    data[col] = data[col].apply(lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x)
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Remove rows with NaN in essential columns
        data = data.dropna(subset=required_cols)
        
        if len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient data points ({len(data)}). Need at least {CONFIG['MIN_DATA_POINTS']}.")
            return pd.DataFrame()
        
        # FIXED TIMEZONE HANDLING
        eastern = pytz.timezone('US/Eastern')
        
        # If index is timezone-naive, localize as UTC first
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc)
        
        # Convert to Eastern time
        data.index = data.index.tz_convert(eastern)
        
        # Add premarket flag
        data['premarket'] = False
        
        # Identify premarket sessions (4:00 AM to 9:30 AM Eastern)
        premarket_mask = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        data.loc[premarket_mask, 'premarket'] = True
        
        return data.reset_index(drop=False)
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate more accurate volume averages with separate premarket handling"""
    if df.empty:
        return df
    
    # Calculate volume averages separately for premarket and regular sessions
    df['avg_vol'] = np.nan
    
    for date, group in df.groupby(df['Datetime'].dt.date):
        # Regular session
        regular = group[~group['premarket']]
        if not regular.empty:
            # Use expanding average during market hours
            regular_avg_vol = regular['Volume'].expanding(min_periods=1).mean()
            df.loc[regular.index, 'avg_vol'] = regular_avg_vol
        
        # Premarket session
        premarket = group[group['premarket']]
        if not premarket.empty:
            # Use cumulative average for premarket
            premarket_avg_vol = premarket['Volume'].expanding(min_periods=1).mean()
            df.loc[premarket.index, 'avg_vol'] = premarket_avg_vol
    
    # Fill any remaining NaN with the overall average
    overall_avg = df['Volume'].mean()
    df['avg_vol'] = df['avg_vol'].fillna(overall_avg)
    
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators with comprehensive error handling"""
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

        # Calculate indicators with minimum data requirements
        try:
            # EMA indicators
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
                
            # RSI
            if len(close) >= 14:
                rsi = RSIIndicator(close=close, window=14)
                df['RSI'] = rsi.rsi()
            else:
                df['RSI'] = np.nan

            # VWAP - calculate separately for premarket and regular session
            df['VWAP'] = np.nan
            df['avg_vol'] = np.nan
            
            # Calculate VWAP separately for each session
            for session, group in df.groupby(pd.Grouper(key='Datetime', freq='D')):
                if group.empty:
                    continue
                
                # Split into premarket and regular session
                premarket = group[group['premarket']]
                regular = group[~group['premarket']]
                
                # Calculate VWAP for regular session
                if not regular.empty:
                    typical_price = (regular['High'] + regular['Low'] + regular['Close']) / 3
                    vwap_cumsum = (regular['Volume'] * typical_price).cumsum()
                    volume_cumsum = regular['Volume'].cumsum()
                    regular_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                    df.loc[regular.index, 'VWAP'] = regular_vwap
                
                # For premarket, use the previous day's close as reference
                if not premarket.empty:
                    # Get previous day's close
                    prev_day = session - datetime.timedelta(days=1)
                    prev_close = df[df['Datetime'].dt.date == prev_day.date()]['Close'].iloc[-1] if not df[df['Datetime'].dt.date == prev_day.date()].empty else premarket['Close'].iloc[0]
                    
                    typical_price = (premarket['High'] + premarket['Low'] + premarket['Close']) / 3
                    vwap_cumsum = (premarket['Volume'] * typical_price).cumsum()
                    volume_cumsum = premarket['Volume'].cumsum()
                    premarket_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                    df.loc[premarket.index, 'VWAP'] = premarket_vwap
                
            # ATR for volatility measurement
            if len(close) >= 14:
                atr = AverageTrueRange(high=high, low=low, close=close, window=14)
                df['ATR'] = atr.average_true_range()
                df['ATR_pct'] = df['ATR'] / close  # ATR as % of price
            else:
                df['ATR'] = np.nan
                df['ATR_pct'] = np.nan
                
        except Exception as e:
            st.error(f"Error computing indicators: {str(e)}")
            return pd.DataFrame()
        
        # Calculate volume averages
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
            
            # Add expiry information
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            
            # Validate required columns exist
            required_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
            
            for df_name, df in [('calls', calls), ('puts', puts)]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing columns in {df_name} for {expiry}: {missing_cols}")
                    # Add placeholder Greeks so we can calculate them later
                    if 'delta' not in df.columns:
                        df['delta'] = np.nan
                    if 'gamma' not in df.columns:
                        df['gamma'] = np.nan
                    if 'theta' not in df.columns:
                        df['theta'] = np.nan
                else:
                    # Ensure Greeks are present
                    if 'delta' not in df.columns:
                        df['delta'] = np.nan
                    if 'gamma' not in df.columns:
                        df['gamma'] = np.nan
                    if 'theta' not in df.columns:
                        df['theta'] = np.nan
            
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
            
            # Add delay after successful fetch
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
    # Simple approximation for delta
    moneyness = spot_price / option['strike']
    
    if option['contractSymbol'].startswith('C'):
        # For calls
        if moneyness > 1.03:  # Deep ITM
            delta = 0.95
            gamma = 0.01
        elif moneyness > 1.0:  # Slightly ITM
            delta = 0.65
            gamma = 0.05
        elif moneyness > 0.97:  # Near the money
            delta = 0.50
            gamma = 0.08
        else:  # OTM
            delta = 0.35
            gamma = 0.05
    else:
        # For puts
        if moneyness < 0.97:  # Deep ITM
            delta = -0.95
            gamma = 0.01
        elif moneyness < 1.0:  # Slightly ITM
            delta = -0.65
            gamma = 0.05
        elif moneyness < 1.03:  # Near the money
            delta = -0.50
            gamma = 0.08
        else:  # OTM
            delta = -0.35
            gamma = 0.05
    
    # Simple approximation for theta (time decay)
    # Higher for near-term options, especially 0DTE
    theta = 0.05 if "today" in option['expiry'] else 0.02
    
    return delta, gamma, theta

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    """Validate that option has required data for analysis"""
    required_fields = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
    
    for field in required_fields:
        if field not in option or pd.isna(option[field]):
            return False
    
    # Check for reasonable values
    if option['lastPrice'] <= 0:
        return False
    
    # Calculate Greeks if missing
    if pd.isna(option.get('delta')) or pd.isna(option.get('gamma')) or pd.isna(option.get('theta')):
        # Use approximate method if Greeks are missing
        delta, gamma, theta = calculate_approximate_greeks(option, spot_price)
        option['delta'] = delta
        option['gamma'] = gamma
        option['theta'] = theta
    
    # Check Greeks are valid
    if pd.isna(option['delta']) or pd.isna(option['gamma']) or pd.isna(option['theta']):
        return False
    
    return True

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
    """Calculate dynamic thresholds with enhanced volatility response"""
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    
    # Get volatility measure (ATR as % of price)
    volatility = stock_data.get('ATR_pct', 0.02)  # Default to 2% if missing
    
    # Enhanced volatility multiplier
    vol_multiplier = 1 + (volatility * 100)
    
    # Adjust delta threshold based on volatility
    if side == 'call':
        thresholds['delta_min'] = max(0.3, min(0.8, 
            thresholds['delta_base'] * vol_multiplier
        ))
    else:
        thresholds['delta_max'] = min(-0.3, max(-0.8, 
            thresholds['delta_base'] * vol_multiplier
        ))
    
    # More responsive gamma adjustment
    thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + 
        thresholds['gamma_vol_multiplier'] * (volatility * 200)  # More sensitive to volatility
    )
    
    # Volume multiplier with floor
    thresholds['volume_multiplier'] = max(0.8, min(2.5,
        thresholds['volume_multiplier_base'] * (1 + 
            thresholds['volume_vol_multiplier'] * (volatility * 150)
        )
    ))
    
    # Apply early market adjustments
    if is_premarket() or is_early_market():
        if side == 'call':
            thresholds['delta_min'] = 0.35  # Lower call delta threshold
        else:
            thresholds['delta_max'] = -0.35  # Higher put delta threshold
        thresholds['volume_multiplier'] *= 0.6  # Relax volume requirement
        thresholds['gamma_min'] *= 0.8  # Slightly relax gamma in early market
    
    # Special handling for 0DTE options
    if is_0dte:
        # Relax thresholds for 0DTE options
        thresholds['volume_multiplier'] *= 0.7  # Relax volume requirement
        thresholds['gamma_min'] *= 0.7  # Relax gamma requirement
        
        if side == 'call':
            thresholds['delta_min'] = max(0.4, thresholds['delta_min'])
        else:
            thresholds['delta_max'] = min(-0.4, thresholds['delta_max'])
    
    return thresholds

def calculate_holding_period(option: pd.Series, spot_price: float) -> str:
    """Determine optimal holding period based on option characteristics"""
    # Calculate time to expiration
    expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - datetime.date.today()).days
    
    # Determine holding period based on option type and time
    if days_to_expiry == 0:  # 0DTE
        return "Intraday (Exit before 3:30 PM)"
    
    # Calculate intrinsic value
    if option['contractSymbol'].startswith('C'):
        intrinsic_value = max(0, spot_price - option['strike'])
    else:
        intrinsic_value = max(0, option['strike'] - spot_price)
    
    # Determine holding strategy based on intrinsic value and theta
    if intrinsic_value > 0:  # In the money
        if option['theta'] < -0.1:  # High time decay
            return "1-2 days (Scalp quickly)"
        else:
            return "3-5 days (Swing trade)"
    else:  # Out of the money
        if days_to_expiry <= 3:
            return "1 day (Gamma play)"
        else:
            return "3-7 days (Wait for move)"

def calculate_profit_targets(option: pd.Series) -> Tuple[float, float]:
    """Calculate profit targets and stop loss levels"""
    entry_price = option['lastPrice']
    profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['call' if option['contractSymbol'].startswith('C') else 'put'])
    stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
    return profit_target, stop_loss

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    """Generate trading signal with detailed analysis using dynamic thresholds"""
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available'}
    
    # Get current price for validation
    current_price = stock_df.iloc[-1]['Close']
    
    if not validate_option_data(option, current_price):
        return {'signal': False, 'reason': 'Insufficient option data'}
    
    latest = stock_df.iloc[-1]
    
    try:
        # Calculate DYNAMIC thresholds based on current market conditions
        thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
        
        # Extract option Greeks
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        option_volume = float(option['volume'])
        
        # Extract stock data
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume
        
        # Check conditions based on side
        conditions = []
        
        if side == "call":
            # Volume condition: Use absolute min volume instead of relative
            volume_ok = option_volume > thresholds['volume_min']  # Absolute volume threshold
            
            conditions = [
                (delta >= thresholds['delta_min'], f"Delta >= {thresholds['delta_min']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']:.1f}", rsi),
                (vwap is not None and close > vwap, "Price > VWAP", f"{close:.2f} > {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}")
            ]
        else:  # put
            # Volume condition: Use absolute min volume instead of relative
            volume_ok = option_volume > thresholds['volume_min']  # Absolute volume threshold
            
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']:.1f}", rsi),
                (vwap is not None and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f} > {thresholds['volume_min']}")
            ]
        
        # Check all conditions
        passed_conditions = [desc for passed, desc, val in conditions if passed]
        failed_conditions = [f"{desc} (got {val})" for passed, desc, val in conditions if not passed]
        
        signal = all(passed for passed, desc, val in conditions)
        
        # Calculate profit targets and holding period if signal is valid
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
            'thresholds': thresholds,  # Return thresholds for display
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period
        }
        
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}'}

# =============================
# STREAMLIT INTERFACE
# =============================

# Initialize session state for refresh functionality
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
        # Show help
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
    
    # Auto-refresh settings
    st.subheader("🔄 Auto-Refresh Settings")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    
    if enable_auto_refresh:
        min_interval = 60  # set a sensible floor
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[60, 120, 300],
            index=1,  # Default to 120 seconds
            format_func=lambda x: f"{x} seconds"
        )
        
        # Start/update auto-refresh
        st.session_state.refresh_system.start(refresh_interval)
        st.info(f"Data will refresh every {refresh_interval} seconds")
    else:
        st.session_state.refresh_system.stop()
    
    # Signal thresholds
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
    
    # Common thresholds
    st.write("**Common**")
    SIGNAL_THRESHOLDS['call']['theta_base'] = SIGNAL_THRESHOLDS['put']['theta_base'] = st.slider("Max Theta", 0.01, 0.1, 0.05, 0.01)
    SIGNAL_THRESHOLDS['call']['volume_multiplier_base'] = SIGNAL_THRESHOLDS['put']['volume_multiplier_base'] = st.slider("Volume Multiplier", 1.0, 3.0, 1.0, 0.1)
    
    # Profit targets
    st.subheader("🎯 Profit Targets")
    CONFIG['PROFIT_TARGETS']['call'] = st.slider("Call Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
    CONFIG['PROFIT_TARGETS']['put'] = st.slider("Put Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
    CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01)
    
    # Dynamic threshold parameters
    st.subheader("📈 Dynamic Threshold Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Call Sensitivities**")
        SIGNAL_THRESHOLDS['call']['delta_vol_multiplier'] = st.slider(
            "Delta Vol Sensitivity", 0.0, 0.5, 0.1, 0.01,
            help="How much Delta threshold adjusts to volatility (higher = more sensitive)"
        )
        SIGNAL_THRESHOLDS['call']['gamma_vol_multiplier'] = st.slider(
            "Gamma Vol Sensitivity", 0.0, 0.5, 0.02, 0.01
        )
        
    with col2:
        st.write("**Put Sensitivities**")
        SIGNAL_THRESHOLDS['put']['delta_vol_multiplier'] = st.slider(
            "Delta Vol Sensitivity ", 0.0, 0.5, 0.1, 0.01
        )
        SIGNAL_THRESHOLDS['put']['gamma_vol_multiplier'] = st.slider(
            "Gamma Vol Sensitivity ", 0.0, 0.5, 0.02, 0.01
        )
    
    # Common parameters
    st.write("**Volume Sensitivity**")
    SIGNAL_THRESHOLDS['call']['volume_vol_multiplier'] = SIGNAL_THRESHOLDS['put']['volume_vol_multiplier'] = st.slider(
        "Volume Vol Multiplier", 0.0, 1.0, 0.3, 0.05,
        help="How much volume requirement increases with volatility"
    )

# Main interface
ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper()

# Create refresh status container
refresh_status = st.empty()

# Show refresh status
if enable_auto_refresh:
    # Create a placeholder for dynamic countdown
    countdown_placeholder = refresh_status.empty()
    
    # Get current time
    current_time = time.time()
    elapsed = current_time - st.session_state.last_refresh
    
    # Calculate remaining time
    if 'auto_refresh_interval' in st.session_state:
        remaining = max(0, st.session_state.auto_refresh_interval - elapsed)
        countdown_placeholder.info(f"⏱️ Next refresh in {int(remaining)} seconds")
    else:
        countdown_placeholder.info("🔄 Auto-refresh starting...")
else:
    refresh_status.empty()  # Clear refresh status

if ticker:
    # Create four columns: for market status, current price, last updated, and refresh button
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if is_market_open():
            st.success("✅ Market is OPEN")
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
    
    # Manual refresh logic
    if manual_refresh:
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.rerun()
    
    # Add refresh counter display
    st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Stock Data", "⚙️ Analysis Details"])
    
    with tab1:
        try:
            with st.spinner("Fetching and analyzing data..."):
                # Get stock data
                df = get_stock_data(ticker)
                
                if df.empty:
                    st.error("Unable to fetch stock data. Please check the ticker symbol.")
                    st.stop()
                
                # Compute indicators
                df = compute_indicators(df)
                
                if df.empty:
                    st.error("Unable to compute technical indicators.")
                    st.stop()
                
                # Display current stock info
                current_price = df.iloc[-1]['Close']
                st.success(f"✅ **{ticker}** - Current Price: **${current_price:.2f}**")
                
                # Display volatility info
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
                
                # Diagnostic Information
                st.subheader("🧠 Diagnostic Information")
                
                # Market status
                if is_premarket():
                    st.warning("⚠️ PREMARKET CONDITIONS: Volume requirements relaxed, delta thresholds adjusted")
                elif is_early_market():
                    st.warning("⚠️ EARLY MARKET CONDITIONS: Volume requirements relaxed, delta thresholds adjusted")
                
                # Show current thresholds
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
                
                # Get options expiries
                expiries = get_options_expiries(ticker)
                
                if not expiries:
                    st.error("No options expiries available for this ticker. If you recently refreshed, please wait due to Yahoo Finance rate limits.")
                    st.stop()
                
                # Expiry selection
                expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"], index=1)
                
                today = datetime.date.today()
                if expiry_mode == "0DTE Only":
                    expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
                else:
                    expiries_to_use = expiries[:5]  # Get more expiries for better analysis
                
                if not expiries_to_use:
                    st.warning("No options expiries available for the selected mode.")
                    st.stop()
                
                st.info(f"Analyzing {len(expiries_to_use)} expiries: {', '.join(expiries_to_use)}")
                
                # Fetch options data
                calls, puts = fetch_options_data(ticker, expiries_to_use)
                
                if calls.empty and puts.empty:
                    st.error("No options data available.")
                    st.stop()
                
                # Identify 0DTE options
                for option_df in [calls, puts]:
                    option_df['is_0dte'] = option_df['expiry'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today)
                
                # Strike range filter - narrowed to ±5
                strike_range = st.slider("Strike Range Around Current Price ($):", -50, 50, (-5, 5), 1)
                min_strike = current_price + strike_range[0]
                max_strike = current_price + strike_range[1]
                
                # Filter options by strike
                calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
                puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
                
                # Add moneyness classification
                if not calls_filtered.empty:
                    calls_filtered['moneyness'] = calls_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                if not puts_filtered.empty:
                    puts_filtered['moneyness'] = puts_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                
                # Moneyness filter
                m_filter = st.multiselect("Filter by Moneyness:", options=["ITM", "NTM", "ATM", "OTM"], default=["ITM", "NTM", "ATM"])
                
                if not calls_filtered.empty:
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
                if not puts_filtered.empty:
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]
                
                # Show filtered options count
                st.write(f"🔍 Filtered Options: {len(calls_filtered)} calls, {len(puts_filtered)} puts "
                         f"(Strike range: ${min_strike:.2f}-${max_strike:.2f})")
                
                # Generate signals
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
                            # Sort by signal score
                            signals_df = signals_df.sort_values('signal_score', ascending=False)
                            
                            # Display key columns
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta', 
                                           'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            
                            st.dataframe(
                                signals_df[available_cols].round(4),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Display dynamic thresholds
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
                            
                            # Show passed conditions for first signal
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
                            # Show why the top option didn't qualify
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
                            # Sort by signal score
                            signals_df = signals_df.sort_values('signal_score', ascending=False)
                            
                            # Display key columns
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta', 
                                           'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            
                            st.dataframe(
                                signals_df[available_cols].round(4),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Display dynamic thresholds
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
                            
                            # Show passed conditions for first signal
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
                            # Show why the top option didn't qualify
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
            
            # Display market session info
            if is_premarket():
                st.info("🔔 Currently showing premarket data")
            elif not is_market_open():
                st.info("🔔 Showing after-hours data")
            
            # Display latest values
            latest = df.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Price", f"${latest['Close']:.2f}")
            
            with col2:
                ema_9 = latest['EMA_9']
                if not pd.isna(ema_9):
                    st.metric("EMA 9", f"${ema_9:.2f}")
                else:
                    st.metric("EMA 9", "N/A")
            
            with col3:
                ema_20 = latest['EMA_20']
                if not pd.isna(ema_20):
                    st.metric("EMA 20", f"${ema_20:.2f}")
                else:
                    st.metric("EMA 20", "N/A")
            
            with col4:
                rsi = latest['RSI']
                if not pd.isna(rsi):
                    st.metric("RSI", f"{rsi:.1f}")
                else:
                    st.metric("RSI", "N/A")
            
            with col5:
                atr_pct = latest['ATR_pct']
                if not pd.isna(atr_pct):
                    st.metric("Volatility (ATR%)", f"{atr_pct*100:.2f}%")
                else:
                    st.metric("Volatility", "N/A")
            
            # Display recent data
            st.subheader("Recent Data")
            display_df = df.tail(10)[['Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'ATR_pct', 'Volume', 'avg_vol']].round(2)
            display_df['ATR_pct'] = display_df['ATR_pct'] * 100  # Convert to percentage
            display_df['Volume Ratio'] = display_df['Volume'] / display_df['avg_vol']
            st.dataframe(display_df.rename(columns={
                'ATR_pct': 'ATR%',
                'avg_vol': 'Avg Vol'
            }), use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Analysis Details")
        
        # Auto-refresh status
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

    # Help on rate limits at the bottom for visibility
    with st.expander("ℹ️ About Rate Limiting"):
        st.markdown("""
        Yahoo Finance may restrict how often data can be retrieved. If you see a "rate limited" warning, please:
        - Wait a few minutes before refreshing again
        - Avoid setting auto-refresh intervals lower than 1 minute
        - Use the app with one ticker at a time to reduce load
        """)

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
