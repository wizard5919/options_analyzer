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
    'CACHE_TTL': 60,  # Reduced for more frequent updates
    'RATE_LIMIT_COOLDOWN': 180,
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
        'volume_min': 1000
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
        'volume_min': 1000
    }
}

# =============================
# AUTO-REFRESH SYSTEM
# =============================
def inject_refresh_script(interval):
    """Inject JavaScript for reliable auto-refresh"""
    refresh_script = f"""
    <script>
        setInterval(function() {{
            window.location.reload();
        }}, {interval * 1000});
    </script>
    """
    st.components.v1.html(refresh_script, height=0)

# =============================
# UTILITY FUNCTIONS
# =============================
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
    market_open_today = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'])
    market_open_today = eastern.localize(market_open_today)
    return (now - market_open_today).total_seconds() < 1800

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_current_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        return data['Close'].iloc[-1] if not data.empty else 0.0
    except Exception as e:
        st.error(f"Error fetching price: {str(e)}")
        return 0.0

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "too many requests" in error_msg or "rate limit" in error_msg:
                st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
                return None
            if attempt == max_retries - 1:
                st.error(f"API call failed: {str(e)}")
                return None
            time.sleep(CONFIG['RETRY_DELAY'] * (2 ** attempt))  # Exponential backoff
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
            interval="1m",  # Higher resolution for real-time
            auto_adjust=True,
            progress=False,
            prepost=True
        )
        if data.empty:
            st.warning(f"No data for {ticker}")
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        required_cols = ['Close', 'High', 'Low', 'Volume']
        if not all(col in data.columns for col in required_cols):
            st.error("Missing required columns")
            return pd.DataFrame()
        data = data.dropna(how='all')
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=required_cols)
        if len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient data: {len(data)} points")
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
    if df.empty:
        return df
    df['avg_vol'] = np.nan
    for date, group in df.groupby(df['Datetime'].dt.date):
        regular = group[~group['premarket']]
        if not regular.empty:
            df.loc[regular.index, 'avg_vol'] = regular['Volume'].expanding(min_periods=1).mean()
        premarket = group[group['premarket']]
        if not premarket.empty:
            df.loc[premarket.index, 'avg_vol'] = premarket['Volume'].expanding(min_periods=1).mean()
    df['avg_vol'] = df['avg_vol'].fillna(df['Volume'].mean())
    return df

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.empty:
            return df
        df = df.copy()
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        if df.empty:
            return df
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        if len(close) >= 9:
            df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        else:
            df['EMA_9'] = np.nan
        if len(close) >= 20:
            df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        else:
            df['EMA_20'] = np.nan
        if len(close) >= 14:
            df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        else:
            df['RSI'] = np.nan
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
        return df
    except Exception as e:
        st.error(f"Error computing indicators: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> List[str]:
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        return list(expiries) if expiries else []
    except Exception as e:
        error_msg = str(e).lower()
        if "too many requests" in error_msg or "rate limit" in error_msg:
            st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN']
        else:
            st.error(f"Error fetching expiries: {str(e)}")
        return []

def fetch_options_data(ticker: str, expiries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_calls = pd.DataFrame()
    all_puts = pd.DataFrame()
    stock = yf.Ticker(ticker)
    for expiry in expiries:
        try:
            chain = safe_api_call(stock.option_chain, expiry)
            if chain is None:
                continue
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            required_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
            for df, name in [(calls, 'calls'), (puts, 'puts')]:
                if not all(col in df.columns for col in required_cols):
                    df['delta'] = np.nan
                    df['gamma'] = np.nan
                    df['theta'] = np.nan
                else:
                    for col in ['delta', 'gamma', 'theta']:
                        if col not in df.columns:
                            df[col] = np.nan
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
            time.sleep(0.5)  # Reduced delay
        except Exception as e:
            error_msg = str(e).lower()
            if "too many requests" in error_msg or "rate limit" in error_msg:
                break
            continue
    return all_calls, all_puts

def classify_moneyness(strike: float, spot: float) -> str:
    diff_pct = abs(strike - spot) / spot
    if diff_pct < 0.01:
        return 'ATM'
    elif strike < spot:
        return 'ITM' if diff_pct >= 0.03 else 'NTM'
    else:
        return 'OTM' if diff_pct >= 0.03 else 'NTM'

def calculate_approximate_greeks(option: dict, spot_price: float) -> Tuple[float, float, float]:
    moneyness = spot_price / option['strike']
    if option['contractSymbol'].startswith('C'):
        if moneyness > 1.03:
            delta, gamma = 0.95, 0.01
        elif moneyness > 1.0:
            delta, gamma = 0.65, 0.05
        elif moneyness > 0.97:
            delta, gamma = 0.50, 0.08
        else:
            delta, gamma = 0.35, 0.05
    else:
        if moneyness < 0.97:
            delta, gamma = -0.95, 0.01
        elif moneyness < 1.0:
            delta, gamma = -0.65, 0.05
        elif moneyness < 1.03:
            delta, gamma = -0.50, 0.08
        else:
            delta, gamma = -0.35, 0.05
    theta = 0.05 if "today" in option['expiry'] else 0.02
    return delta, gamma, theta

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    required_fields = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
    if not all(field in option and not pd.isna(option[field]) for field in required_fields):
        return False
    if option['lastPrice'] <= 0:
        return False
    if pd.isna(option.get('delta')) or pd.isna(option.get('gamma')) or pd.isna(option.get('theta')):
        delta, gamma, theta = calculate_approximate_greeks(option, spot_price)
        option['delta'] = delta
        option['gamma'] = gamma
        option['theta'] = theta
    return not (pd.isna(option['delta']) or pd.isna(option['gamma']) or pd.isna(option['theta']))

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
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
        thresholds['delta_min'] = 0.35 if side == 'call' else thresholds.get('delta_min')
        thresholds['delta_max'] = -0.35 if side == 'put' else thresholds.get('delta_max')
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
    expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - datetime.date.today()).days
    if days_to_expiry == 0:
        return "Intraday (Exit before 3:30 PM)"
    intrinsic_value = max(0, spot_price - option['strike']) if option['contractSymbol'].startswith('C') else max(0, option['strike'] - spot_price)
    if intrinsic_value > 0:
        return "1-2 days (Scalp quickly)" if option['theta'] < -0.1 else "3-5 days (Swing trade)"
    return "1 day (Gamma play)" if days_to_expiry <= 3 else "3-7 days (Wait for move)"

def calculate_profit_targets(option: pd.Series) -> Tuple[float, float]:
    entry_price = option['lastPrice']
    side = 'call' if option['contractSymbol'].startswith('C') else 'put'
    profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS'][side])
    stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
    return profit_target, stop_loss

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    try:
        if stock_df.empty:
            return {'signal': False, 'reason': 'No stock data'}
        current_price = stock_df.iloc[-1]['Close']
        if not validate_option_data(option, current_price):
            return {'signal': False, 'reason': 'Invalid option data'}
        latest = stock_df.iloc[-1]
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
        if side == "call":
            conditions = [
                (delta >= thresholds['delta_min'], f"Delta ≥ {thresholds['delta_min']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma ≥ {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta ≤ {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']:.1f}", rsi),
                (vwap is not None and close > vwap, "Price > VWAP", f"{close:.2f} > {vwap:.2f}" if vwap else "N/A"),
                (option_volume > thresholds['volume_min'], f"Option Vol > {thresholds['volume_min']}", option_volume)
            ]
        else:
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta ≤ {thresholds['delta_max']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma ≥ {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta ≤ {thresholds['theta_base']:.3f}", theta),
                (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']:.1f}", rsi),
                (vwap is not None and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (option_volume > thresholds['volume_min'], f"Option Vol > {thresholds['volume_min']}", option_volume)
            ]
        passed_conditions = [desc for passed, desc, _ in conditions if passed]
        failed_conditions = [f"{desc} (got {val})" for passed, desc, val in conditions if not passed]
        signal = all(passed for passed, _, _ in conditions)
        profit_target, stop_loss, holding_period = None, None, None
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
        return {'signal': False, 'reason': f'Signal generation error: {str(e)}'}

# =============================
# STREAMLIT INTERFACE
# =============================
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Rate limit check
if 'rate_limited_until' in st.session_state and time.time() < st.session_state['rate_limited_until']:
    remaining = int(st.session_state['rate_limited_until'] - time.time())
    st.warning(f"API rate limited. Wait {remaining} seconds.")
    st.stop()

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .call-section { background-color: #e6ffe6; padding: 10px; border-radius: 5px; }
    .put-section { background-color: #ffe6e6; padding: 10px; border-radius: 5px; }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Stock Ticker (e.g., SPY, AAPL):", value="SPY").upper()
    
    # Auto-refresh
    st.subheader("🔄 Auto-Refresh")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    if enable_auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[30, 60, 120],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
        inject_refresh_script(refresh_interval)
        st.session_state['auto_refresh_interval'] = refresh_interval
        st.info(f"Refreshing every {refresh_interval} seconds")
    
    # Thresholds
    st.subheader("📏 Auto-Adjusted Signal Thresholds")
    try:
        if ticker:
            df = get_stock_data(ticker)
            if not df.empty:
                latest = df.iloc[-1]
                call_th = calculate_dynamic_thresholds(latest, 'call', is_0dte=False)
                put_th = calculate_dynamic_thresholds(latest, 'put', is_0dte=False)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='call-section'>", unsafe_allow_html=True)
                    st.write("**Calls**")
                    st.caption(f"Δ ≥ {call_th['delta_min']:.2f} | Γ ≥ {call_th['gamma_min']:.3f} | Θ ≤ {call_th['theta_base']:.3f}")
                    st.caption(f"RSI > {call_th['rsi_min']:.1f} | Vol > {call_th['volume_min']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='put-section'>", unsafe_allow_html=True)
                    st.write("**Puts**")
                    st.caption(f"Δ ≤ {put_th['delta_max']:.2f} | Γ ≥ {put_th['gamma_min']:.3f} | Θ ≤ {put_th['theta_base']:.3f}")
                    st.caption(f"RSI < {put_th['rsi_max']:.1f} | Vol > {put_th['volume_min']}")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Enter a valid ticker to see thresholds")
    except Exception as e:
        st.error(f"Error calculating thresholds: {str(e)}")
    
    # Manual threshold adjustments
    st.subheader("Base Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='call-section'>", unsafe_allow_html=True)
        SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider("Call Delta", 0.1, 1.0, 0.5, 0.1)
        SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider("Call Gamma", 0.01, 0.2, 0.05, 0.01)
        SIGNAL_THRESHOLDS['call']['rsi_min'] = st.slider("Call RSI Min", 30, 70, 50, 5)
        SIGNAL_THRESHOLDS['call']['volume_min'] = st.slider("Call Volume Min", 100, 5000, 1000, 100)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='put-section'>", unsafe_allow_html=True)
        SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider("Put Delta", -1.0, -0.1, -0.5, 0.1)
        SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider("Put Gamma", 0.01, 0.2, 0.05, 0.01)
        SIGNAL_THRESHOLDS['put']['rsi_max'] = st.slider("Put RSI Max", 30, 70, 50, 5)
        SIGNAL_THRESHOLDS['put']['volume_min'] = st.slider("Put Volume Min", 100, 5000, 1000, 100)
        st.markdown("</div>", unsafe_allow_html=True)
    SIGNAL_THRESHOLDS['call']['theta_base'] = SIGNAL_THRESHOLDS['put']['theta_base'] = st.slider("Max Theta", 0.01, 0.1, 0.05, 0.01)

# Main interface
if ticker:
    try:
        with st.spinner("Fetching data..."):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status = "OPEN" if is_market_open() else "PREMARKET" if is_premarket() else "CLOSED"
                st.metric("Market Status", status)
            with col2:
                current_price = get_current_price(ticker)
                st.metric("Current Price", f"${current_price:.2f}")
            with col3:
                st.metric("Last Updated", datetime.datetime.now().strftime("%H:%M:%S"))
            with col4:
                if st.button("🔁 Refresh Now"):
                    st.cache_data.clear()
                    st.session_state.last_refresh = time.time()
                    st.session_state.refresh_counter += 1
                    st.rerun()
            
            tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Stock Data", "⚙️ Analysis Details"])
            
            with tab1:
                df = get_stock_data(ticker)
                if df.empty:
                    st.error("Unable to fetch stock data.")
                    st.stop()
                df = compute_indicators(df)
                if df.empty:
                    st.error("Unable to compute indicators.")
                    st.stop()
                st.success(f"**{ticker}** - Price: **${df.iloc[-1]['Close']:.2f}**")
                atr_pct = df.iloc[-1].get('ATR_pct', 0)
                vol_status = "Low" if atr_pct <= CONFIG['VOLATILITY_THRESHOLDS']['low'] else "Medium" if atr_pct <= CONFIG['VOLATILITY_THRESHOLDS']['medium'] else "High" if atr_pct <= CONFIG['VOLATILITY_THRESHOLDS']['high'] else "Extreme"
                st.info(f"Volatility: {atr_pct*100:.2f}% - **{vol_status}**")
                
                expiries = get_options_expiries(ticker)
                if not expiries:
                    st.error("No options expiries available.")
                    st.stop()
                
                expiry_mode = st.radio("Expiration Filter:", ["0DTE Only", "All Near-Term"], index=1)
                today = datetime.date.today()
                expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today] if expiry_mode == "0DTE Only" else expiries[:5]
                if not expiries_to_use:
                    st.warning("No expiries for selected mode.")
                    st.stop()
                
                calls, puts = fetch_options_data(ticker, expiries_to_use)
                if calls.empty and puts.empty:
                    st.error("No options data.")
                    st.stop()
                
                for option_df in [calls, puts]:
                    option_df['is_0dte'] = option_df['expiry'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today)
                
                strike_range = st.slider("Strike Range ($):", -50, 50, (-5, 5), 1)
                min_strike, max_strike = current_price + strike_range[0], current_price + strike_range[1]
                calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
                puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
                
                if not calls_filtered.empty:
                    calls_filtered['moneyness'] = calls_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                if not puts_filtered.empty:
                    puts_filtered['moneyness'] = puts_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
                
                m_filter = st.multiselect("Moneyness:", ["ITM", "NTM", "ATM", "OTM"], default=["ITM", "NTM", "ATM"])
                calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)] if not calls_filtered.empty else calls_filtered
                puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)] if not puts_filtered.empty else puts_filtered
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='call-section'>", unsafe_allow_html=True)
                    st.subheader("📈 Call Signals")
                    if not calls_filtered.empty:
                        call_signals = []
                        for _, row in calls_filtered.iterrows():
                            signal_result = generate_signal(row, "call", df, row.get('is_0dte', False))
                            if signal_result['signal']:
                                row_dict = row.to_dict()
                                row_dict.update({
                                    'signal_score': signal_result['score'],
                                    'thresholds': signal_result['thresholds'],
                                    'passed_conditions': signal_result['passed_conditions'],
                                    'is_0dte': row.get('is_0dte', False),
                                    'profit_target': signal_result['profit_target'],
                                    'stop_loss': signal_result['stop_loss'],
                                    'holding_period': signal_result['holding_period']
                                })
                                call_signals.append(row_dict)
                        if call_signals:
                            signals_df = pd.DataFrame(call_signals).sort_values('signal_score', ascending=False)
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta', 'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte']
                            st.dataframe(signals_df[display_cols].round(4), use_container_width=True, hide_index=True)
                            th = signals_df.iloc[0]['thresholds']
                            st.info(f"Thresholds: Δ ≥ {th['delta_min']:.2f} | Γ ≥ {th['gamma_min']:.3f} | Θ ≤ {th['theta_base']:.3f} | RSI > {th['rsi_min']:.1f} | Vol > {th['volume_min']}")
                            with st.expander("Top Signal Conditions"):
                                for condition in signals_df.iloc[0]['passed_conditions']:
                                    st.write(f"- {condition}")
                            st.success(f"{len(call_signals)} call signals found!")
                        else:
                            st.info("No call signals.")
                            if not calls_filtered.empty:
                                result = generate_signal(calls_filtered.iloc[0], "call", df, calls_filtered.iloc[0].get('is_0dte', False))
                                if 'failed_conditions' in result:
                                    st.write("Top call failed:")
                                    for condition in result['failed_conditions']:
                                        st.write(f"- {condition}")
                    else:
                        st.info("No calls available.")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='put-section'>", unsafe_allow_html=True)
                    st.subheader("📉 Put Signals")
                    if not puts_filtered.empty:
                        put_signals = []
                        for _, row in puts_filtered.iterrows():
                            signal_result = generate_signal(row, "put", df, row.get('is_0dte', False))
                            if signal_result['signal']:
                                row_dict = row.to_dict()
                                row_dict.update({
                                    'signal_score': signal_result['score'],
                                    'thresholds': signal_result['thresholds'],
                                    'passed_conditions': signal_result['passed_conditions'],
                                    'is_0dte': row.get('is_0dte', False),
                                    'profit_target': signal_result['profit_target'],
                                    'stop_loss': signal_result['stop_loss'],
                                    'holding_period': signal_result['holding_period']
                                })
                                put_signals.append(row_dict)
                        if put_signals:
                            signals_df = pd.DataFrame(put_signals).sort_values('signal_score', ascending=False)
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta', 'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte']
                            st.dataframe(signals_df[display_cols].round(4), use_container_width=True, hide_index=True)
                            th = signals_df.iloc[0]['thresholds']
                            st.info(f"Thresholds: Δ ≤ {th['delta_max']:.2f} | Γ ≥ {th['gamma_min']:.3f} | Θ ≤ {th['theta_base']:.3f} | RSI < {th['rsi_max']:.1f} | Vol > {th['volume_min']}")
                            with st.expander("Top Signal Conditions"):
                                for condition in signals_df.iloc[0]['passed_conditions']:
                                    st.write(f"- {condition}")
                            st.success(f"{len(put_signals)} put signals found!")
                        else:
                            st.info("No put signals.")
                            if not puts_filtered.empty:
                                result = generate_signal(puts_filtered.iloc[0], "put", df, puts_filtered.iloc[0].get('is_0dte', False))
                                if 'failed_conditions' in result:
                                    st.write("Top put failed:")
                                    for condition in result['failed_conditions']:
                                        st.write(f"- {condition}")
                    else:
                        st.info("No puts available.")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with tab2:
                if not df.empty:
                    st.subheader("📊 Stock Data")
                    latest = df.iloc[-1]
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Price", f"${latest['Close']:.2f}")
                    with col2:
                        st.metric("EMA 9", f"${latest['EMA_9']:.2f}" if not pd.isna(latest['EMA_9']) else "N/A")
                    with col3:
                        st.metric("EMA 20", f"${latest['EMA_20']:.2f}" if not pd.isna(latest['EMA_20']) else "N/A")
                    with col4:
                        st.metric("RSI", f"{latest['RSI']:.1f}" if not pd.isna(latest['RSI']) else "N/A")
                    with col5:
                        st.metric("ATR%", f"{latest['ATR_pct']*100:.2f}%" if not pd.isna(latest['ATR_pct']) else "N/A")
                    st.dataframe(df.tail(10)[['Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'ATR_pct', 'Volume']].round(2), use_container_width=True)
            
            with tab3:
                st.subheader("🔍 Analysis Details")
                st.json(SIGNAL_THRESHOLDS)
                st.json(CONFIG)
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Enter a ticker to begin.")
