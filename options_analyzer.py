import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import pytz
import threading
import logging
from typing import Optional, Tuple, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, KeltnerChannel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    'CACHE_TTL': 60,  # Reduced for fresher data
    'RATE_LIMIT_COOLDOWN': 300,  # Increased to avoid frequent rate limits
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
    'TIME_DECAY': {
        'theta_scalar': 0.0001,  # Base theta decay per second for 0DTE
        'hours_to_close': 6.5    # Hours from open to close
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
        'volume_min': 1000,
        'macd_above_signal': True,
        'price_above_keltner': True,
        'ema_50_above_200': True
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
        'macd_below_signal': True,
        'price_below_keltner': True,
        'ema_50_below_200': True
    }
}

# =============================
# AUTO-REFRESH SYSTEM
# =============================

class AutoRefreshSystem:
    def __init__(self):
        self.running = False
        self.thread = None
        self.refresh_interval = 1  # Default to 1 second

    def start(self, interval):
        if self.running and interval == self.refresh_interval:
            return
        self.stop()
        self.running = True
        self.refresh_interval = max(1, interval)

        def refresh_loop():
            while self.running:
                start_time = time.time()
                if 'rate_limited_until' in st.session_state and time.time() < st.session_state['rate_limited_until']:
                    logger.warning(f"Skipping refresh due to rate limit. Cooldown until {datetime.datetime.fromtimestamp(st.session_state['rate_limited_until'])}")
                    time.sleep(1)
                    continue
                try:
                    with st.spinner("Refreshing data..."):
                        st.rerun()
                except Exception as e:
                    logger.error(f"Refresh error: {str(e)}")
                elapsed = time.time() - start_time
                time.sleep(max(0, self.refresh_interval - elapsed))

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
    market_open_today = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN']).replace(tzinfo=eastern)
    return (now - market_open_today).total_seconds() < 1800

def get_current_price(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if data.empty:
            logger.warning(f"No price data for {ticker}")
            return 0.0
        return float(data['Close'].iloc[-1])
    except Exception as e:
        logger.error(f"Price error for {ticker}: {str(e)}")
        st.error(f"Error getting price for {ticker}: {str(e)}")
        return 0.0

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "too many requests" in error_msg or "rate limit" in error_msg:
                st.session_state['rate_limited_until'] = time.time() + CONFIG['RATE_LIMIT_COOLDOWN'] * (2 ** attempt)
                logger.warning(f"Rate limit hit, cooling down for {CONFIG['RATE_LIMIT_COOLDOWN'] * (2 ** attempt)} seconds")
                st.warning(f"Yahoo Finance rate limit reached. Wait {CONFIG['RATE_LIMIT_COOLDOWN'] * (2 ** attempt)} seconds.")
                return None
            if attempt == max_retries - 1:
                logger.error(f"API call failed after {max_retries} attempts: {str(e)}")
                st.error(f"API call failed: {str(e)}")
                return None
            time.sleep(CONFIG['RETRY_DELAY'] * (2 ** attempt))
    return None

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=10)
        data = yf.download(ticker, start=start, end=end, interval="5m", auto_adjust=True, progress=False, prepost=True)
        if data.empty:
            logger.warning(f"No data for {ticker}")
            st.warning(f"No data for {ticker}")
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns for {ticker}: {missing_cols}")
            st.error(f"Missing columns: {missing_cols}")
            return pd.DataFrame()
        data = data.dropna(how='all')
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=required_cols)
        if len(data) < CONFIG['MIN_DATA_POINTS']:
            logger.warning(f"Insufficient data points for {ticker}: {len(data)}")
            st.warning(f"Insufficient data ({len(data)}). Need {CONFIG['MIN_DATA_POINTS']}.")
            return pd.DataFrame()
        eastern = pytz.timezone('US/Eastern')
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(eastern)
        data['premarket'] = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        return data.reset_index(drop=False)
    except Exception as e:
        logger.error(f"Stock data error for {ticker}: {str(e)}")
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
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
    except Exception as e:
        logger.error(f"Volume averages error: {str(e)}")
        return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        df = df.copy()
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            st.error(f"Missing columns: {missing_cols}")
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
            df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        if len(close) >= 20:
            df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        if len(close) >= 50:
            df['EMA_50'] = EMAIndicator(close=close, window=50).ema_indicator()
        if len(close) >= 200:
            df['EMA_200'] = EMAIndicator(close=close, window=200).ema_indicator()
        if len(close) >= 14:
            df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        if len(close) >= 26:
            macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
        if len(close) >= 20:
            keltner = KeltnerChannel(high=high, low=low, close=close, window=20, window_atr=10)
            df['Keltner_Upper'] = keltner.keltner_channel_hband()
            df['Keltner_Middle'] = keltner.keltner_channel_mband()
            df['Keltner_Lower'] = keltner.keltner_channel_lband()
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
                df.loc[regular.index, 'VWAP'] = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
            if not premarket.empty:
                prev_day = session - datetime.timedelta(days=1)
                prev_close = df[df['Datetime'].dt.date == prev_day.date()]['Close'].iloc[-1] if not df[df['Datetime'].dt.date == prev_day.date()].empty else premarket['Close'].iloc[0]
                typical_price = (premarket['High'] + premarket['Low'] + premarket['Close']) / 3
                vwap_cumsum = (premarket['Volume'] * typical_price).cumsum()
                volume_cumsum = premarket['Volume'].cumsum()
                df.loc[premarket.index, 'VWAP'] = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
        if len(close) >= 14:
            df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
            df['ATR_pct'] = df['ATR'] / close
        df = calculate_volume_averages(df)
        return df
    except Exception as e:
        logger.error(f"Indicators error: {str(e)}")
        st.error(f"Error computing indicators: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> List[str]:
    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options  # Direct access instead of safe_api_call
        if not expiries:
            logger.warning(f"No options expiries for {ticker}. Possible rate limit or non-optionable ticker.")
            st.warning(f"No options expiries for {ticker}. Check ticker or wait if recently refreshed.")
            return []
        return list(expiries)
    except Exception as e:
        logger.error(f"Expiries error for {ticker}: {str(e)}")
        st.error(f"Error fetching expiries for {ticker}: {str(e)}")
        return []

def fetch_options_data(ticker: str, expiries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_calls = pd.DataFrame()
    all_puts = pd.DataFrame()
    stock = yf.Ticker(ticker)
    failed_expiries = []
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
            for df, name in [(calls, 'calls'), (puts, 'puts')]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {name} for {expiry}: {missing_cols}")
                    continue
                for col in ['delta', 'gamma', 'theta']:
                    if col not in df.columns:
                        df[col] = np.nan
                all_calls = pd.concat([all_calls, calls], ignore_index=True)
                all_puts = pd.concat([all_puts, puts], ignore_index=True)
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Options fetch error for {expiry}: {str(e)}")
            failed_expiries.append(expiry)
            continue
    if failed_expiries:
        st.info(f"Failed to fetch data for expiries: {', '.join(failed_expiries)}")
    return all_calls, all_puts

def classify_moneyness(strike: float, spot: float) -> str:
    diff_pct = abs(strike - spot) / spot
    if diff_pct < 0.01:
        return 'ATM'
    elif strike < spot:
        return 'ITM' if diff_pct > 0.03 else 'NTM'
    else:
        return 'OTM' if diff_pct > 0.03 else 'NTM'

def calculate_time_decay(option: pd.Series, is_0dte: bool) -> float:
    if not is_0dte:
        return 0.0
    try:
        theta = float(option.get('theta', CONFIG['TIME_DECAY']['theta_scalar']))
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        market_close = datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE']).replace(tzinfo=eastern)
        seconds_remaining = max(0, (market_close - now).total_seconds())
        decay = theta * seconds_remaining * CONFIG['TIME_DECAY']['theta_scalar']
        return decay
    except Exception as e:
        logger.error(f"Time decay error: {str(e)}")
        return 0.0

def calculate_approximate_greeks(option: dict, spot_price: float) -> Tuple[float, float, float]:
    moneyness = spot_price / option['strike']
    if option['contractSymbol'].startswith('C'):
        delta = 0.95 if moneyness > 1.03 else 0.65 if moneyness > 1.0 else 0.50 if moneyness > 0.97 else 0.35
        gamma = 0.01 if moneyness > 1.03 else 0.05 if moneyness > 1.0 else 0.08 if moneyness > 0.97 else 0.05
    else:
        delta = -0.95 if moneyness < 0.97 else -0.65 if moneyness < 1.0 else -0.50 if moneyness < 1.03 else -0.35
        gamma = 0.01 if moneyness < 0.97 else 0.05 if moneyness < 1.0 else 0.08 if moneyness < 1.03 else 0.05
    theta = 0.05 if "today" in option['expiry'] else 0.02
    return delta, gamma, theta

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    required_fields = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
    try:
        for field in required_fields:
            if field not in option or pd.isna(option[field]) or not isinstance(option[field], (int, float)):
                return False
        if float(option['lastPrice']) <= 0 or float(option['volume']) < 0:
            return False
        if pd.isna(option.get('delta')) or pd.isna(option.get('gamma')) or pd.isna(option.get('theta')):
            delta, gamma, theta = calculate_approximate_greeks(option, spot_price)
            option['delta'] = delta
            option['gamma'] = gamma
            option['theta'] = theta
        return all(not pd.isna(option[col]) for col in ['delta', 'gamma', 'theta'])
    except Exception as e:
        logger.error(f"Option validation error: {str(e)}")
        return False

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    try:
        volatility = float(stock_data.get('ATR_pct', 0.02))
        vol_multiplier = 1 + (volatility * 100)
        if side == 'call':
            thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_multiplier))
        else:
            thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_multiplier))
        thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + thresholds['gamma_vol_multiplier'] * (volatility * 200))
        thresholds['volume_multiplier'] = max(0.8, min(2.5, thresholds['volume_multiplier_base'] * (1 + thresholds['volume_vol_multiplier'] * (volatility * 150))))
        if is_premarket() or is_early_market():
            thresholds['delta_min' if side == 'call' else 'delta_max'] = 0.35 if side == 'call' else -0.35
            thresholds['volume_multiplier'] *= 0.6
            thresholds['gamma_min'] *= 0.8
        if is_0dte:
            thresholds['volume_multiplier'] *= 0.7
            thresholds['gamma_min'] *= 0.7
            thresholds['delta_min' if side == 'call' else 'delta_max'] = max(0.4, thresholds['delta_min']) if side == 'call' else min(-0.4, thresholds['delta_max'])
        return thresholds
    except Exception as e:
        logger.error(f"Dynamic thresholds error: {str(e)}")
        return thresholds

def calculate_holding_period(option: pd.Series, spot_price: float) -> str:
    try:
        expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
        days_to_expiry = (expiry_date - datetime.date.today()).days
        if days_to_expiry == 0:
            return "Intraday (Exit before 3:30 PM)"
        intrinsic_value = max(0, spot_price - option['strike']) if option['contractSymbol'].startswith('C') else max(0, option['strike'] - spot_price)
        if intrinsic_value > 0:
            return "1-2 days (Scalp quickly)" if float(option['theta']) < -0.1 else "3-5 days (Swing trade)"
        return "1 day (Gamma play)" if days_to_expiry <= 3 else "3-7 days (Wait for move)"
    except Exception as e:
        logger.error(f"Holding period error: {str(e)}")
        return "N/A"

def calculate_profit_targets(option: pd.Series) -> Tuple[float, float]:
    try:
        entry_price = float(option['lastPrice'])
        profit_target = entry_price * (1 + CONFIG['PROFIT_TARGETS']['call' if option['contractSymbol'].startswith('C') else 'put'])
        stop_loss = entry_price * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
        return profit_target, stop_loss
    except Exception as e:
        logger.error(f"Profit targets error: {str(e)}")
        return 0.0, 0.0

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data'}
    try:
        current_price = float(stock_df.iloc[-1]['Close'])
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
        ema_50 = float(latest['EMA_50']) if not pd.isna(latest['EMA_50']) else None
        ema_200 = float(latest['EMA_200']) if not pd.isna(latest['EMA_200']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
        macd_signal = float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else None
        keltner_upper = float(latest['Keltner_Upper']) if not pd.isna(latest['Keltner_Upper']) else None
        keltner_lower = float(latest['Keltner_Lower']) if not pd.isna(latest['Keltner_Lower']) else None
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume
        conditions = []
        if side == "call":
            volume_ok = option_volume > thresholds['volume_min']
            conditions = [
                (delta >= thresholds['delta_min'], f"Delta >= {thresholds['delta_min']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 and ema_20 and close > ema_9 > ema_20, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (ema_50 and ema_200 and ema_50 > ema_200, "EMA50 > EMA200", f"{ema_50:.2f} > {ema_200:.2f}" if ema_50 and ema_200 else "N/A"),
                (rsi and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']:.1f}", rsi),
                (vwap and close > vwap, "Price > VWAP", f"{close:.2f} > {vwap:.2f}" if vwap else "N/A"),
                (macd and macd_signal and macd > macd_signal, "MACD > Signal", f"{macd:.2f} > {macd_signal:.2f}" if macd and macd_signal else "N/A"),
                (keltner_upper and close > keltner_upper, "Price > Keltner Upper", f"{close:.2f} > {keltner_upper:.2f}" if keltner_upper else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f}")
            ]
        else:
            volume_ok = option_volume > thresholds['volume_min']
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']:.2f}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']:.3f}", gamma),
                (theta <= thresholds['theta_base'], f"Theta <= {thresholds['theta_base']:.3f}", theta),
                (ema_9 and ema_20 and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (ema_50 and ema_200 and ema_50 < ema_200, "EMA50 < EMA200", f"{ema_50:.2f} < {ema_200:.2f}" if ema_50 and ema_200 else "N/A"),
                (rsi and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']:.1f}", rsi),
                (vwap and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (macd and macd_signal and macd < macd_signal, "MACD < Signal", f"{macd:.2f} < {macd_signal:.2f}" if macd and macd_signal else "N/A"),
                (keltner_lower and close < keltner_lower, "Price < Keltner Lower", f"{close:.2f} < {keltner_lower:.2f}" if keltner_lower else "N/A"),
                (volume_ok, f"Option Vol > {thresholds['volume_min']}", f"{option_volume:.0f}")
            ]
        passed_conditions = [desc for passed, desc, _ in conditions]
        failed_conditions = [f"{desc} (got {val})" for passed, desc, val in conditions if not passed]
        signal = all(passed for passed, _, _ in conditions)
        score = len(passed_conditions) / len(conditions)
        if is_0dte:
            time_decay = calculate_time_decay(option, is_0dte)
            score *= max(0.5, 1 - (time_decay / option['lastPrice'])) if option['lastPrice'] > 0 else 0.5
        else:
            time_decay = None
        profit_target, stop_loss = calculate_profit_targets(option) if signal else (None, None)
        holding_period = calculate_holding_period(option, current_price) if signal else None
        return {
            'signal': signal,
            'passed_conditions': passed_conditions,
            'failed_conditions': failed_conditions,
            'score': score,
            'thresholds': thresholds,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period,
            'time_decay': time_decay
        }
    except Exception as e:
        logger.error(f"Signal generation error: {str(e)}")
        return {'signal': False, 'reason': f'Signal error: {str(e)}'}

def calculate_scanner_score(stock_df: pd.DataFrame, side: str) -> float:
    if stock_df.empty:
        return 0.0
    try:
        latest = stock_df.iloc[-1]
        score = 0.0
        max_score = 5.0
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        ema_50 = float(latest['EMA_50']) if not pd.isna(latest['EMA_50']) else None
        ema_200 = float(latest['EMA_200']) if not pd.isna(latest['EMA_200']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
        macd_signal = float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else None
        keltner_upper = float(latest['Keltner_Upper']) if not pd.isna(latest['Keltner_Upper']) else None
        keltner_lower = float(latest['Keltner_Lower']) if not pd.isna(latest['Keltner_Lower']) else None
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
        logger.error(f"Scanner score error: {str(e)}")
        st.error(f"Scanner score error: {str(e)}")
        return 0.0

# =============================
# STREAMLIT INTERFACE
# =============================

if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'refresh_system' not in st.session_state:
    st.session_state.refresh_system = AutoRefreshSystem()

if 'rate_limited_until' in st.session_state and time.time() < st.session_state['rate_limited_until']:
    remaining = int(st.session_state['rate_limited_until'] - time.time())
    st.warning(f"Yahoo Finance API rate limited. Wait {remaining} seconds.")
    with st.expander("ℹ️ About Rate Limiting"):
        st.markdown("""
        Yahoo Finance may restrict data retrieval. If rate limited:
        - Wait a few minutes before retrying
        - Avoid auto-refresh intervals below 1 minute
        - Use one ticker at a time
        """)
    st.stop()

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Enhanced for volatile markets** with real-time updates")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🔄 Auto-Refresh Settings")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    if enable_auto_refresh:
        refresh_interval = st.selectbox("Refresh Interval", options=[1, 5, 10, 30, 60], index=0, format_func=lambda x: f"{x} second{'s' if x != 1 else ''}")
        st.session_state.refresh_system.start(refresh_interval)
        st.info(f"Data will refresh every {refresh_interval} second{'s' if refresh_interval != 1 else ''}")
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
    st.write("**Common**")
    SIGNAL_THRESHOLDS['call']['theta_base'] = SIGNAL_THRESHOLDS['put']['theta_base'] = st.slider("Max Theta", 0.01, 0.1, 0.05, 0.01)
    SIGNAL_THRESHOLDS['call']['volume_multiplier_base'] = SIGNAL_THRESHOLDS['put']['volume_multiplier_base'] = st.slider("Volume Multiplier", 1.0, 3.0, 1.0, 0.1)
    st.subheader("🎯 Profit Targets")
    CONFIG['PROFIT_TARGETS']['call'] = st.slider("Call Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
    CONFIG['PROFIT_TARGETS']['put'] = st.slider("Put Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
    CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01)

ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper()

refresh_status = st.empty()
if enable_auto_refresh:
    elapsed = time.time() - st.session_state.last_refresh
    remaining = max(0, refresh_interval - elapsed)
    refresh_status.info(f"⏱️ Next refresh in {int(remaining)} second{'s' if int(remaining) != 1 else ''}")
else:
    refresh_status.empty()

if ticker:
    try:
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
            if st.button("🔁 Refresh Now", key="manual_refresh"):
                st.cache_data.clear()
                st.session_state.last_refresh = time.time()
                st.session_state.refresh_counter += 1
                st.rerun()
        st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")

        st.subheader("📊 Call/Put Scanner")
        df = get_stock_data(ticker)
        if not df.empty:
            df = compute_indicators(df)
            call_score = calculate_scanner_score(df, "call")
            put_score = calculate_scanner_score(df, "put")
            col1, col2 = st.columns(2)
            with col1:
                st.progress(min(call_score / 100, 1.0), text=f"Call Signal Strength: {call_score:.1f}%")
            with col2:
                st.progress(min(put_score / 100, 1.0), text=f"Put Signal Strength: {put_score:.1f}%")
            if call_score > 80:
                st.success("🚀 Strong Call Opportunity Detected!")
            elif put_score > 80:
                st.success("📉 Strong Put Opportunity Detected!")
            elif call_score > 60 or put_score > 60:
                st.info("⚠️ Moderate Opportunity Detected")
            else:
                st.info("🛑 No Strong Opportunities")

        tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Stock Data", "⚙️ Analysis Details"])
        with tab1:
            with st.spinner("Fetching and analyzing data..."):
                if df.empty:
                    st.error(f"No stock data for {ticker}. Check ticker or try again.")
                    st.stop()
                df = compute_indicators(df)
                if df.empty:
                    st.error("Cannot compute indicators.")
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
                    st.info(f"📈 Volatility (ATR%): {atr_pct*100:.2f}% - **{volatility_status}**")
                st.subheader("🧠 Diagnostic Information")
                if is_premarket():
                    st.warning("⚠️ PREMARKET: Relaxed thresholds")
                elif is_early_market():
                    st.warning("⚠️ EARLY MARKET: Relaxed thresholds")
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
                    st.error(f"No options expiries for {ticker}. Possible causes: rate limit, non-optionable ticker, or API issue. Try another ticker or wait.")
                    st.stop()
                expiry_mode = st.radio("Expiration Filter:", ["0DTE Only", "All Near-Term"], index=1)
                today = datetime.date.today()
                expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today] if expiry_mode == "0DTE Only" else expiries[:5]
                if not expiries_to_use:
                    st.warning("No expiries for selected mode.")
                    st.stop()
                st.info(f"Analyzing {len(expiries_to_use)} expiries: {', '.join(expiries_to_use)}")
                calls, puts = fetch_options_data(ticker, expiries_to_use)
                if calls.empty and puts.empty:
                    st.error("No options data.")
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
                m_filter = st.multiselect("Filter by Moneyness:", ["ITM", "NTM", "ATM", "OTM"], default=["ITM", "NTM", "ATM"])
                if not calls_filtered.empty:
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
                if not puts_filtered.empty:
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]
                st.write(f"🔍 Filtered Options: {len(calls_filtered)} calls, {len(puts_filtered)} puts "
                         f"(Strike range: ${min_strike:.2f}-${max_strike:.2f})")
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
                                row_dict.update({
                                    'signal_score': signal_result['score'],
                                    'thresholds': signal_result['thresholds'],
                                    'passed_conditions': signal_result['passed_conditions'],
                                    'is_0dte': is_0dte,
                                    'profit_target': signal_result['profit_target'],
                                    'stop_loss': signal_result['stop_loss'],
                                    'holding_period': signal_result['holding_period'],
                                    'time_decay': signal_result['time_decay']
                                })
                                call_signals.append(row_dict)
                        if call_signals:
                            signals_df = pd.DataFrame(call_signals).sort_values('signal_score', ascending=False)
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta',
                                           'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte', 'time_decay']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            st.dataframe(signals_df[available_cols].round(4), use_container_width=True, hide_index=True)
                            if signals_df.iloc[0]['thresholds']:
                                th = signals_df.iloc[0]['thresholds']
                                st.info(f"Thresholds: Δ ≥ {th['delta_min']:.2f} | Γ ≥ {th['gamma_min']:.3f} | "
                                        f"Θ ≤ {th['theta_base']:.3f} | RSI > {th['rsi_min']:.1f} | Vol > {th['volume_min']}")
                            with st.expander("View Conditions for Top Signal"):
                                if signals_df.iloc[0]['passed_conditions']:
                                    st.write("✅ Passed Conditions:")
                                    for condition in signals_df.iloc[0]['passed_conditions']:
                                        st.write(f"- {condition}")
                                else:
                                    st.info("No conditions passed")
                            st.success(f"Found {len(call_signals)} call signals!")
                        else:
                            st.info("No call signals found.")
                            if not calls_filtered.empty:
                                sample_call = calls_filtered.iloc[0]
                                is_0dte = sample_call.get('is_0dte', False)
                                result = generate_signal(sample_call, "call", df, is_0dte)
                                if 'failed_conditions' in result:
                                    st.write("Top call option failed conditions:")
                                    for condition in result['failed_conditions']:
                                        st.write(f"- {condition}")
                    else:
                        st.info("No call options for selected filters.")
                with col2:
                    st.subheader("📉 Put Option Signals")
                    if not puts_filtered.empty:
                        put_signals = []
                        for _, row in puts_filtered.iterrows():
                            is_0dte = row.get('is_0dte', False)
                            signal_result = generate_signal(row, "put", df, is_0dte)
                            if signal_result['signal']:
                                row_dict = row.to_dict()
                                row_dict.update({
                                    'signal_score': signal_result['score'],
                                    'thresholds': signal_result['thresholds'],
                                    'passed_conditions': signal_result['passed_conditions'],
                                    'is_0dte': is_0dte,
                                    'profit_target': signal_result['profit_target'],
                                    'stop_loss': signal_result['stop_loss'],
                                    'holding_period': signal_result['holding_period'],
                                    'time_decay': signal_result['time_decay']
                                })
                                put_signals.append(row_dict)
                        if put_signals:
                            signals_df = pd.DataFrame(put_signals).sort_values('signal_score', ascending=False)
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'volume', 'delta', 'gamma', 'theta',
                                           'moneyness', 'signal_score', 'profit_target', 'stop_loss', 'holding_period', 'is_0dte', 'time_decay']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            st.dataframe(signals_df[available_cols].round(4), use_container_width=True, hide_index=True)
                            if signals_df.iloc[0]['thresholds']:
                                th = signals_df.iloc[0]['thresholds']
                                st.info(f"Thresholds: Δ ≤ {th['delta_max']:.2f} | Γ ≥ {th['gamma_min']:.3f} | "
                                        f"Θ ≤ {th['theta_base']:.3f} | RSI < {th['rsi_max']:.1f} | Vol > {th['volume_min']}")
                            with st.expander("View Conditions for Top Signal"):
                                if signals_df.iloc[0]['passed_conditions']:
                                    st.write("✅ Passed Conditions:")
                                    for condition in signals_df.iloc[0]['passed_conditions']:
                                        st.write(f"- {condition}")
                                else:
                                    st.info("No conditions passed")
                            st.success(f"Found {len(put_signals)} put signals!")
                        else:
                            st.info("No put signals found.")
                            if not puts_filtered.empty:
                                sample_put = puts_filtered.iloc[0]
                                is_0dte = sample_put.get('is_0dte', False)
                                result = generate_signal(sample_put, "put", df, is_0dte)
                                if 'failed_conditions' in result:
                                    st.write("Top put option failed conditions:")
                                    for condition in result['failed_conditions']:
                                        st.write(f"- {condition}")
                    else:
                        st.info("No put options for selected filters.")
                del calls_filtered, puts_filtered, df  # Memory cleanup
        with tab2:
            if 'df' in locals() and not df.empty:
                st.subheader("📊 Stock Data & Indicators")
                if is_premarket():
                    st.info("🔔 Pre-market data")
                elif not is_market_open():
                    st.info("🔔 After-hours data")
                latest = df.iloc[-1]
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Current Price", f"${latest['Close']:.2f}")
                with col2:
                    st.metric("EMA 9", f"${latest['EMA_9']:.2f}" if not pd.isna(latest['EMA_9']) else "N/A")
                with col3:
                    st.metric("EMA 20", f"${latest['EMA_20']:.2f}" if not pd.isna(latest['EMA_20']) else "N/A")
                with col4:
                    st.metric("EMA 50", f"${latest['EMA_50']:.2f}" if not pd.isna(latest['EMA_50']) else "N/A")
                with col5:
                    st.metric("EMA 200", f"${latest['EMA_200']:.2f}" if not pd.isna(latest['EMA_200']) else "N/A")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RSI", f"{latest['RSI']:.1f}" if not pd.isna(latest['RSI']) else "N/A")
                with col2:
                    st.metric("VWAP", f"${latest['VWAP']:.2f}" if not pd.isna(latest['VWAP']) else "N/A")
                with col3:
                    st.metric("MACD", f"{latest['MACD']:.2f}" if not pd.isna(latest['MACD']) else "N/A")
                with col4:
                    st.metric("Volatility (ATR%)", f"{latest['ATR_pct']*100:.2f}%" if not pd.isna(latest['ATR_pct']) else "N/A")
                st.subheader("Recent Data")
                display_df = df.tail(10)[['Close', 'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'RSI', 'VWAP', 'MACD', 'MACD_Signal', 'Keltner_Upper', 'Keltner_Lower', 'ATR_pct', 'Volume', 'avg_vol']].round(2)
                display_df['ATR_pct'] = display_df['ATR_pct'] * 100
                display_df['Volume Ratio'] = display_df['Volume'] / display_df['avg_vol']
                st.dataframe(display_df.rename(columns={'ATR_pct': 'ATR%', 'avg_vol': 'Avg Vol'}), use_container_width=True)
        with tab3:
            st.subheader("🔍 Analysis Details")
            if enable_auto_refresh:
                st.info(f"🔄 Auto-refresh: Every {refresh_interval} second{'s' if refresh_interval != 1 else ''}")
            else:
                st.info("🔄 Auto-refresh disabled")
            if 'calls_filtered' in locals() and not calls_filtered.empty:
                st.write("**Sample Call Analysis:**")
                sample_call = calls_filtered.iloc[0]
                is_0dte = sample_call.get('is_0dte', False)
                result = generate_signal(sample_call, "call", df, is_0dte)
                st.json(result)
            st.write("**Current Signal Thresholds:**")
            st.json(SIGNAL_THRESHOLDS)
            st.write("**Profit Targets:**")
            st.json(CONFIG['PROFIT_TARGETS'])
            st.write("**System Configuration:**")
            st.json(CONFIG)
    except Exception as e:
        logger.error(f"Main interface error: {str(e)}")
        st.error(f"Error: {str(e)}. Refresh page or try another ticker.")
else:
    st.info("Enter a ticker to begin.")
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Steps to analyze options:**
        1. Enter a stock ticker (e.g., SPY, QQQ, AAPL)
        2. Configure auto-refresh settings (optional)
        3. Select expiration filter (0DTE or near-term)
        4. Adjust strike range
        5. Filter by moneyness (ITM, ATM, OTM)
        6. Review signals

        **Key Features:**
        - **Real-Time Refresh:** Updates every 1 second
        - **0DTE Time Decay:** Adjusts signals for theta decay
        - **Robust Error Handling:** Prevents crashes
        - **Dynamic Thresholds:** Adapts to volatility
        - **Profit Targets & Stops:** Clear exit strategies
        """)
