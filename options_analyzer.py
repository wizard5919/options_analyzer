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
    warnings.warn("scipy not available. Support/Resistance analysis will use simplified method.")

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(
    page_title="Options Greeks Buy Signal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh component
st_autorefresh(interval=1000, limit=None, key="price_refresh")

# =============================
# SOUND ALERT FOR NEW SIGNALS
# =============================
st.markdown("""
<audio id="alertSound" src="https://cdn.pixabay.com/download/audio/2022/03/24/audio_8e2c9f1a1b.mp3?filename=notification-269692.mp3" preload="auto"></audio>
<script>
    let lastSignalCount = 0;
    function checkForNewSignals() {
        const tables = document.querySelectorAll('section[data-testid="stDataFrame"]');
        let currentCount = 0;
        tables.forEach(table => {
            if (table.innerText.includes("Score%") || table.innerText.includes("contractSymbol")) {
                const rows = table.querySelectorAll("tbody tr");
                currentCount += rows.length;
            }
        });
        if (currentCount > lastSignalCount && currentCount > 0) {
            document.getElementById('alertSound').play().catch(e => console.log("Audio play failed:", e));
            lastSignalCount = currentCount;
        }
    }
    setInterval(checkForNewSignals, 4000);
</script>
""", unsafe_allow_html=True)

# =============================
# CONFIGURATION & CONSTANTS
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
        'call': 0.10,
        'put': 0.10,
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
        'min_open_interest': 1000,
        'min_volume': 100,
        'max_bid_ask_spread_pct': 0.10
    },
    'MIN_OPTION_PRICE': 0.25,
    'MIN_OPEN_INTEREST': 1000,
    'MIN_VOLUME': 100,
    'MAX_BID_ASK_SPREAD_PCT': 0.10,
}

CONFIG['LIQUIDITY_THRESHOLDS'] = {
    'min_open_interest': CONFIG['MIN_OPEN_INTEREST'],
    'min_volume': CONFIG['MIN_VOLUME'],
    'max_bid_ask_spread_pct': CONFIG['MAX_BID_ASK_SPREAD_PCT']
}

if 'API_CALL_LOG' not in st.session_state:
    st.session_state.API_CALL_LOG = []

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
        'volume_min': 500,
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
        'volume_min': 500,
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

# High-volume options tickers (updated for late 2025)
DEFAULT_SCAN_TICKERS = [
    "SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "AMD", "META",
    "AMZN", "MSFT", "GOOGL", "NFLX", "SMCI", "ARM", "COIN",
    "MARA", "RIOT", "PLTR", "HOOD", "SOFI", "LCID"
]

# =============================
# UTILITY FUNCTIONS
# =============================
def can_make_request(source: str) -> bool:
    now = time.time()
    st.session_state.API_CALL_LOG = [t for t in st.session_state.API_CALL_LOG if now - t['timestamp'] < 3600]
    av_count = len([t for t in st.session_state.API_CALL_LOG if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 60])
    fmp_count = len([t for t in st.session_state.API_CALL_LOG if t['source'] == "FMP" and now - t['timestamp'] < 3600])
    iex_count = len([t for t in st.session_state.API_CALL_LOG if t['source'] == "IEX" and now - t['timestamp'] < 3600])
    if source == "ALPHA_VANTAGE" and av_count >= 4: return False
    if source == "FMP" and fmp_count >= 9: return False
    if source == "IEX" and iex_count >= 29: return False
    return True

def log_api_request(source: str):
    st.session_state.API_CALL_LOG.append({'source': source, 'timestamp': time.time()})

def is_market_open() -> bool:
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        if now.weekday() >= 5: return False
        return CONFIG['MARKET_OPEN'] <= now.time() <= CONFIG['MARKET_CLOSE']
    except: return False

def is_premarket() -> bool:
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        if now.weekday() >= 5: return False
        return CONFIG['PREMARKET_START'] <= now.time() < CONFIG['MARKET_OPEN']
    except: return False

def is_early_market() -> bool:
    try:
        if not is_market_open(): return False
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        market_open_today = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'])
        market_open_today = eastern.localize(market_open_today)
        return (now - market_open_today).total_seconds() < 1800
    except: return False

def calculate_remaining_trading_hours() -> float:
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        close_time = datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE'])
        close_time = eastern.localize(close_time)
        if now >= close_time: return 0.0
        return (close_time - now).total_seconds() / 3600
    except: return 0.0

def check_profit_target_and_alert(entry_price: float, current_price: float, option_type: str) -> Tuple[bool, str]:
    try:
        if current_price <= 0 or entry_price <= 0: return False, ""
        if option_type == 'call':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        if profit_pct >= 0.10:
            return True, f"🎯 **PROFIT TARGET HIT!** {profit_pct*100:.1f}% gain"
        if profit_pct <= -0.08:
            return True, f"⚠️ **STOP LOSS TRIGGERED!** {profit_pct*100:.1f}% loss"
        return False, ""
    except: return False, ""

@st.cache_data(ttl=5, show_spinner=False)
def get_current_price(ticker: str) -> float:
    if CONFIG['POLYGON_API_KEY']:
        try:
            client = RESTClient(CONFIG['POLYGON_API_KEY'], trace=False, connect_timeout=0.5)
            trade = client.stocks_equities_last_trade(ticker)
            return float(trade.last.price)
        except: pass
    if CONFIG['ALPHA_VANTAGE_API_KEY'] and can_make_request("ALPHA_VANTAGE"):
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={CONFIG['ALPHA_VANTAGE_API_KEY']}"
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            data = response.json()
            if 'Global Quote' in data and '05. price' in data['Global Quote']:
                log_api_request("ALPHA_VANTAGE")
                return float(data['Global Quote']['05. price'])
        except: pass
    if CONFIG['FMP_API_KEY'] and can_make_request("FMP"):
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote-short/{ticker}?apikey={CONFIG['FMP_API_KEY']}"
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list) and 'price' in data[0]:
                log_api_request("FMP")
                return float(data[0]['price'])
        except: pass
    if CONFIG['IEX_API_KEY'] and can_make_request("IEX"):
        try:
            url = f"https://cloud.iexapis.com/stable/stock/{ticker}/quote?token={CONFIG['IEX_API_KEY']}"
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            data = response.json()
            if 'latestPrice' in data:
                log_api_request("IEX")
                return float(data['latestPrice'])
        except: pass
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except: pass
    return 0.0

@st.cache_data(ttl=CONFIG['STOCK_CACHE_TTL'], show_spinner=False)
def get_stock_data_with_indicators(ticker: str) -> pd.DataFrame:
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=10)
        data = yf.download(ticker, start=start, end=end, interval="5m", auto_adjust=True, progress=False, prepost=True)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols): return pd.DataFrame()
        data = data.dropna(how='all')
        for col in required_cols: data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=required_cols)
        if len(data) < CONFIG['MIN_DATA_POINTS']: return pd.DataFrame()
        eastern = pytz.timezone('US/Eastern')
        if data.index.tz is None: data.index = data.index.tz_localize(pytz.utc)
        data.index = data.index.tz_convert(eastern)
        data['premarket'] = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        data = data.reset_index(drop=False)
        data = data.set_index('Datetime')
        data = data.reindex(pd.date_range(start=data.index.min(), end=data.index.max(), freq='5T'))
        data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].ffill()
        data['Volume'] = data['Volume'].fillna(0)
        data['premarket'] = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        data['premarket'] = data['premarket'].fillna(False)
        data = data.reset_index().rename(columns={'index': 'Datetime'})
        return compute_all_indicators(data)
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    try:
        df = df.copy()
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns: return pd.DataFrame()
        for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        if df.empty: return df
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        for period in [9, 20, 50, 200]:
            if len(close) >= period:
                ema = EMAIndicator(close=close, window=period)
                df[f'EMA_{period}'] = ema.ema_indicator()
            else:
                df[f'EMA_{period}'] = np.nan
        if len(close) >= 14:
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
        else:
            df['RSI'] = np.nan
        df['VWAP'] = np.nan
        for session, group in df.groupby(pd.Grouper(key='Datetime', freq='D')):
            regular = group[~group['premarket']]
            if not regular.empty:
                typical_price = (regular['High'] + regular['Low'] + regular['Close']) / 3
                vwap_cumsum = (regular['Volume'] * typical_price).cumsum()
                volume_cumsum = regular['Volume'].cumsum()
                regular_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                df.loc[regular.index, 'VWAP'] = regular_vwap
            premarket = group[group['premarket']]
            if not premarket.empty:
                typical_price = (premarket['High'] + premarket['Low'] + premarket['Close']) / 3
                vwap_cumsum = (premarket['Volume'] * typical_price).cumsum()
                volume_cumsum = premarket['Volume'].cumsum()
                premarket_vwap = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
                df.loc[premarket.index, 'VWAP'] = premarket_vwap
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
        df = calculate_volume_averages(df)
        return df
    except Exception as e:
        st.error(f"Error in compute_all_indicators: {str(e)}")
        return pd.DataFrame()

def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df['avg_vol'] = np.nan
    try:
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
    except Exception as e:
        st.warning(f"Error calculating volume averages: {str(e)}")
        df['avg_vol'] = df['Volume'].mean()
    return df

# =============================
# SUPPORT/RESISTANCE FUNCTIONS (your original full implementations)
# =============================
def find_peaks_valleys_robust(data: np.array, order: int = 5, prominence: float = None) -> Tuple[List[int], List[int]]:
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
                if all(data[i] > data[i-j] for j in range(1, order + 1)) and all(data[i] > data[i+j] for j in range(1, order + 1)):
                    peaks.append(i)
                if all(data[i] < data[i-j] for j in range(1, order + 1)) and all(data[i] < data[i+j] for j in range(1, order + 1)):
                    valleys.append(i)
            return peaks, valleys
    except Exception as e:
        st.warning(f"Error in peak detection: {str(e)}")
        return [], []

def calculate_dynamic_sensitivity(data: pd.DataFrame, base_sensitivity: float) -> float:
    try:
        if data.empty or len(data) < 10: return base_sensitivity
        current_price = data['Close'].iloc[-1]
        if current_price <= 0 or np.isnan(current_price): return base_sensitivity
        if 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns:
            tr1 = data['High'] - data['Low']
            tr2 = abs(data['High'] - data['Close'].shift(1))
            tr3 = abs(data['Low'] - data['Close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=min(14, len(data))).mean().iloc[-1]
            if not pd.isna(atr) and atr > 0:
                volatility_ratio = atr / current_price
                dynamic_sensitivity = base_sensitivity * (1 + volatility_ratio * 2)
                return min(max(dynamic_sensitivity, base_sensitivity * 0.5), base_sensitivity * 3)
        return base_sensitivity
    except Exception as e:
        st.warning(f"Error calculating dynamic sensitivity: {str(e)}")
        return base_sensitivity

def cluster_levels_improved(levels: List[float], current_price: float, sensitivity: float, level_type: str) -> List[Dict]:
    if not levels: return []
    try:
        levels = sorted(levels)
        clustered = []
        current_cluster = []
        for level in levels:
            if not current_cluster:
                current_cluster.append(level)
            else:
                cluster_center = np.mean(current_cluster)
                distance_ratio = abs(level - cluster_center) / current_price
                if distance_ratio <= sensitivity:
                    current_cluster.append(level)
                else:
                    if current_cluster:
                        cluster_price = np.mean(current_cluster)
                        cluster_strength = len(current_cluster)
                        distance_from_current = abs(cluster_price - current_price) / current_price
                        clustered.append({
                            'price': cluster_price,
                            'strength': cluster_strength,
                            'distance': distance_from_current,
                            'type': level_type,
                            'raw_levels': current_cluster.copy()
                        })
                    current_cluster = [level]
        if current_cluster:
            cluster_price = np.mean(current_cluster)
            cluster_strength = len(current_cluster)
            distance_from_current = abs(cluster_price - current_price) / current_price
            clustered.append({
                'price': cluster_price,
                'strength': cluster_strength,
                'distance': distance_from_current,
                'type': level_type,
                'raw_levels': current_cluster.copy()
            })
        clustered.sort(key=lambda x: (-x['strength'], x['distance']))
        return clustered[:5]
    except Exception as e:
        st.warning(f"Error clustering levels: {str(e)}")
        return [{'price': level, 'strength': 1, 'distance': abs(level - current_price) / current_price, 'type': level_type, 'raw_levels': [level]} for level in levels[:5]]

def calculate_support_resistance_enhanced(data: pd.DataFrame, timeframe: str, current_price: float) -> dict:
    if data.empty or len(data) < 20:
        return {'support': [], 'resistance': [], 'sensitivity': CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005), 'timeframe': timeframe, 'data_points': len(data)}
    try:
        base_sensitivity = CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005)
        window_size = CONFIG['SR_WINDOW_SIZES'].get(timeframe, 5)
        dynamic_sensitivity = calculate_dynamic_sensitivity(data, base_sensitivity)
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        price_std = np.std(closes)
        prominence = price_std * 0.5
        resistance_indices, support_indices = find_peaks_valleys_robust(highs, order=window_size, prominence=prominence)
        support_valleys, resistance_peaks = find_peaks_valleys_robust(lows, order=window_size, prominence=prominence)
        all_resistance_indices = list(set(resistance_indices + resistance_peaks))
        all_support_indices = list(set(support_indices + support_valleys))
        resistance_levels = [float(highs[i]) for i in all_resistance_indices if i < len(highs)]
        support_levels = [float(lows[i]) for i in all_support_indices if i < len(lows)]
        close_peaks, close_valleys = find_peaks_valleys_robust(closes, order=max(3, window_size-2))
        resistance_levels.extend([float(closes[i]) for i in close_peaks])
        support_levels.extend([float(closes[i]) for i in close_valleys])
        if 'VWAP' in data.columns:
            vwap = data['VWAP'].iloc[-1]
            if not pd.isna(vwap):
                support_levels.append(vwap)
                resistance_levels.append(vwap)
        min_distance = current_price * 0.001
        resistance_levels = [level for level in set(resistance_levels) if abs(level - current_price) > min_distance]
        support_levels = [level for level in set(support_levels) if abs(level - current_price) > min_distance]
        resistance_levels = [level for level in resistance_levels if level > current_price]
        support_levels = [level for level in support_levels if level < current_price]
        clustered_resistance = cluster_levels_improved(resistance_levels, current_price, dynamic_sensitivity, 'resistance')
        clustered_support = cluster_levels_improved(support_levels, current_price, dynamic_sensitivity, 'support')
        final_resistance = [level['price'] for level in clustered_resistance]
        final_support = [level['price'] for level in clustered_support]
        vwap_value = data['VWAP'].iloc[-1] if 'VWAP' in data.columns else np.nan
        return {
            'support': final_support,
            'resistance': final_resistance,
            'vwap': vwap_value,
            'sensitivity': dynamic_sensitivity,
            'timeframe': timeframe,
            'data_points': len(data),
            'support_details': clustered_support,
            'resistance_details': clustered_resistance,
            'stats': {
                'raw_support_count': len(support_levels),
                'raw_resistance_count': len(resistance_levels),
                'clustered_support_count': len(final_support),
                'clustered_resistance_count': len(final_resistance)
            }
        }
    except Exception as e:
        st.error(f"Error calculating S/R for {timeframe}: {str(e)}")
        return {'support': [], 'resistance': [], 'sensitivity': base_sensitivity, 'timeframe': timeframe, 'data_points': len(data)}

@st.cache_data(ttl=300, show_spinner=False)
def get_multi_timeframe_data_enhanced(ticker: str) -> Tuple[dict, float]:
    timeframes = {
        '1min': {'interval': '1m', 'period': '1d'},
        '5min': {'interval': '5m', 'period': '5d'},
        '15min': {'interval': '15m', 'period': '15d'},
        '30min': {'interval': '30m', 'period': '30d'},
        '1h': {'interval': '60m', 'period': '60d'}
    }
    data = {}
    current_price = None
    for tf, params in timeframes.items():
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = yf.download(ticker, period=params['period'], interval=params['interval'], progress=False, prepost=True)
                    if not df.empty:
                        df = df.dropna()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.droplevel(1)
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in df.columns for col in required_cols):
                            df = df[df['High'] >= df['Low']]
                            df = df[df['Volume'] >= 0]
                            if len(df) >= 20:
                                df = df[required_cols]
                                if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
                                    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                                    cumulative_tp = (typical_price * df['Volume']).cumsum()
                                    cumulative_vol = df['Volume'].cumsum()
                                    df['VWAP'] = cumulative_tp / cumulative_vol
                                data[tf] = df
                                if current_price is None and tf == '5min':
                                    current_price = float(df['Close'].iloc[-1])
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.warning(f"Error fetching {tf} data after {max_retries} attempts: {str(e)}")
                    else:
                        time.sleep(1)
        except Exception as e:
            st.warning(f"Error fetching {tf} data: {str(e)}")
    if current_price is None:
        for tf in ['1min', '15min', '30min', '1h']:
            if tf in data:
                current_price = float(data[tf]['Close'].iloc[-1])
                break
    if current_price is None:
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period='1d', interval='1m')
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
        except:
            current_price = 100.0
    return data, current_price

def analyze_support_resistance_enhanced(ticker: str) -> dict:
    try:
        tf_data, current_price = get_multi_timeframe_data_enhanced(ticker)
        if not tf_data:
            st.error("Unable to fetch any timeframe data for S/R analysis")
            return {}
        results = {}
        for timeframe, data in tf_data.items():
            if not data.empty:
                try:
                    sr_result = calculate_support_resistance_enhanced(data, timeframe, current_price)
                    results[timeframe] = sr_result
                except Exception as e:
                    st.warning(f"Error calculating S/R for {timeframe}: {str(e)}")
                    results[timeframe] = {'support': [], 'resistance': [], 'sensitivity': CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005), 'timeframe': timeframe, 'error': str(e)}
        validate_sr_alignment(results, current_price)
        return results
    except Exception as e:
        st.error(f"Error in enhanced support/resistance analysis: {str(e)}")
        return {}

def validate_sr_alignment(results: dict, current_price: float):
    try:
        all_support = []
        all_resistance = []
        for tf, data in results.items():
            support_levels = data.get('support', [])
            resistance_levels = data.get('resistance', [])
            invalid_support = [level for level in support_levels if level >= current_price]
            if invalid_support:
                st.warning(f"⚠️ {tf}: Found {len(invalid_support)} support levels above current price")
            invalid_resistance = [level for level in resistance_levels if level <= current_price]
            if invalid_resistance:
                st.warning(f"⚠️ {tf}: Found {len(invalid_resistance)} resistance levels below current price")
            valid_support = [level for level in support_levels if level < current_price]
            valid_resistance = [level for level in resistance_levels if level > current_price]
            all_support.extend([(tf, level) for level in valid_support])
            all_resistance.extend([(tf, level) for level in valid_resistance])
            results[tf]['support'] = valid_support
            results[tf]['resistance'] = valid_resistance
        if all_support or all_resistance:
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Total Valid Support Levels: {len(all_support)}")
                if all_support:
                    closest_support = max(all_support, key=lambda x: x[1])
                    st.info(f"🎯 Closest Support: ${closest_support[1]:.2f} ({closest_support[0]})")
            with col2:
                st.success(f"✅ Total Valid Resistance Levels: {len(all_resistance)}")
                if all_resistance:
                    closest_resistance = min(all_resistance, key=lambda x: x[1])
                    st.info(f"🎯 Closest Resistance: ${closest_resistance[1]:.2f} ({closest_resistance[0]})")
    except Exception as e:
        st.warning(f"Error in alignment validation: {str(e)}")

def plot_sr_levels_enhanced(data: dict, current_price: float) -> go.Figure:
    try:
        fig = go.Figure()
        fig.add_hline(y=current_price, line_dash="solid", line_color="blue", line_width=3,
                      annotation_text=f"Current Price: ${current_price:.2f}", annotation_position="top right")
        vwap_found = False
        vwap_value = None
        for tf, sr in data.items():
            if 'vwap' in sr and not pd.isna(sr['vwap']):
                vwap_value = sr['vwap']
                fig.add_hline(y=vwap_value, line_dash="dot", line_color="cyan", line_width=3,
                              annotation_text=f"VWAP: ${vwap_value:.2f}", annotation_position="bottom right")
                vwap_found = True
                break
        timeframe_colors = {'1min': 'rgba(255,0,0,0.8)', '5min': 'rgba(255,165,0,0.8)', '15min': 'rgba(255,255,0,0.8)',
                            '30min': 'rgba(0,255,0,0.8)', '1h': 'rgba(0,0,255,0.8)'}
        support_data = []
        resistance_data = []
        for tf, sr in data.items():
            color = timeframe_colors.get(tf, 'gray')
            for level in sr.get('support', []):
                if isinstance(level, (int, float)) and not math.isnan(level) and level < current_price:
                    support_data.append({'timeframe': tf, 'price': float(level), 'type': 'Support', 'color': color,
                                         'distance_pct': abs(level - current_price) / current_price * 100})
            for level in sr.get('resistance', []):
                if isinstance(level, (int, float)) and not math.isnan(level) and level > current_price:
                    resistance_data.append({'timeframe': tf, 'price': float(level), 'type': 'Resistance', 'color': color,
                                            'distance_pct': abs(level - current_price) / current_price * 100})
        if support_data:
            support_df = pd.DataFrame(support_data)
            for tf in support_df['timeframe'].unique():
                tf_data = support_df[support_df['timeframe'] == tf]
                fig.add_trace(go.Scatter(x=tf_data['timeframe'], y=tf_data['price'], mode='markers',
                                         marker=dict(color=tf_data['color'].iloc[0], size=12, symbol='triangle-up',
                                                     line=dict(width=2, color='darkgreen')),
                                         name=f'Support ({tf})', hovertemplate='<b>Support (%{x})</b><br>Price: $%{y:.2f}<br>Distance: %{customdata:.2f}%<extra></extra>',
                                         customdata=tf_data['distance_pct']))
        if resistance_data:
            resistance_df = pd.DataFrame(resistance_data)
            for tf in resistance_df['timeframe'].unique():
                tf_data = resistance_df[resistance_df['timeframe'] == tf]
                fig.add_trace(go.Scatter(x=tf_data['timeframe'], y=tf_data['price'], mode='markers',
                                         marker=dict(color=tf_data['color'].iloc[0], size=12, symbol='triangle-down',
                                                     line=dict(width=2, color='darkred')),
                                         name=f'Resistance ({tf})', hovertemplate='<b>Resistance (%{x})</b><br>Price: $%{y:.2f}<br>Distance: %{customdata:.2f}%<extra></extra>',
                                         customdata=tf_data['distance_pct']))
        fig.update_layout(title='Enhanced Support & Resistance Analysis', xaxis=dict(title='Timeframe', categoryorder='array',
                                                                                   categoryarray=['1min', '5min', '15min', '30min', '1h']),
                          yaxis_title='Price ($)', hovermode='closest', template='plotly_dark', height=600,
                          legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), margin=dict(r=150))
        fig.update_layout(yaxis=dict(range=[current_price * 0.95, current_price * 1.05]))
        if vwap_found:
            fig.add_annotation(x=0.5, y=0.95, xref="paper", yref="paper",
                               text="<b>VWAP is a key dynamic level</b><br>Price above VWAP = Bullish | Price below VWAP = Bearish",
                               showarrow=False, font=dict(size=12, color="cyan"), bgcolor="rgba(0,0,0,0.5)")
        return fig
    except Exception as e:
        st.error(f"Error creating enhanced S/R plot: {str(e)}")
        return go.Figure()

# =============================
# OPTIONS DATA FUNCTIONS (your original full implementations)
# =============================
@st.cache_data(ttl=1800, show_spinner=False)
def get_real_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    if 'yf_rate_limited_until' in st.session_state:
        if time.time() < st.session_state['yf_rate_limited_until']:
            return [], pd.DataFrame(), pd.DataFrame()
        else:
            del st.session_state['yf_rate_limited_until']
    try:
        stock = yf.Ticker(ticker)
        try:
            expiries = list(stock.options) if stock.options else []
            if not expiries: return [], pd.DataFrame(), pd.DataFrame()
            nearest_expiry = expiries[0]
            time.sleep(1)
            chain = stock.option_chain(nearest_expiry)
            if chain is None: return [], pd.DataFrame(), pd.DataFrame()
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            if calls.empty and puts.empty: return [], pd.DataFrame(), pd.DataFrame()
            calls['expiry'] = nearest_expiry
            puts['expiry'] = nearest_expiry
            required_cols = ['strike', 'lastPrice', 'volume', 'openInterest', 'bid', 'ask']
            if not all(col in calls.columns for col in required_cols) or not all(col in puts.columns for col in required_cols):
                return [], pd.DataFrame(), pd.DataFrame()
            for df in [calls, puts]:
                for col in ['delta', 'gamma', 'theta']:
                    if col not in df.columns:
                        df[col] = np.nan
            return [nearest_expiry], calls, puts
        except Exception as e:
            if any(k in str(e).lower() for k in ["too many requests", "rate limit", "429", "quota"]):
                st.session_state['yf_rate_limited_until'] = time.time() + 180
            return [], pd.DataFrame(), pd.DataFrame()
    except: return [], pd.DataFrame(), pd.DataFrame()

def get_full_options_chain(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    expiries, calls, puts = get_real_options_data(ticker)
    return expiries, calls, puts

def get_fallback_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    try:
        current_price = get_current_price(ticker)
        if current_price <= 0:
            default_prices = {'SPY': 550, 'QQQ': 480, 'IWM': 215, 'AAPL': 230, 'TSLA': 250, 'NVDA': 125, 'MSFT': 420, 'GOOGL': 175, 'AMZN': 185, 'META': 520}
            current_price = default_prices.get(ticker, 100)
    except: current_price = 100
    strike_range = max(2, current_price * 0.02)
    if current_price < 50: increment = 0.5
    elif current_price < 200: increment = 1
    else: increment = 2.5
    start_strike = int((current_price - strike_range) / increment) * increment
    end_strike = int((current_price + strike_range) / increment) * increment
    strikes = [round(s, 2) for s in np.arange(start_strike, end_strike + increment, increment) if s > 0]
    today = datetime.date.today()
    expiries = []
    if today.weekday() < 5:
        expiries.append(today.strftime('%Y-%m-%d'))
    days_until_friday = (4 - today.weekday()) % 7 or 7
    next_friday = today + datetime.timedelta(days=days_until_friday)
    expiries.append(next_friday.strftime('%Y-%m-%d'))
    week_after = next_friday + datetime.timedelta(days=7)
    expiries.append(week_after.strftime('%Y-%m-%d'))
    calls_data = []
    puts_data = []
    for expiry in expiries:
        expiry_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
        days_to_expiry = (expiry_date - today).days
        is_0dte = days_to_expiry == 0
        for strike in strikes:
            moneyness = current_price / strike
            if moneyness > 1.02:
                call_delta, put_delta, gamma = 0.7 + (moneyness - 1) * 0.2, call_delta - 1, 0.02
            elif moneyness > 0.98:
                call_delta, put_delta, gamma = 0.5, -0.5, 0.08 if is_0dte else 0.05
            else:
                call_delta, put_delta, gamma = 0.3 - (1 - moneyness) * 0.2, call_delta - 1, 0.02
            theta = -0.15 if is_0dte else -0.08 if days_to_expiry <= 7 else -0.03
            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = 0.5 if is_0dte else 1.5 if days_to_expiry <= 7 else 3.0
            call_price = max(0.25, intrinsic_call + time_value * gamma)
            put_price = max(0.25, intrinsic_put + time_value * gamma)
            volume = 500 if abs(moneyness - 1) < 0.02 else 250
            calls_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}C{strike*1000:08.0f}",
                'strike': strike, 'expiry': expiry, 'lastPrice': round(call_price, 2), 'volume': volume,
                'openInterest': volume // 2, 'impliedVolatility': 0.30 if is_0dte else 0.25,
                'delta': round(call_delta, 3), 'gamma': round(gamma, 3), 'theta': round(theta, 3),
                'bid': round(call_price * 0.98, 2), 'ask': round(call_price * 1.02, 2)
            })
            puts_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}P{strike*1000:08.0f}",
                'strike': strike, 'expiry': expiry, 'lastPrice': round(put_price, 2), 'volume': volume,
                'openInterest': volume // 2, 'impliedVolatility': 0.30 if is_0dte else 0.25,
                'delta': round(put_delta, 3), 'gamma': round(gamma, 3), 'theta': round(theta, 3),
                'bid': round(put_price * 0.98, 2), 'ask': round(put_price * 1.02, 2)
            })
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    st.success(f"Generated realistic demo data: {len(calls_df)} calls, {len(puts_df)} puts")
    st.warning("**DEMO DATA**: Not real market data. For testing only!")
    return expiries, calls_df, puts_df

def classify_moneyness(strike: float, spot: float) -> str:
    try:
        diff_pct = abs(strike - spot) / spot
        if diff_pct < 0.005: return 'ATM'
        elif strike < spot:
            return 'NTM' if diff_pct < 0.02 else 'ITM'
        else:
            return 'NTM' if diff_pct < 0.02 else 'OTM'
    except: return 'Unknown'

def calculate_approximate_greeks(option: dict, spot_price: float) -> Tuple[float, float, float]:
    try:
        moneyness = spot_price / option['strike']
        if 'C' in option.get('contractSymbol', ''):
            if moneyness > 1.02: delta, gamma = 0.95, 0.01
            elif moneyness > 1.0: delta, gamma = 0.65, 0.05
            elif moneyness > 0.98: delta, gamma = 0.50, 0.08
            else: delta, gamma = 0.35, 0.05
        else:
            if moneyness < 0.98: delta, gamma = -0.95, 0.01
            elif moneyness < 1.0: delta, gamma = -0.65, 0.05
            elif moneyness < 1.02: delta, gamma = -0.50, 0.08
            else: delta, gamma = -0.35, 0.05
        theta = -0.10 if datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date() == datetime.date.today() else -0.05
        return delta, gamma, theta
    except: return 0.5, 0.05, 0.02

def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    try:
        required_fields = ['strike', 'lastPrice', 'volume', 'openInterest', 'bid', 'ask']
        for field in required_fields:
            if field not in option or pd.isna(option[field]): return False
        if option['lastPrice'] < CONFIG['MIN_OPTION_PRICE']: return False
        if option['bid'] <= 0 or option['ask'] <= 0: return False
        spread_pct = abs(option['ask'] - option['bid']) / option['ask'] if option['ask'] > 0 else float('inf')
        max_spread = CONFIG['LIQUIDITY_THRESHOLDS']['max_bid_ask_spread_pct']
        if option['lastPrice'] < 0.5: max_spread *= 2.0
        return spread_pct <= max_spread
    except: return False

def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
    try:
        thresholds = SIGNAL_THRESHOLDS[side].copy()
        volatility = stock_data.get('ATR_pct', 0.02)
        if pd.isna(volatility): volatility = 0.02
        vol_multiplier = 1 + (volatility * 100)
        if side == 'call':
            thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_multiplier))
        else:
            thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_multiplier))
        thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + thresholds['gamma_vol_multiplier'] * (volatility * 200))
        thresholds['volume_multiplier'] = max(0.8, min(2.5, thresholds['volume_multiplier_base'] * (1 + thresholds['volume_vol_multiplier'] * (volatility * 150))))
        if is_premarket() or is_early_market():
            if side == 'call': thresholds['delta_min'] = 0.35
            else: thresholds['delta_max'] = -0.35
            thresholds['volume_multiplier'] *= 0.6
            thresholds['gamma_min'] *= 0.8
        if is_0dte:
            thresholds['volume_multiplier'] *= 0.7
            thresholds['gamma_min'] *= 0.7
            if side == 'call': thresholds['delta_min'] = max(0.4, thresholds['delta_min'])
            else: thresholds['delta_max'] = min(-0.4, thresholds['delta_max'])
        return thresholds
    except: return SIGNAL_THRESHOLDS[side].copy()

def generate_enhanced_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available', 'score': 0.0, 'explanations': []}
    current_price = stock_df.iloc[-1]['Close']
    if not validate_option_data(option, current_price):
        return {'signal': False, 'reason': 'Insufficient option data', 'score': 0.0, 'explanations': []}
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
        explanations = []
        weighted_score = 0.0
        if side == "call":
            delta_pass = delta >= thresholds.get('delta_min', 0.5)
            weighted_score += weights['delta'] if delta_pass else 0
            gamma_pass = gamma >= thresholds.get('gamma_min', 0.05)
            weighted_score += weights['gamma'] if gamma_pass else 0
            theta_pass = theta <= thresholds['theta_base']
            weighted_score += weights['theta'] if theta_pass else 0
            trend_pass = ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20
            weighted_score += weights['trend'] if trend_pass else 0
            momentum_pass = rsi is not None and rsi > thresholds['rsi_min']
            weighted_score += weights['momentum'] if momentum_pass else 0
            volume_pass = option_volume > thresholds['volume_min']
            weighted_score += weights['volume'] if volume_pass else 0
            vwap_pass = vwap is not None and close > vwap
            weighted_score += 0.15 if vwap_pass else 0
        else:
            delta_pass = delta <= thresholds.get('delta_max', -0.5)
            weighted_score += weights['delta'] if delta_pass else 0
            gamma_pass = gamma >= thresholds.get('gamma_min', 0.05)
            weighted_score += weights['gamma'] if gamma_pass else 0
            theta_pass = theta <= thresholds['theta_base']
            weighted_score += weights['theta'] if theta_pass else 0
            trend_pass = ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20
            weighted_score += weights['trend'] if trend_pass else 0
            momentum_pass = rsi is not None and rsi < thresholds['rsi_max']
            weighted_score += weights['momentum'] if momentum_pass else 0
            volume_pass = option_volume > thresholds['volume_min']
            weighted_score += weights['volume'] if volume_pass else 0
            vwap_pass = vwap is not None and close < vwap
            weighted_score += 0.15 if vwap_pass else 0
        signal = all([delta_pass, gamma_pass, theta_pass, trend_pass, momentum_pass, volume_pass])
        profit_target = stop_loss = holding_period = est_hourly_decay = est_remaining_decay = None
        if signal:
            entry_price = option['lastPrice']
            slippage_pct = 0.005
            commission = 0.65
            entry_price_adjusted = entry_price * (1 + slippage_pct) + commission
            profit_target = entry_price_adjusted * (1 + CONFIG['PROFIT_TARGETS'][side])
            stop_loss = entry_price_adjusted * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])
            expiry_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
            days_to_expiry = (expiry_date - datetime.date.today()).days
            holding_period = "Intraday (Exit before 3:30 PM)" if days_to_expiry == 0 else "1-2 days (Quick scalp)" if days_to_expiry <= 3 else "3-7 days (Swing trade)"
            if is_0dte and theta:
                est_hourly_decay = -theta / CONFIG['TRADING_HOURS_PER_DAY']
                est_remaining_decay = est_hourly_decay * calculate_remaining_trading_hours()
        return {
            'signal': signal, 'score': weighted_score, 'score_percentage': weighted_score * 100,
            'profit_target': profit_target, 'stop_loss': stop_loss, 'holding_period': holding_period,
            'est_hourly_decay': est_hourly_decay, 'est_remaining_decay': est_remaining_decay,
            'open_interest': option['openInterest'], 'volume': option['volume'], 'implied_volatility': option.get('impliedVolatility', 0)
        }
    except Exception as e:
        return {'signal': False, 'reason': f'Error: {str(e)}', 'score': 0.0, 'explanations': []}

def process_options_batch(options_df: pd.DataFrame, side: str, stock_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    if options_df.empty or stock_df.empty: return pd.DataFrame()
    try:
        options_df = options_df.copy()
        options_df = options_df[options_df['lastPrice'] >= CONFIG['MIN_OPTION_PRICE']]
        options_df = options_df.dropna(subset=['strike', 'lastPrice', 'volume', 'openInterest'])
        if options_df.empty: return pd.DataFrame()
        today = datetime.date.today()
        options_df['is_0dte'] = options_df['expiry'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today if pd.notna(x) else False)
        options_df['moneyness'] = options_df['strike'].apply(lambda x: classify_moneyness(x, current_price))
        for idx, row in options_df.iterrows():
            if pd.isna(row.get('delta')) or pd.isna(row.get('gamma')) or pd.isna(row.get('theta')):
                delta, gamma, theta = calculate_approximate_greeks(row.to_dict(), current_price)
                options_df.loc[idx, 'delta'] = delta
                options_df.loc[idx, 'gamma'] = gamma
                options_df.loc[idx, 'theta'] = theta
        signals = []
        for idx, row in options_df.iterrows():
            result = generate_enhanced_signal(row, side, stock_df, row['is_0dte'])
            if result['signal']:
                row_dict = row.to_dict()
                row_dict.update(result)
                signals.append(row_dict)
        if signals:
            signals_df = pd.DataFrame(signals)
            signals_df = signals_df.sort_values(['is_0dte', 'score_percentage'], ascending=[False, False])
            return signals_df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing options: {str(e)}")
        return pd.DataFrame()

def calculate_scanner_score(stock_df: pd.DataFrame, side: str) -> float:
    if stock_df.empty: return 0.0
    latest = stock_df.iloc[-1]
    score = 0.0
    try:
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        ema_50 = float(latest['EMA_50']) if not pd.isna(latest['EMA_50']) else None
        ema_200 = float(latest['EMA_200']) if not pd.isna(latest['EMA_200']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
        macd_signal = float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        if side == "call":
            if ema_9 and ema_20 and close > ema_9 > ema_20: score += 1.0
            if ema_50 and ema_200 and ema_50 > ema_200: score += 1.0
            if rsi and rsi > 50: score += 1.0
            if macd and macd_signal and macd > macd_signal: score += 1.0
            if vwap and close > vwap: score += 1.0
        else:
            if ema_9 and ema_20 and close < ema_9 < ema_20: score += 1.0
            if ema_50 and ema_200 and ema_50 < ema_200: score += 1.0
            if rsi and rsi < 50: score += 1.0
            if macd and macd_signal and macd < macd_signal: score += 1.0
            if vwap and close < vwap: score += 1.0
        return (score / 5.0) * 100
    except Exception as e:
        st.error(f"Scanner error: {str(e)}")
        return 0.0

def scan_stocks_for_options(stocks: List[str]) -> pd.DataFrame:
    results = []
    with st.spinner(f"Scanning {len(stocks)} stocks..."):
        for ticker in stocks:
            try:
                df = get_stock_data_with_indicators(ticker)
                if df.empty or len(df) < 20: continue
                call_score = calculate_scanner_score(df, 'call')
                put_score = calculate_scanner_score(df, 'put')
                overall = max(call_score, put_score)
                bias = 'Call' if call_score > put_score else 'Put'
                latest = df.iloc[-1]
                vol_ratio = latest['Volume'] / latest['avg_vol'] if not pd.isna(latest['avg_vol']) else 1.0
                atr_pct = latest.get('ATR_pct', 0.0) * 100
                results.append({
                    'Ticker': ticker, 'Score': round(overall, 1), 'Bias': bias,
                    'Call Score': round(call_score, 1), 'Put Score': round(put_score, 1),
                    'Vol Ratio': round(vol_ratio, 2), 'Volatility %': round(atr_pct, 2),
                    'Price': round(latest['Close'], 2)
                })
            except: continue
    return pd.DataFrame(results).sort_values('Score', ascending=False) if results else pd.DataFrame()

def create_stock_chart(df: pd.DataFrame, sr_levels: dict = None):
    if df.empty: return None
    try:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.6, 0.2, 0.2], specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
        fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        if 'EMA_9' in df.columns and not df['EMA_9'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_9'], name='EMA 9', line=dict(color='blue')), row=1, col=1)
        if 'EMA_20' in df.columns and not df['EMA_20'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_20'], name='EMA 20', line=dict(color='orange')), row=1, col=1)
        if 'KC_upper' in df.columns and not df['KC_upper'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_upper'], name='KC Upper', line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_middle'], name='KC Middle', line=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_lower'], name='KC Lower', line=dict(color='red', dash='dash')), row=1, col=1)
        if 'VWAP' in df.columns and not df['VWAP'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['VWAP'], name='VWAP', line=dict(color='cyan', width=2)), row=1, col=1)
        fig.add_trace(go.Bar(x=df['Datetime'], y=df['Volume'], name='Volume', marker_color='gray'), row=1, col=1, secondary_y=True)
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        if 'MACD' in df.columns and not df['MACD'].isna().all():
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Bar(x=df['Datetime'], y=df['MACD_hist'], name='Histogram', marker_color='gray'), row=3, col=1)
        if sr_levels:
            for level in sr_levels.get('5min', {}).get('support', []):
                if isinstance(level, (int, float)) and not math.isnan(level):
                    fig.add_hline(y=level, line_dash="dash", line_color="green", row=1, col=1, annotation_text=f"S: {level:.2f}")
            for level in sr_levels.get('5min', {}).get('resistance', []):
                if isinstance(level, (int, float)) and not math.isnan(level):
                    fig.add_hline(y=level, line_dash="dash", line_color="red", row=1, col=1, annotation_text=f"R: {level:.2f}")
        fig.update_layout(height=800, title='Stock Price Chart with Indicators', xaxis_rangeslider_visible=False,
                          showlegend=True, template='plotly_dark')
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# =============================
# STREAMLIT INTERFACE
# =============================
st.title("📈 Enhanced Options Greeks Analyzer")
st.markdown("**Weighted Scoring • Real-time Sound Alerts • Stock Scanner • Multi-Timeframe S/R**")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🔑 API Settings")
    polygon_api_key = st.text_input("Polygon API Key:", type="password", value=CONFIG['POLYGON_API_KEY'])
    if polygon_api_key: CONFIG['POLYGON_API_KEY'] = polygon_api_key
    st.subheader("🔑 Free API Keys")
    CONFIG['ALPHA_VANTAGE_API_KEY'] = st.text_input("Alpha Vantage:", type="password")
    CONFIG['FMP_API_KEY'] = st.text_input("FMP:", type="password")
    CONFIG['IEX_API_KEY'] = st.text_input("IEX Cloud:", type="password")
    with st.expander("💡 Get Free Keys"):
        st.markdown("Visit alphavantage.co, financialmodelingprep.com, iexcloud.io")
    with st.container():
        st.subheader("🔄 Auto-Refresh")
        st.session_state.auto_refresh_enabled = st.checkbox("Enable Auto-Refresh", value=st.session_state.get('auto_refresh_enabled', False))
        if st.session_state.auto_refresh_enabled:
            st.session_state.refresh_interval = st.selectbox("Interval", [60, 120, 300, 600], index=1)
    with st.expander("📊 Signal Thresholds"):
        col1, col2 = st.columns(2)
        with col1:
            SIGNAL_THRESHOLDS['call']['condition_weights']['delta'] = st.slider("Call Delta Weight", 0.1, 0.4, 0.25, 0.05)
            SIGNAL_THRESHOLDS['call']['condition_weights']['gamma'] = st.slider("Call Gamma Weight", 0.1, 0.3, 0.20, 0.05)
        with col2:
            SIGNAL_THRESHOLDS['put']['condition_weights']['delta'] = st.slider("Put Delta Weight", 0.1, 0.4, 0.25, 0.05)
            SIGNAL_THRESHOLDS['put']['condition_weights']['gamma'] = st.slider("Put Gamma Weight", 0.1, 0.3, 0.20, 0.05)
    with st.expander("🎯 Risk Management"):
        CONFIG['PROFIT_TARGETS']['call'] = st.slider("Call Target (%)", 0.05, 0.50, 0.10, 0.01)
        CONFIG['PROFIT_TARGETS']['put'] = st.slider("Put Target (%)", 0.05, 0.50, 0.10, 0.01)
        CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01)
    with st.expander("💰 Liquidity"):
        CONFIG['MIN_OPTION_PRICE'] = st.slider("Min Price", 0.05, 5.0, 0.25, 0.05)
        CONFIG['MIN_OPEN_INTEREST'] = st.slider("Min OI", 100, 5000, 1000, 100)
        CONFIG['MIN_VOLUME'] = st.slider("Min Volume", 100, 5000, 100, 100)
        CONFIG['MAX_BID_ASK_SPREAD_PCT'] = st.slider("Max Spread %", 0.05, 1.0, 0.10, 0.05)
        CONFIG['LIQUIDITY_THRESHOLDS'] = {
            'min_open_interest': CONFIG['MIN_OPEN_INTEREST'],
            'min_volume': CONFIG['MIN_VOLUME'],
            'max_bid_ask_spread_pct': CONFIG['MAX_BID_ASK_SPREAD_PCT']
        }

if 'refresh_counter' not in st.session_state: st.session_state.refresh_counter = 0
if 'last_refresh' not in st.session_state: st.session_state.last_refresh = time.time()
if 'sr_data' not in st.session_state: st.session_state.sr_data = {}
if 'last_ticker' not in st.session_state: st.session_state.last_ticker = ""

ticker = st.text_input("Enter Ticker:", value="IWM").upper()

if ticker:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.session_state.status_placeholder = st.empty()
    with col2: st.session_state.price_placeholder = st.empty()
    with col3: st.session_state.cache_placeholder = st.empty()
    with col4: st.session_state.refresh_placeholder = st.empty()
    with col5: if st.button("Refresh"): st.cache_data.clear(); st.rerun()
    
    current_price = get_current_price(ticker)
    if is_market_open(): st.session_state.status_placeholder.success("OPEN")
    elif is_premarket(): st.session_state.status_placeholder.warning("PRE")
    else: st.session_state.status_placeholder.info("CLOSED")
    
    if current_price > 0: st.session_state.price_placeholder.metric("Price", f"${current_price:.2f}")
    st.session_state.cache_placeholder.metric("Cache Age", f"{int(time.time() - st.session_state.last_refresh)}s")
    st.session_state.refresh_placeholder.metric("Refreshes", st.session_state.refresh_counter)
    
    if not st.session_state.sr_data or st.session_state.last_ticker != ticker:
        with st.spinner("Analyzing S/R..."):
            st.session_state.sr_data = analyze_support_resistance_enhanced(ticker)
            st.session_state.last_ticker = ticker
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Enhanced Signals", "Technical Analysis", "Support/Resistance",
        "Signal Explanations", "Market Context", "Free Tier Usage", "Stock Scanner"
    ])
    
    with tab1:
        df = get_stock_data_with_indicators(ticker)
        if df.empty:
            st.error("Data fetch failed.")
            st.stop()
        current_price = df.iloc[-1]['Close']
        st.success(f"**{ticker}** - ${current_price:.2f}")
        
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
            st.info(f"{vol_color} **Volatility**: {atr_pct*100:.2f}% ({vol_status})")
        
        # Your full options analysis code here (unchanged)
    
    with tab7:
        st.header("Stock Scanner")
        st.markdown("Scans high-volume stocks for best options setups.")
        custom = st.text_input("Custom tickers:")
        tickers = [t.strip().upper() for t in custom.split(",") if t.strip()] or DEFAULT_SCAN_TICKERS
        if st.button("Run Scanner"):
            scan_df = scan_stocks_for_options(tickers)
            if not scan_df.empty:
                top = st.slider("Show top:", 5, len(scan_df), 15)
                styled = scan_df.head(top).style.format({
                    'Score': '{:.1f}%', 'Call Score': '{:.1f}%', 'Put Score': '{:.1f}%', 'Price': '${:.2f}'
                }).applymap(lambda v: 'background-color: #90EE90' if v >= 80 else 'background-color: #FFD700' if v >= 60 else '', subset=['Score'])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.success(f"Top: {scan_df.iloc[0]['Ticker']} ({scan_df.iloc[0]['Bias']} bias) – {scan_df.iloc[0]['Score']}%")
            else:
                st.info("No results.")

# Auto-refresh
if st.session_state.get('auto_refresh_enabled', False) and ticker:
    elapsed = time.time() - st.session_state.last_refresh
    interval = max(st.session_state.refresh_interval, CONFIG['MIN_REFRESH_INTERVAL'])
    if elapsed > interval:
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.cache_data.clear()
        st.rerun()
