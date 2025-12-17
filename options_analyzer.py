"""
DISCLAIMER:
- This tool is for EDUCATIONAL / RESEARCH USE ONLY.
- It does NOT and CANNOT provide 100% accurate buy/sell signals.
- Options trading is risky. Always manage risk and use your own judgment.
"""

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

# Optional SciPy for advanced S/R peak detection
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Support/Resistance analysis will use simplified method.")

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ----------------- Streamlit Page Config ----------------- #
st.set_page_config(
    page_title="Options Greeks Buy Signal Analyzer (Enhanced)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh for real-time updates (lightweight)
refresh_interval = st_autorefresh(interval=1000, limit=None, key="price_refresh")


# =============================
# GLOBAL CONFIGURATION
# =============================

CONFIG = {
    'POLYGON_API_KEY': '',  # Will be set from sidebar
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
    'MIN_REFRESH_INTERVAL': 60,  # hard floor for safety
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
    # Base liquidity thresholds (adjustable in sidebar)
    'LIQUIDITY_THRESHOLDS': {
        'min_open_interest': 1000,
        'min_volume': 500,
        'max_bid_ask_spread_pct': 0.25  # 25%
    },
    'MIN_OPTION_PRICE': 0.20,   # Minimum option price to consider
    'MIN_OPEN_INTEREST': 1000,
    'MIN_VOLUME': 500,
    'MAX_BID_ASK_SPREAD_PCT': 0.25,
}

# Keep LIQUIDITY_THRESHOLDS in sync with top-level values
CONFIG['LIQUIDITY_THRESHOLDS'] = {
    'min_open_interest': CONFIG['MIN_OPEN_INTEREST'],
    'min_volume': CONFIG['MIN_VOLUME'],
    'max_bid_ask_spread_pct': CONFIG['MAX_BID_ASK_SPREAD_PCT']
}

# Initialize session state logging for API calls
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
# UTILITY: RATE-LIMIT TRACKING
# =============================

def can_make_request(source: str) -> bool:
    """Check if we can make another request for a given free API source."""
    now = time.time()

    # Clean up old entries (older than 1 hour)
    st.session_state.API_CALL_LOG = [
        t for t in st.session_state.API_CALL_LOG
        if now - t['timestamp'] < 3600
    ]

    av_count = len([t for t in st.session_state.API_CALL_LOG
                    if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 60])
    fmp_count = len([t for t in st.session_state.API_CALL_LOG
                     if t['source'] == "FMP" and now - t['timestamp'] < 3600])
    iex_count = len([t for t in st.session_state.API_CALL_LOG
                     if t['source'] == "IEX" and now - t['timestamp'] < 3600])

    if source == "ALPHA_VANTAGE" and av_count >= 4:
        return False
    if source == "FMP" and fmp_count >= 9:
        return False
    if source == "IEX" and iex_count >= 29:
        return False

    return True


def log_api_request(source: str):
    """Log an API request to track usage."""
    st.session_state.API_CALL_LOG.append({
        'source': source,
        'timestamp': time.time()
    })


# =============================
# SUPPORT / RESISTANCE HELPERS
# =============================

def find_peaks_valleys_robust(data: np.array, order: int = 5, prominence: float = None) -> Tuple[List[int], List[int]]:
    """Robust peak and valley detection with optional prominence."""
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
                # Peak
                if all(data[i] > data[i - j] and data[i] > data[i + j] for j in range(1, order + 1)):
                    peaks.append(i)
                # Valley
                if all(data[i] < data[i - j] and data[i] < data[i + j] for j in range(1, order + 1)):
                    valleys.append(i)
            return peaks, valleys
    except Exception as e:
        st.warning(f"Error in peak detection: {str(e)}")
        return [], []


def calculate_dynamic_sensitivity(data: pd.DataFrame, base_sensitivity: float) -> float:
    """Dynamic sensitivity based on ATR volatility."""
    try:
        if data.empty or len(data) < 10:
            return base_sensitivity

        current_price = data['Close'].iloc[-1]
        if current_price <= 0 or np.isnan(current_price):
            return base_sensitivity

        # ATR-based volatility
        tr1 = data['High'] - data['Low']
        tr2 = (data['High'] - data['Close'].shift(1)).abs()
        tr3 = (data['Low'] - data['Close'].shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=min(14, len(data))).mean().iloc[-1]

        if not pd.isna(atr) and atr > 0:
            vol_ratio = atr / current_price
            dynamic = base_sensitivity * (1 + vol_ratio * 2)
            return min(max(dynamic, base_sensitivity * 0.5), base_sensitivity * 3)

        return base_sensitivity
    except Exception as e:
        st.warning(f"Error calculating dynamic sensitivity: {str(e)}")
        return base_sensitivity


def cluster_levels_improved(levels: List[float], current_price: float,
                            sensitivity: float, level_type: str) -> List[Dict]:
    """Cluster raw S/R levels into strong zones."""
    if not levels:
        return []
    try:
        levels = sorted(levels)
        clustered = []
        current_cluster = []

        for level in levels:
            if not current_cluster:
                current_cluster.append(level)
            else:
                center = np.mean(current_cluster)
                distance_ratio = abs(level - center) / max(current_price, 1e-9)
                if distance_ratio <= sensitivity:
                    current_cluster.append(level)
                else:
                    cluster_price = np.mean(current_cluster)
                    cluster_strength = len(current_cluster)
                    distance_from_current = abs(cluster_price - current_price) / max(current_price, 1e-9)
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
            distance_from_current = abs(cluster_price - current_price) / max(current_price, 1e-9)
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
        return []


def calculate_support_resistance_enhanced(data: pd.DataFrame,
                                          timeframe: str,
                                          current_price: float) -> dict:
    """Enhanced S/R calc for a single timeframe."""
    if data.empty or len(data) < 20:
        return {
            'support': [],
            'resistance': [],
            'sensitivity': CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005),
            'timeframe': timeframe,
            'data_points': len(data)
        }

    base_sensitivity = CONFIG['SR_SENSITIVITY'].get(timeframe, 0.005)
    try:
        window_size = CONFIG['SR_WINDOW_SIZES'].get(timeframe, 5)
        dyn_sens = calculate_dynamic_sensitivity(data, base_sensitivity)

        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values

        price_std = np.std(closes)
        prominence = price_std * 0.5 if price_std > 0 else None

        res_idx, sup_idx = find_peaks_valleys_robust(highs, order=window_size, prominence=prominence)
        sup_valley, res_peak = find_peaks_valleys_robust(lows, order=window_size, prominence=prominence)

        all_res_idx = list(set(res_idx + res_peak))
        all_sup_idx = list(set(sup_idx + sup_valley))

        resistance_levels = [float(highs[i]) for i in all_res_idx if i < len(highs)]
        support_levels = [float(lows[i]) for i in all_sup_idx if i < len(lows)]

        # Add close-based pivots
        close_peaks, close_valleys = find_peaks_valleys_robust(closes, order=max(3, window_size - 2))
        resistance_levels.extend([float(closes[i]) for i in close_peaks])
        support_levels.extend([float(closes[i]) for i in close_valleys])

        # Include VWAP if present
        vwap_val = data['VWAP'].iloc[-1] if 'VWAP' in data.columns else np.nan
        if not pd.isna(vwap_val):
            resistance_levels.append(vwap_val)
            support_levels.append(vwap_val)

        # Filter near-current-price duplicates
        min_dist = current_price * 0.001
        resistance_levels = [l for l in set(resistance_levels) if abs(l - current_price) > min_dist and l > current_price]
        support_levels = [l for l in set(support_levels) if abs(l - current_price) > min_dist and l < current_price]

        clustered_res = cluster_levels_improved(resistance_levels, current_price, dyn_sens, 'resistance')
        clustered_sup = cluster_levels_improved(support_levels, current_price, dyn_sens, 'support')

        return {
            'support': [c['price'] for c in clustered_sup],
            'resistance': [c['price'] for c in clustered_res],
            'vwap': vwap_val,
            'sensitivity': dyn_sens,
            'timeframe': timeframe,
            'data_points': len(data),
            'support_details': clustered_sup,
            'resistance_details': clustered_res,
            'stats': {
                'raw_support_count': len(support_levels),
                'raw_resistance_count': len(resistance_levels),
                'clustered_support_count': len(clustered_sup),
                'clustered_resistance_count': len(clustered_res),
            }
        }
    except Exception as e:
        st.error(f"Error calculating S/R for {timeframe}: {str(e)}")
        return {
            'support': [],
            'resistance': [],
            'sensitivity': base_sensitivity,
            'timeframe': timeframe,
            'data_points': len(data),
            'error': str(e)
        }


@st.cache_data(ttl=300, show_spinner=False)
def get_multi_timeframe_data_enhanced(ticker: str) -> Tuple[dict, float]:
    """Fetch multi-timeframe stock data for S/R."""
    timeframes = {
        '1min': {'interval': '1m', 'period': '1d'},
        '5min': {'interval': '5m', 'period': '5d'},
        '15min': {'interval': '15m', 'period': '15d'},
        '30min': {'interval': '30m', 'period': '30d'},
        '1h': {'interval': '60m', 'period': '60d'},
    }

    data = {}
    current_price = None

    for tf, params in timeframes.items():
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = yf.download(
                        ticker,
                        period=params['period'],
                        interval=params['interval'],
                        progress=False,
                        prepost=True
                    )
                    if not df.empty:
                        df = df.dropna()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.droplevel(1)
                        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(c in df.columns for c in required_cols):
                            df = df[df['High'] >= df['Low']]
                            df = df[df['Volume'] >= 0]
                            if len(df) >= 20:
                                df = df[required_cols]
                                # VWAP
                                typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                                cum_tp = (typical_price * df['Volume']).cumsum()
                                cum_vol = df['Volume'].cumsum()
                                df['VWAP'] = np.where(cum_vol != 0, cum_tp / cum_vol, np.nan)
                                data[tf] = df
                                if current_price is None and tf == '5min':
                                    current_price = float(df['Close'].iloc[-1])
                    break
                except Exception:
                    if attempt == max_retries - 1:
                        st.warning(f"Error fetching {tf} data after {max_retries} attempts.")
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
            t_obj = yf.Ticker(ticker)
            h = t_obj.history(period='1d', interval='1m')
            if not h.empty:
                current_price = float(h['Close'].iloc[-1])
        except Exception:
            current_price = 100.0

    return data, current_price


def analyze_support_resistance_enhanced(ticker: str) -> dict:
    """Top-level S/R analysis for all timeframes."""
    try:
        tf_data, current_price = get_multi_timeframe_data_enhanced(ticker)
        if not tf_data:
            st.error("Unable to fetch any timeframe data for S/R analysis")
            return {}

        st.info(f"📊 Analyzing S/R with current price: ${current_price:.2f}")
        results = {}
        for timeframe, data in tf_data.items():
            if not data.empty:
                try:
                    sr = calculate_support_resistance_enhanced(data, timeframe, current_price)
                    results[timeframe] = sr
                    st.caption(f"✅ {timeframe}: {len(sr['support'])} support, {len(sr['resistance'])} resistance")
                except Exception as e:
                    st.warning(f"Error in S/R for {timeframe}: {str(e)}")
        validate_sr_alignment(results, current_price)
        return results
    except Exception as e:
        st.error(f"Error in enhanced support/resistance analysis: {str(e)}")
        return {}


def validate_sr_alignment(results: dict, current_price: float):
    """Ensure support & resistance are located relative to current price correctly."""
    try:
        st.subheader("🔍 S/R Alignment Validation")
        all_support = []
        all_resistance = []

        for tf, data in results.items():
            sup = data.get('support', [])
            res = data.get('resistance', [])

            invalid_support = [lvl for lvl in sup if lvl >= current_price]
            invalid_res = [lvl for lvl in res if lvl <= current_price]

            if invalid_support:
                st.warning(f"⚠️ {tf}: {len(invalid_support)} 'support' levels above current price")
            if invalid_res:
                st.warning(f"⚠️ {tf}: {len(invalid_res)} 'resistance' levels below current price")

            valid_support = [lvl for lvl in sup if lvl < current_price]
            valid_res = [lvl for lvl in res if lvl > current_price]

            results[tf]['support'] = valid_support
            results[tf]['resistance'] = valid_res

            all_support.extend([(tf, lvl) for lvl in valid_support])
            all_resistance.extend([(tf, lvl) for lvl in valid_res])

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Total Valid Support Levels: {len(all_support)}")
            if all_support:
                closest_sup = max(all_support, key=lambda x: x[1])
                st.info(f"🎯 Closest Support: ${closest_sup[1]:.2f} ({closest_sup[0]})")
        with col2:
            st.success(f"✅ Total Valid Resistance Levels: {len(all_resistance)}")
            if all_resistance:
                closest_res = min(all_resistance, key=lambda x: x[1])
                st.info(f"🎯 Closest Resistance: ${closest_res[1]:.2f} ({closest_res[0]})")
    except Exception as e:
        st.warning(f"Error in alignment validation: {str(e)}")


def plot_sr_levels_enhanced(data: dict, current_price: float) -> go.Figure:
    """Plot multi-timeframe S/R relative to current price."""
    fig = go.Figure()
    try:
        # Current price line
        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="blue",
            line_width=3,
            annotation_text=f"Current Price: ${current_price:.2f}",
            annotation_position="top right"
        )

        # VWAP line (take first found)
        vwap_found = False
        for tf, sr in data.items():
            v = sr.get('vwap', np.nan)
            if not pd.isna(v):
                fig.add_hline(
                    y=v,
                    line_dash="dot",
                    line_color="cyan",
                    line_width=3,
                    annotation_text=f"VWAP: ${v:.2f}",
                    annotation_position="bottom right"
                )
                vwap_found = True
                break

        colors = {
            '1min': 'rgba(255,0,0,0.8)',
            '5min': 'rgba(255,165,0,0.8)',
            '15min': 'rgba(255,255,0,0.8)',
            '30min': 'rgba(0,255,0,0.8)',
            '1h': 'rgba(0,0,255,0.8)'
        }

        support_points = []
        resistance_points = []

        for tf, sr in data.items():
            col = colors.get(tf, 'gray')
            for lvl in sr.get('support', []):
                if isinstance(lvl, (int, float)) and not math.isnan(lvl) and lvl < current_price:
                    support_points.append({
                        'tf': tf,
                        'price': float(lvl),
                        'color': col,
                        'dist': abs(lvl - current_price) / max(current_price, 1e-9) * 100,
                    })
            for lvl in sr.get('resistance', []):
                if isinstance(lvl, (int, float)) and not math.isnan(lvl) and lvl > current_price:
                    resistance_points.append({
                        'tf': tf,
                        'price': float(lvl),
                        'color': col,
                        'dist': abs(lvl - current_price) / max(current_price, 1e-9) * 100,
                    })

        if support_points:
            sdf = pd.DataFrame(support_points)
            for tf in sdf['tf'].unique():
                sub = sdf[sdf['tf'] == tf]
                fig.add_trace(go.Scatter(
                    x=sub['tf'],
                    y=sub['price'],
                    mode='markers',
                    marker=dict(
                        color=sub['color'].iloc[0],
                        size=12,
                        symbol='triangle-up',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name=f'Support ({tf})',
                    customdata=sub['dist'],
                    hovertemplate=(
                        f'<b>Support ({tf})</b><br>'
                        'Price: $%{y:.2f}<br>'
                        'Distance: %{customdata:.2f}%<extra></extra>'
                    )
                ))

        if resistance_points:
            rdf = pd.DataFrame(resistance_points)
            for tf in rdf['tf'].unique():
                sub = rdf[rdf['tf'] == tf]
                fig.add_trace(go.Scatter(
                    x=sub['tf'],
                    y=sub['price'],
                    mode='markers',
                    marker=dict(
                        color=sub['color'].iloc[0],
                        size=12,
                        symbol='triangle-down',
                        line=dict(width=2, color='darkred')
                    ),
                    name=f'Resistance ({tf})',
                    customdata=sub['dist'],
                    hovertemplate=(
                        f'<b>Resistance ({tf})</b><br>'
                        'Price: $%{y:.2f}<br>'
                        'Distance: %{customdata:.2f}%<extra></extra>'
                    )
                ))

        fig.update_layout(
            title="Enhanced Support & Resistance Analysis",
            xaxis=dict(
                title='Timeframe',
                categoryorder='array',
                categoryarray=['1min', '5min', '15min', '30min', '1h']
            ),
            yaxis_title='Price ($)',
            hovermode='closest',
            template='plotly_dark',
            height=600,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(r=150)
        )

        fig.update_yaxes(range=[current_price * 0.95, current_price * 1.05])

        if vwap_found:
            fig.add_annotation(
                x=0.5, y=0.95,
                xref="paper", yref="paper",
                text="<b>VWAP is a key dynamic level</b><br>Price above VWAP = Bullish | Price below VWAP = Bearish",
                showarrow=False,
                font=dict(size=12, color="cyan"),
                bgcolor="rgba(0,0,0,0.5)"
            )

    except Exception as e:
        st.error(f"Error creating enhanced S/R plot: {str(e)}")

    return fig


# =============================
# MARKET SESSION HELPERS
# =============================

def is_market_open() -> bool:
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        if now.weekday() >= 5:
            return False
        return CONFIG['MARKET_OPEN'] <= now.time() <= CONFIG['MARKET_CLOSE']
    except Exception:
        return False


def is_premarket() -> bool:
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        if now.weekday() >= 5:
            return False
        return CONFIG['PREMARKET_START'] <= now.time() < CONFIG['MARKET_OPEN']
    except Exception:
        return False


def is_early_market() -> bool:
    try:
        if not is_market_open():
            return False
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        mo = eastern.localize(datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN']))
        return (now - mo).total_seconds() < 1800
    except Exception:
        return False


def calculate_remaining_trading_hours() -> float:
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        close_time = eastern.localize(datetime.datetime.combine(now.date(), CONFIG['MARKET_CLOSE']))
        if now >= close_time:
            return 0.0
        return (close_time - now).total_seconds() / 3600
    except Exception:
        return 0.0


# =============================
# PRICE FETCHING
# =============================

@st.cache_data(ttl=5, show_spinner=False)
def get_current_price(ticker: str) -> float:
    """Price from Polygon -> Alpha Vantage -> FMP -> IEX -> yfinance."""
    # Polygon
    if CONFIG['POLYGON_API_KEY']:
        try:
            client = RESTClient(CONFIG['POLYGON_API_KEY'], trace=False, connect_timeout=0.5)
            trade = client.stocks_equities_last_trade(ticker)
            return float(trade.last.price)
        except Exception:
            pass

    # Alpha Vantage
    if CONFIG['ALPHA_VANTAGE_API_KEY'] and can_make_request("ALPHA_VANTAGE"):
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={CONFIG['ALPHA_VANTAGE_API_KEY']}"
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()
            data = resp.json()
            if 'Global Quote' in data and '05. price' in data['Global Quote']:
                log_api_request("ALPHA_VANTAGE")
                return float(data['Global Quote']['05. price'])
        except Exception:
            pass

    # FMP
    if CONFIG['FMP_API_KEY'] and can_make_request("FMP"):
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote-short/{ticker}?apikey={CONFIG['FMP_API_KEY']}"
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()
            data = resp.json()
            if data and isinstance(data, list) and 'price' in data[0]:
                log_api_request("FMP")
                return float(data[0]['price'])
        except Exception:
            pass

    # IEX
    if CONFIG['IEX_API_KEY'] and can_make_request("IEX"):
        try:
            url = f"https://cloud.iexapis.com/stable/stock/{ticker}/quote?token={CONFIG['IEX_API_KEY']}"
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()
            data = resp.json()
            if 'latestPrice' in data:
                log_api_request("IEX")
                return float(data['latestPrice'])
        except Exception:
            pass

    # yfinance fallback
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m', prepost=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception:
        pass

    return 0.0


# =============================
# STOCK DATA + INDICATORS
# =============================

@st.cache_data(ttl=CONFIG['STOCK_CACHE_TTL'], show_spinner=False)
def get_stock_data_with_indicators(ticker: str) -> pd.DataFrame:
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
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if any(c not in data.columns for c in required):
            return pd.DataFrame()

        data = data.dropna(how='all')
        for col in required:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(subset=required)

        if len(data) < CONFIG['MIN_DATA_POINTS']:
            return pd.DataFrame()

        eastern = pytz.timezone('US/Eastern')
        if data.index.tz is None:
            data.index = data.index.tz_localize(pytz.utc)
        data.index = data.index.tz_convert(eastern)

        data['premarket'] = (data.index.time >= CONFIG['PREMARKET_START']) & (data.index.time < CONFIG['MARKET_OPEN'])
        data = data.reset_index(drop=False)

        # Fill gaps (5-minute bars)
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
    if df.empty:
        return df
    try:
        df = df.copy()
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required)
        if df.empty:
            return df

        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)

        # EMAs
        for period in [9, 20, 50, 200]:
            if len(close) >= period:
                df[f'EMA_{period}'] = EMAIndicator(close=close, window=period).ema_indicator()
            else:
                df[f'EMA_{period}'] = np.nan

        # RSI
        if len(close) >= 14:
            df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        else:
            df['RSI'] = np.nan

        # Session VWAP
        df['VWAP'] = np.nan
        for session, group in df.groupby(pd.Grouper(key='Datetime', freq='D')):
            if group.empty:
                continue
            # regular
            regular = group[~group['premarket']]
            if not regular.empty:
                tp = (regular['High'] + regular['Low'] + regular['Close']) / 3
                vwap_cum = (regular['Volume'] * tp).cumsum()
                vol_cum = regular['Volume'].cumsum()
                vwap_vals = np.where(vol_cum != 0, vwap_cum / vol_cum, np.nan)
                df.loc[regular.index, 'VWAP'] = vwap_vals
            # premarket
            pre = group[group['premarket']]
            if not pre.empty:
                tp = (pre['High'] + pre['Low'] + pre['Close']) / 3
                vwap_cum = (pre['Volume'] * tp).cumsum()
                vol_cum = pre['Volume'].cumsum()
                vwap_vals = np.where(vol_cum != 0, vwap_cum / vol_cum, np.nan)
                df.loc[pre.index, 'VWAP'] = vwap_vals

        # ATR
        if len(close) >= 14:
            atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
            df['ATR'] = atr
            current_price = df['Close'].iloc[-1]
            df['ATR_pct'] = atr / close if current_price > 0 else np.nan
        else:
            df['ATR'] = np.nan
            df['ATR_pct'] = np.nan

        # MACD + Keltner
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
    if df.empty:
        return df
    df = df.copy()
    df['avg_vol'] = np.nan
    try:
        for date, group in df.groupby(df['Datetime'].dt.date):
            regular = group[~group['premarket']]
            if not regular.empty:
                reg_avg = regular['Volume'].expanding(min_periods=1).mean()
                df.loc[regular.index, 'avg_vol'] = reg_avg
            pre = group[group['premarket']]
            if not pre.empty:
                pre_avg = pre['Volume'].expanding(min_periods=1).mean()
                df.loc[pre.index, 'avg_vol'] = pre_avg
        overall = df['Volume'].mean()
        df['avg_vol'] = df['avg_vol'].fillna(overall)
    except Exception as e:
        st.warning(f"Error calculating volume averages: {str(e)}")
        df['avg_vol'] = df['Volume'].mean()
    return df


# =============================
# OPTIONS DATA
# =============================

@st.cache_data(ttl=1800, show_spinner=False)
def get_real_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Real options data via yfinance (nearest expiry only)."""
    if 'yf_rate_limited_until' in st.session_state:
        remaining = st.session_state['yf_rate_limited_until'] - time.time()
        if remaining > 0:
            return [], pd.DataFrame(), pd.DataFrame()
        else:
            del st.session_state['yf_rate_limited_until']

    try:
        stock = yf.Ticker(ticker)
        expiries = list(stock.options) if stock.options else []
        if not expiries:
            return [], pd.DataFrame(), pd.DataFrame()

        nearest = expiries[0]
        time.sleep(1)  # gentle delay
        chain = stock.option_chain(nearest)
        if chain is None:
            return [], pd.DataFrame(), pd.DataFrame()

        calls = chain.calls.copy()
        puts = chain.puts.copy()
        if calls.empty and puts.empty:
            return [], pd.DataFrame(), pd.DataFrame()

        calls['expiry'] = nearest
        puts['expiry'] = nearest

        required = ['strike', 'lastPrice', 'volume', 'openInterest', 'bid', 'ask']
        if not all(c in calls.columns for c in required) or not all(c in puts.columns for c in required):
            return [], pd.DataFrame(), pd.DataFrame()

        for df_name, df in (('calls', calls), ('puts', puts)):
            for greek in ['delta', 'gamma', 'theta']:
                if greek not in df.columns:
                    df[greek] = np.nan

        return [nearest], calls, puts

    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["too many requests", "rate limit", "429", "quota"]):
            st.session_state['yf_rate_limited_until'] = time.time() + 180  # 3 minutes
        return [], pd.DataFrame(), pd.DataFrame()


def clear_rate_limit():
    if 'yf_rate_limited_until' in st.session_state:
        del st.session_state['yf_rate_limited_until']
    st.success("✅ Rate limit status cleared")
    st.rerun()


def get_full_options_chain(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    expiries, calls, puts = get_real_options_data(ticker)
    return expiries, calls, puts


def get_fallback_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Demo options data if real data is unavailable."""
    try:
        current_price = get_current_price(ticker)
        if current_price <= 0:
            default_prices = {
                'SPY': 550, 'QQQ': 480, 'IWM': 215, 'AAPL': 230,
                'TSLA': 250, 'NVDA': 125, 'MSFT': 420, 'GOOGL': 175,
                'AMZN': 185, 'META': 520
            }
            current_price = default_prices.get(ticker, 100)
    except Exception:
        current_price = 100

    strike_range = max(5, current_price * 0.1)
    if current_price < 50:
        inc = 1
    elif current_price < 200:
        inc = 5
    else:
        inc = 10

    start_strike = int((current_price - strike_range) / inc) * inc
    end_strike = int((current_price + strike_range) / inc) * inc
    strikes = [s for s in range(start_strike, end_strike + inc, inc) if s > 0]

    today = datetime.date.today()
    expiries = []
    if today.weekday() < 5:
        expiries.append(today.strftime('%Y-%m-%d'))
    days_to_fri = (4 - today.weekday()) % 7 or 7
    next_fri = today + datetime.timedelta(days=days_to_fri)
    expiries.append(next_fri.strftime('%Y-%m-%d'))
    expiries.append((next_fri + datetime.timedelta(days=7)).strftime('%Y-%m-%d'))

    st.info(f"📊 Generated {len(strikes)} strikes around ${current_price:.2f} for {ticker}")

    calls_data = []
    puts_data = []

    for expiry in expiries:
        exp_date = datetime.datetime.strptime(expiry, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        is_0dte = dte == 0

        for strike in strikes:
            m = current_price / strike
            if m > 1.05:
                call_delta = 0.7 + (m - 1) * 0.2
                put_delta = call_delta - 1
                gamma = 0.02
            elif m > 0.95:
                call_delta = 0.5
                put_delta = -0.5
                gamma = 0.08 if is_0dte else 0.05
            else:
                call_delta = 0.3 - (1 - m) * 0.2
                put_delta = call_delta - 1
                gamma = 0.02

            if is_0dte:
                theta = -0.1
            elif dte <= 7:
                theta = -0.05
            else:
                theta = -0.02

            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = 5 if is_0dte else 10 if dte <= 7 else 15
            call_price = intrinsic_call + time_value * gamma
            put_price = intrinsic_put + time_value * gamma

            vol = 1000 if abs(m - 1) < 0.05 else 500

            calls_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}C{int(strike * 1000):08d}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(call_price, 2),
                'volume': vol,
                'openInterest': vol // 2,
                'impliedVolatility': 0.25,
                'delta': round(call_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(call_price * 0.98, 2),
                'ask': round(call_price * 1.02, 2)
            })
            puts_data.append({
                'contractSymbol': f"{ticker}{expiry.replace('-', '')}P{int(strike * 1000):08d}",
                'strike': strike,
                'expiry': expiry,
                'lastPrice': round(put_price, 2),
                'volume': vol,
                'openInterest': vol // 2,
                'impliedVolatility': 0.25,
                'delta': round(put_delta, 3),
                'gamma': round(gamma, 3),
                'theta': round(theta, 3),
                'bid': round(put_price * 0.98, 2),
                'ask': round(put_price * 1.02, 2)
            })

    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)

    st.success(f"✅ Generated realistic demo data: {len(calls_df)} calls, {len(puts_df)} puts")
    st.warning("⚠️ DEMO DATA ONLY – not live market data.")
    return expiries, calls_df, puts_df


def classify_moneyness(strike: float, spot: float) -> str:
    try:
        diff = abs(strike - spot)
        diff_pct = diff / spot
        if diff_pct < 0.01:
            return 'ATM'
        elif strike < spot:
            return 'NTM' if diff_pct < 0.03 else 'ITM'
        else:
            return 'NTM' if diff_pct < 0.03 else 'OTM'
    except Exception:
        return 'Unknown'


def calculate_approximate_greeks(option: dict, spot_price: float) -> Tuple[float, float, float]:
    """Approximate Greeks if missing."""
    try:
        m = spot_price / option['strike']
        is_call = 'C' in option.get('contractSymbol', '')
        if is_call:
            if m > 1.03:
                delta, gamma = 0.95, 0.01
            elif m > 1.0:
                delta, gamma = 0.65, 0.05
            elif m > 0.97:
                delta, gamma = 0.50, 0.08
            else:
                delta, gamma = 0.35, 0.05
        else:  # put
            if m < 0.97:
                delta, gamma = -0.95, 0.01
            elif m < 1.0:
                delta, gamma = -0.65, 0.05
            elif m < 1.03:
                delta, gamma = -0.50, 0.08
            else:
                delta, gamma = -0.35, 0.05

        exp_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
        theta = 0.05 if exp_date == datetime.date.today() else 0.02
        return delta, gamma, theta
    except Exception:
        return 0.5, 0.05, 0.02


def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    """Basic and liquidity validation for an option row."""
    try:
        required = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'bid', 'ask']
        for f in required:
            if f not in option or pd.isna(option[f]):
                return False
        if option['lastPrice'] <= 0:
            return False

        liq = CONFIG['LIQUIDITY_THRESHOLDS']
        if option['openInterest'] < liq['min_open_interest']:
            return False
        if option['volume'] < liq['min_volume']:
            return False

        spread = abs(option['ask'] - option['bid'])
        spread_pct = spread / option['lastPrice'] if option['lastPrice'] > 0 else float('inf')
        if spread_pct > liq['max_bid_ask_spread_pct']:
            return False

        # Fill missing Greeks
        if pd.isna(option.get('delta')) or pd.isna(option.get('gamma')) or pd.isna(option.get('theta')):
            d, g, t = calculate_approximate_greeks(option.to_dict(), spot_price)
            option['delta'], option['gamma'], option['theta'] = d, g, t

        return True
    except Exception:
        return False


def calculate_dynamic_thresholds(stock_data: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
    """Dynamic thresholds for delta/gamma/volume given volatility."""
    try:
        thresholds = SIGNAL_THRESHOLDS[side].copy()
        volatility = stock_data.get('ATR_pct', 0.02)
        if pd.isna(volatility):
            volatility = 0.02

        vol_mult = 1 + (volatility * 100)

        if side == 'call':
            thresholds['delta_min'] = max(0.3, min(0.8, thresholds['delta_base'] * vol_mult))
        else:
            thresholds['delta_max'] = min(-0.3, max(-0.8, thresholds['delta_base'] * vol_mult))

        thresholds['gamma_min'] = thresholds['gamma_base'] * (1 + thresholds['gamma_vol_multiplier'] * (volatility * 200))
        thresholds['volume_multiplier'] = max(
            0.8,
            min(2.5, thresholds['volume_multiplier_base'] * (1 + thresholds['volume_vol_multiplier'] * (volatility * 150)))
        )

        if is_premarket() or is_early_market():
            if side == 'call':
                thresholds['delta_min'] = max(thresholds.get('delta_min', 0.5), 0.35)
            else:
                thresholds['delta_max'] = min(thresholds.get('delta_max', -0.5), -0.35)
            thresholds['volume_multiplier'] *= 0.6
            thresholds['gamma_min'] *= 0.8

        if is_0dte:
            thresholds['volume_multiplier'] *= 0.7
            thresholds['gamma_min'] *= 0.7
            if side == 'call':
                thresholds['delta_min'] = max(0.4, thresholds.get('delta_min', 0.5))
            else:
                thresholds['delta_max'] = min(-0.4, thresholds.get('delta_max', -0.5))

        return thresholds
    except Exception:
        return SIGNAL_THRESHOLDS[side].copy()


# =============================
# SIGNAL GENERATION
# =============================

def generate_enhanced_signal(option: pd.Series,
                             side: str,
                             stock_df: pd.DataFrame,
                             is_0dte: bool) -> Dict:
    """Generate signal + score for one option contract."""
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data', 'score': 0.0, 'explanations': []}

    current_price = stock_df.iloc[-1]['Close']
    if not validate_option_data(option, current_price):
        return {'signal': False, 'reason': 'Invalid/illiquid option', 'score': 0.0, 'explanations': []}

    latest = stock_df.iloc[-1]
    explanations = []
    conditions = []
    weighted_score = 0.0

    try:
        thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
        weights = thresholds['condition_weights']

        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        opt_vol = float(option['volume'])

        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
        macd_signal = float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else opt_vol

        # Delta condition
        if side == "call":
            delta_pass = delta >= thresholds.get('delta_min', 0.5)
            delta_thr = thresholds.get('delta_min', 0.5)
        else:
            delta_pass = delta <= thresholds.get('delta_max', -0.5)
            delta_thr = thresholds.get('delta_max', -0.5)

        delta_score = weights['delta'] if delta_pass else 0.0
        weighted_score += delta_score
        conditions.append((delta_pass, 'Delta', delta))
        explanations.append({
            'condition': 'Delta',
            'passed': delta_pass,
            'value': delta,
            'threshold': delta_thr,
            'weight': weights['delta'],
            'score': delta_score,
            'explanation': f"Delta {delta:.3f} vs threshold {delta_thr:.2f}"
        })

        # Gamma
        gamma_pass = gamma >= thresholds.get('gamma_min', 0.05)
        gamma_thr = thresholds.get('gamma_min', 0.05)
        gamma_score = weights['gamma'] if gamma_pass else 0.0
        weighted_score += gamma_score
        conditions.append((gamma_pass, 'Gamma', gamma))
        explanations.append({
            'condition': 'Gamma',
            'passed': gamma_pass,
            'value': gamma,
            'threshold': gamma_thr,
            'weight': weights['gamma'],
            'score': gamma_score,
            'explanation': f"Gamma {gamma:.3f} vs threshold {gamma_thr:.3f}"
        })

        # Theta
        theta_pass = theta <= thresholds['theta_base']
        theta_thr = thresholds['theta_base']
        theta_score = weights['theta'] if theta_pass else 0.0
        weighted_score += theta_score
        conditions.append((theta_pass, 'Theta', theta))
        explanations.append({
            'condition': 'Theta',
            'passed': theta_pass,
            'value': theta,
            'threshold': theta_thr,
            'weight': weights['theta'],
            'score': theta_score,
            'explanation': f"Theta {theta:.3f} vs threshold {theta_thr:.3f}"
        })

        # Trend
        if side == "call":
            trend_pass = (ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20)
            trend_text = "Price > EMA9 > EMA20"
        else:
            trend_pass = (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20)
            trend_text = "Price < EMA9 < EMA20"

        trend_score = weights['trend'] if trend_pass else 0.0
        weighted_score += trend_score
        conditions.append((trend_pass, 'Trend', trend_text))
        explanations.append({
            'condition': 'Trend',
            'passed': trend_pass,
            'value': trend_text,
            'threshold': trend_text,
            'weight': weights['trend'],
            'score': trend_score,
            'explanation': f"Trend condition {'passed' if trend_pass else 'failed'}: {trend_text}"
        })

        # Momentum (RSI)
        if rsi is not None:
            if side == "call":
                momentum_pass = rsi > thresholds['rsi_min']
                rsi_thr = thresholds['rsi_min']
            else:
                momentum_pass = rsi < thresholds['rsi_max']
                rsi_thr = thresholds['rsi_max']
        else:
            momentum_pass = False
            rsi_thr = thresholds['rsi_min'] if side == "call" else thresholds['rsi_max']

        mom_score = weights['momentum'] if momentum_pass else 0.0
        weighted_score += mom_score
        conditions.append((momentum_pass, 'Momentum', rsi))
        explanations.append({
            'condition': 'Momentum (RSI)',
            'passed': momentum_pass,
            'value': rsi,
            'threshold': rsi_thr,
            'weight': weights['momentum'],
            'score': mom_score,
            'explanation': f"RSI {rsi:.1f} vs threshold {rsi_thr:.1f}" if rsi is not None else "RSI unavailable"
        })

        # Volume
        volume_pass = opt_vol > thresholds['volume_min']
        volume_thr = thresholds['volume_min']
        vol_score = weights['volume'] if volume_pass else 0.0
        weighted_score += vol_score
        conditions.append((volume_pass, 'Volume', opt_vol))
        explanations.append({
            'condition': 'Volume',
            'passed': volume_pass,
            'value': opt_vol,
            'threshold': volume_thr,
            'weight': weights['volume'],
            'score': vol_score,
            'explanation': f"Option volume {opt_vol:.0f} vs min {volume_thr}"
        })

        # VWAP (extra weight)
        vwap_score = 0.0
        vwap_weight = 0.15  # fixed extra weight
        if vwap is not None and not math.isnan(vwap):
            if side == "call":
                vwap_pass = close > vwap
                desc = f"Price ${close:.2f} vs VWAP ${vwap:.2f} (calls want above)"
            else:
                vwap_pass = close < vwap
                desc = f"Price ${close:.2f} vs VWAP ${vwap:.2f} (puts want below)"

            if vwap_pass:
                vwap_score = vwap_weight
                weighted_score += vwap_score

            conditions.append((vwap_pass, 'VWAP', vwap))
            explanations.append({
                'condition': 'VWAP',
                'passed': vwap_pass,
                'value': vwap,
                'threshold': 'Price > VWAP' if side == 'call' else 'Price < VWAP',
                'weight': vwap_weight,
                'score': vwap_score,
                'explanation': desc
            })
        else:
            vwap_pass = False

        # Final signal: all core conditions must pass (excluding VWAP if you want)
        core_pass = all(p for p, _, _ in conditions if _ != 'VWAP')
        signal = core_pass

        # Normalize score (IMPORTANT FIX)
        total_weight = sum(weights.values()) + vwap_weight
        max_score = total_weight
        score_pct = (weighted_score / max_score * 100) if max_score > 0 else 0.0

        # Risk targets if signal
        profit_target = None
        stop_loss = None
        holding_period = None
        est_hourly_decay = 0.0
        est_remaining_decay = 0.0

        if signal:
            entry_price = float(option['lastPrice'])
            opt_type = 'call' if side == 'call' else 'put'
            slippage_pct = 0.005
            commission = 0.65
            adj_entry = entry_price * (1 + slippage_pct) + commission

            profit_target = adj_entry * (1 + CONFIG['PROFIT_TARGETS'][opt_type])
            stop_loss = adj_entry * (1 - CONFIG['PROFIT_TARGETS']['stop_loss'])

            exp_date = datetime.datetime.strptime(option['expiry'], "%Y-%m-%d").date()
            dte = (exp_date - datetime.date.today()).days
            if dte == 0:
                holding_period = "Intraday (exit before ~3:30pm ET)"
            elif dte <= 3:
                holding_period = "1–2 days (quick scalp)"
            else:
                holding_period = "3–7 days (short swing)"

            if is_0dte and theta:
                est_hourly_decay = -theta / CONFIG['TRADING_HOURS_PER_DAY']
                rem_hours = calculate_remaining_trading_hours()
                est_remaining_decay = est_hourly_decay * rem_hours

        return {
            'signal': signal,
            'score': weighted_score,
            'max_score': max_score,
            'score_percentage': score_pct,
            'explanations': explanations,
            'thresholds': thresholds,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'holding_period': holding_period,
            'est_hourly_decay': est_hourly_decay,
            'est_remaining_decay': est_remaining_decay,
            'passed_conditions': [e['condition'] for e in explanations if e['passed']],
            'failed_conditions': [e['condition'] for e in explanations if not e['passed']],
            'open_interest': option['openInterest'],
            'volume': option['volume'],
            'implied_volatility': option['impliedVolatility']
        }
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}', 'score': 0.0, 'explanations': []}


def process_options_batch(options_df: pd.DataFrame,
                          side: str,
                          stock_df: pd.DataFrame,
                          current_price: float) -> pd.DataFrame:
    """Batch-process calls/puts with the enhanced signal engine."""
    if options_df.empty or stock_df.empty:
        return pd.DataFrame()
    try:
        options_df = options_df.copy()
        options_df = options_df[options_df['lastPrice'] > 0]
        options_df = options_df.dropna(subset=['strike', 'lastPrice', 'volume', 'openInterest'])
        if options_df.empty:
            return pd.DataFrame()

        today = datetime.date.today()
        options_df['is_0dte'] = options_df['expiry'].apply(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today
        )
        options_df['moneyness'] = options_df['strike'].apply(
            lambda s: classify_moneyness(s, current_price)
        )

        # Ensure Greeks
        for idx, row in options_df.iterrows():
            if pd.isna(row.get('delta')) or pd.isna(row.get('gamma')) or pd.isna(row.get('theta')):
                d, g, t = calculate_approximate_greeks(row.to_dict(), current_price)
                options_df.loc[idx, 'delta'] = d
                options_df.loc[idx, 'gamma'] = g
                options_df.loc[idx, 'theta'] = t

        signals = []
        for idx, row in options_df.iterrows():
            result = generate_enhanced_signal(row, side, stock_df, row['is_0dte'])
            if result['signal']:
                rdict = row.to_dict()
                rdict.update(result)
                signals.append(rdict)

        if signals:
            return pd.DataFrame(signals).sort_values('score_percentage', ascending=False)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing options batch: {str(e)}")
        return pd.DataFrame()


def calculate_scanner_score(stock_df: pd.DataFrame, side: str) -> float:
    """High-level technical bias score for calls/puts."""
    if stock_df.empty:
        return 0.0
    latest = stock_df.iloc[-1]
    score = 0.0
    max_score = 5.0

    try:
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        ema_50 = float(latest['EMA_50']) if not pd.isna(latest['EMA_50']) else None
        ema_200 = float(latest['EMA_200']) if not pd.isna(latest['EMA_200']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        macd = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
        macd_sig = float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None

        if side == "call":
            if ema_9 and ema_20 and close > ema_9 > ema_20:
                score += 1.0
            if ema_50 and ema_200 and ema_50 > ema_200:
                score += 1.0
            if rsi and rsi > 50:
                score += 1.0
            if macd and macd_sig and macd > macd_sig:
                score += 1.0
            if vwap and close > vwap:
                score += 1.0
        else:
            if ema_9 and ema_20 and close < ema_9 < ema_20:
                score += 1.0
            if ema_50 and ema_200 and ema_50 < ema_200:
                score += 1.0
            if rsi and rsi < 50:
                score += 1.0
            if macd and macd_sig and macd < macd_sig:
                score += 1.0
            if vwap and close < vwap:
                score += 1.0

        return (score / max_score) * 100
    except Exception as e:
        st.error(f"Error in scanner score calculation: {str(e)}")
        return 0.0


def create_stock_chart(df: pd.DataFrame, sr_levels: dict = None):
    """TradingView-style chart with indicators."""
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

        if 'EMA_9' in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_9'], name='EMA 9'), row=1, col=1)
        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_20'], name='EMA 20'), row=1, col=1)
        if 'KC_upper' in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_upper'], name='KC Upper', line=dict(dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_middle'], name='KC Mid'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['KC_lower'], name='KC Lower', line=dict(dash='dash')), row=1, col=1)

        if 'VWAP' in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['VWAP'], name='VWAP'), row=1, col=1)

        fig.add_trace(go.Bar(x=df['Datetime'], y=df['Volume'], name='Volume'), row=1, col=1, secondary_y=True)

        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI'), row=2, col=1)

        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_signal'], name='Signal'), row=3, col=1)
            fig.add_trace(go.Bar(x=df['Datetime'], y=df['MACD_hist'], name='Hist'), row=3, col=1)

        if sr_levels:
            for lvl in sr_levels.get('5min', {}).get('support', []):
                fig.add_hline(y=lvl, line_dash="dash", line_color="green")
            for lvl in sr_levels.get('5min', {}).get('resistance', []):
                fig.add_hline(y=lvl, line_dash="dash", line_color="red")

        fig.update_layout(
            height=800,
            title='Stock Price Chart with Indicators',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None


# =============================
# PERFORMANCE MONITORING
# =============================

def measure_performance():
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            'start_time': time.time(),
            'api_calls': 0,
            'data_points_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_usage': 0
        }
    try:
        import psutil
        process = psutil.Process()
        st.session_state.performance_metrics['memory_usage'] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    with st.expander("⚡ Performance Metrics", expanded=True):
        elapsed = time.time() - st.session_state.performance_metrics['start_time']
        st.metric("Uptime", f"{elapsed:.1f}s")
        st.metric("API Calls", st.session_state.performance_metrics['api_calls'])
        st.metric("Data Points", st.session_state.performance_metrics['data_points_processed'])
        hits = st.session_state.performance_metrics['cache_hits']
        miss = st.session_state.performance_metrics['cache_misses']
        ratio = hits / max(1, hits + miss) * 100
        st.metric("Cache Hit Ratio", f"{ratio:.1f}%")
        st.metric("Memory Usage", f"{st.session_state.performance_metrics['memory_usage']:.1f} MB")


# =============================
# BACKTESTING (SIMULATED)
# =============================

def run_backtest(signals_df: pd.DataFrame, stock_df: pd.DataFrame, side: str):
    if signals_df.empty or stock_df.empty:
        return None
    try:
        results = []
        for _, row in signals_df.iterrows():
            entry_price = row['ask'] if side == 'call' else row['bid']
            slippage_pct = 0.02
            adj_entry = entry_price * (1 + slippage_pct)
            commission = 0.65
            total_entry_cost = adj_entry + commission

            exit_scenarios = []
            if row['profit_target'] and row['profit_target'] > total_entry_cost:
                exit_scenarios.append(row['profit_target'] - total_entry_cost)
            exit_scenarios.append(row['stop_loss'] - total_entry_cost)
            exit_scenarios.append(-total_entry_cost)

            if len(exit_scenarios) >= 2:
                weights = [0.5, 0.3, 0.2]
                weighted_returns = sum(p * w for p, w in zip(exit_scenarios, weights))
                avg_pnl = weighted_returns
            else:
                avg_pnl = -total_entry_cost

            pnl_pct = (avg_pnl / total_entry_cost) * 100
            results.append({
                'contract': row['contractSymbol'],
                'entry_price': entry_price,
                'adjusted_entry': total_entry_cost,
                'avg_pnl': avg_pnl,
                'pnl_pct': pnl_pct,
                'score': row['score_percentage']
            })

        bt = pd.DataFrame(results)
        if not bt.empty:
            returns = bt['pnl_pct'] / 100
            mean_ret = returns.mean()
            std_ret = returns.std()
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0
            cum = (1 + returns).cumprod()
            run_max = cum.expanding().max()
            drawdown = (cum - run_max) / run_max
            max_dd = drawdown.min()
            gp = bt[bt['avg_pnl'] > 0]['avg_pnl'].sum()
            gl = abs(bt[bt['avg_pnl'] < 0]['avg_pnl'].sum())
            profit_factor = gp / gl if gl != 0 else float('inf')
            win_rate = (bt['avg_pnl'] > 0).mean() * 100

            bt['sharpe_ratio'] = sharpe
            bt['max_drawdown_pct'] = max_dd * 100
            bt['profit_factor'] = profit_factor
            bt['win_rate'] = win_rate
        return bt.sort_values('pnl_pct', ascending=False)
    except Exception as e:
        st.error(f"Error in backtest: {str(e)}")
        return None


# =============================
# STREAMLIT STATE INIT
# =============================

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
if 'force_demo' not in st.session_state:
    st.session_state.force_demo = False

if 'rate_limited_until' in st.session_state:
    if time.time() < st.session_state['rate_limited_until']:
        remaining = int(st.session_state['rate_limited_until'] - time.time())
        st.error(f"⚠️ API rate limited. Please wait {remaining}s.")
        st.stop()
    else:
        del st.session_state['rate_limited_until']


# =============================
# MAIN UI
# =============================

st.title("📈 Enhanced Options Greeks Analyzer")
st.markdown("**Research Tool (NOT 100% accurate)** • Weighted Scoring • Smart Caching • Liquidity Filters")

# ----- Sidebar ----- #
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🔑 API Settings")
    polygon_api_key = st.text_input("Polygon API Key:", type="password", value=CONFIG['POLYGON_API_KEY'])
    if polygon_api_key:
        CONFIG['POLYGON_API_KEY'] = polygon_api_key
        st.success("✅ Polygon API key saved!")
    else:
        st.warning("⚠️ Using free/slow sources only")

    st.subheader("🔑 Free API Keys")
    CONFIG['ALPHA_VANTAGE_API_KEY'] = st.text_input("Alpha Vantage Key:", type="password",
                                                    value=CONFIG['ALPHA_VANTAGE_API_KEY'])
    CONFIG['FMP_API_KEY'] = st.text_input("FMP Key:", type="password",
                                          value=CONFIG['FMP_API_KEY'])
    CONFIG['IEX_API_KEY'] = st.text_input("IEX Cloud Key:", type="password",
                                          value=CONFIG['IEX_API_KEY'])

    with st.expander("💡 How to get free keys"):
        st.markdown("""
        - Alpha Vantage (5 req/min): alphavantage.co
        - FMP (250 req/day): financialmodelingprep.com
        - IEX Cloud: iexcloud.io
        """)

    st.subheader("🔄 Smart Auto-Refresh")
    st.session_state.auto_refresh_enabled = st.checkbox("Enable Auto-Refresh",
                                                        value=st.session_state.auto_refresh_enabled)
    if st.session_state.auto_refresh_enabled:
        refresh_opts = [60, 120, 300, 600]
        selected = st.selectbox(
            "Refresh Interval:",
            options=refresh_opts,
            index=1,
            format_func=lambda x: f"{x}s" if x < 60 else f"{x // 60} min"
        )
        st.session_state.refresh_interval = selected

    with st.expander("📊 Signal Thresholds & Weights"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📈 Calls**")
            SIGNAL_THRESHOLDS['call']['condition_weights']['delta'] = st.slider(
                "Call Delta Weight", 0.1, 0.4, 0.25, 0.05)
            SIGNAL_THRESHOLDS['call']['condition_weights']['gamma'] = st.slider(
                "Call Gamma Weight", 0.1, 0.3, 0.20, 0.05)
            SIGNAL_THRESHOLDS['call']['condition_weights']['trend'] = st.slider(
                "Call Trend Weight", 0.1, 0.3, 0.20, 0.05)
        with col2:
            st.markdown("**📉 Puts**")
            SIGNAL_THRESHOLDS['put']['condition_weights']['delta'] = st.slider(
                "Put Delta Weight", 0.1, 0.4, 0.25, 0.05)
            SIGNAL_THRESHOLDS['put']['condition_weights']['gamma'] = st.slider(
                "Put Gamma Weight", 0.1, 0.3, 0.20, 0.05)
            SIGNAL_THRESHOLDS['put']['condition_weights']['trend'] = st.slider(
                "Put Trend Weight", 0.1, 0.3, 0.20, 0.05)

        st.markdown("---")
        st.markdown("**🎯 Base Thresholds**")
        col1, col2 = st.columns(2)
        with col1:
            SIGNAL_THRESHOLDS['call']['delta_base'] = st.slider("Call Base Delta", 0.1, 1.0, 0.5, 0.1)
            SIGNAL_THRESHOLDS['call']['gamma_base'] = st.slider("Call Base Gamma", 0.01, 0.2, 0.05, 0.01)
            SIGNAL_THRESHOLDS['call']['volume_min'] = st.slider("Call Min Volume", 100, 5000, 1000, 100)
        with col2:
            SIGNAL_THRESHOLDS['put']['delta_base'] = st.slider("Put Base Delta", -1.0, -0.1, -0.5, 0.1)
            SIGNAL_THRESHOLDS['put']['gamma_base'] = st.slider("Put Base Gamma", 0.01, 0.2, 0.05, 0.01)
            SIGNAL_THRESHOLDS['put']['volume_min'] = st.slider("Put Min Volume", 100, 5000, 1000, 100)

    with st.expander("🎯 Risk Management"):
        CONFIG['PROFIT_TARGETS']['call'] = st.slider("Call Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
        CONFIG['PROFIT_TARGETS']['put'] = st.slider("Put Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
        CONFIG['PROFIT_TARGETS']['stop_loss'] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01)

    with st.expander("💰 Liquidity Filters"):
        CONFIG['MIN_OPTION_PRICE'] = st.slider("Min Option Price", 0.05, 5.0, 0.20, 0.05)
        CONFIG['MIN_OPEN_INTEREST'] = st.slider("Min Open Interest", 100, 5000, 1000, 100)
        CONFIG['MIN_VOLUME'] = st.slider("Min Volume", 100, 5000, 500, 100)
        CONFIG['MAX_BID_ASK_SPREAD_PCT'] = st.slider("Max Spread %", 0.05, 1.0, 0.25, 0.05)
        CONFIG['LIQUIDITY_THRESHOLDS'] = {
            'min_open_interest': CONFIG['MIN_OPEN_INTEREST'],
            'min_volume': CONFIG['MIN_VOLUME'],
            'max_bid_ask_spread_pct': CONFIG['MAX_BID_ASK_SPREAD_PCT']
        }

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
        st.caption(f"ET: {now.strftime('%H:%M:%S')}")
    except Exception:
        pass

    measure_performance()

# ----- Header metrics ----- #
ticker = st.text_input("Ticker (e.g. QQQ, IWM, SPY):", value="IWM").upper().strip()

if ticker:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        status_placeholder = st.empty()
    with col2:
        price_placeholder = st.empty()
    with col3:
        cache_placeholder = st.empty()
    with col4:
        refresh_placeholder = st.empty()
    with col5:
        manual_refresh = st.button("🔄 Refresh")

    current_price = get_current_price(ticker)
    cache_age = int(time.time() - st.session_state.get('last_refresh', time.time()))

    if is_market_open():
        status_placeholder.success("🟢 OPEN")
    elif is_premarket():
        status_placeholder.warning("🟡 PRE")
    else:
        status_placeholder.info("🔴 CLOSED")

    if current_price > 0:
        price_placeholder.metric("Price", f"${current_price:.2f}")
    else:
        price_placeholder.error("Price Error")

    cache_placeholder.metric("Cache Age", f"{cache_age}s")
    refresh_placeholder.metric("Refreshes", st.session_state.refresh_counter)

    if manual_refresh:
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.rerun()

    # S/R init (once per ticker)
    if not st.session_state.sr_data or st.session_state.last_ticker != ticker:
        with st.spinner("Analyzing support/resistance levels..."):
            try:
                st.session_state.sr_data = analyze_support_resistance_enhanced(ticker)
                st.session_state.last_ticker = ticker
            except Exception as e:
                st.error(f"S/R error: {str(e)}")
                st.session_state.sr_data = {}

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Enhanced Signals",
        "📊 Technical Analysis",
        "📈 Support/Resistance",
        "🔍 Signal Explanations",
        "📰 Market Context",
        "📊 Free Tier Usage"
    ])

    # ------------- TAB 1: Signals ------------- #
    with tab1:
        try:
            with st.spinner("Loading data & options..."):
                df = get_stock_data_with_indicators(ticker)
                if df.empty:
                    st.error("Unable to fetch stock data. Check ticker or rate limits.")
                    st.stop()

                current_price = df.iloc[-1]['Close']
                st.success(f"{ticker} — ${current_price:.2f}")

                atr_pct = df.iloc[-1].get('ATR_pct', 0)
                if not pd.isna(atr_pct):
                    if atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['high']:
                        vol_tag = "🌪️ Extreme"
                    elif atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['medium']:
                        vol_tag = "🟡 High"
                    elif atr_pct > CONFIG['VOLATILITY_THRESHOLDS']['low']:
                        vol_tag = "🟠 Medium"
                    else:
                        vol_tag = "🟢 Low"
                    st.info(f"{vol_tag} volatility: {atr_pct*100:.2f}% ATR%")

                expiries, all_calls, all_puts = get_full_options_chain(ticker)

            show_demo = False  # default

            if not expiries and not st.session_state.force_demo:
                st.error("Unable to fetch real options data.")

                rate_limited = False
                remaining_time = 0
                if 'yf_rate_limited_until' in st.session_state:
                    remaining_time = max(0, int(st.session_state['yf_rate_limited_until'] - time.time()))
                    rate_limited = remaining_time > 0

                with st.expander("💡 Real Data Tips", expanded=True):
                    st.markdown("""
                    - Wait 2–5 minutes and try again (rate limits).
                    - Use liquid tickers like SPY, QQQ, IWM, AAPL.
                    - Use during regular market hours for best availability.
                    """)
                    if rate_limited:
                        st.warning(f"Rate limited for ~{remaining_time} more seconds.")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("Clear Rate Limit"):
                            clear_rate_limit()
                    with col_b:
                        if st.button("Force Retry Now"):
                            if 'yf_rate_limited_until' in st.session_state:
                                del st.session_state['yf_rate_limited_until']
                            st.cache_data.clear()
                            st.rerun()
                    with col_c:
                        show_demo = st.button("Use Demo Data")

                if show_demo:
                    st.session_state.force_demo = True
                    expiries, all_calls, all_puts = get_fallback_options_data(ticker)
                else:
                    st.stop()

            if st.session_state.force_demo:
                st.warning("Using DEMO options data (for interface testing).")
            else:
                st.success(f"Real options data loaded: {len(all_calls)} calls, {len(all_puts)} puts")

            col_exp_a, col_exp_b = st.columns(2)
            with col_exp_a:
                mode = st.radio(
                    "Expiration Filter",
                    ["0DTE Only", "This Week", "All Near-Term"],
                    index=1
                )
            today = datetime.date.today()
            if mode == "0DTE Only":
                expiries_to_use = [e for e in expiries
                                   if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
            elif mode == "This Week":
                week_end = today + datetime.timedelta(days=7)
                expiries_to_use = [e for e in expiries
                                   if today <= datetime.datetime.strptime(e, "%Y-%m-%d").date() <= week_end]
            else:
                expiries_to_use = expiries[:5]

            if not expiries_to_use:
                st.warning(f"No expiries available for '{mode}'.")
                st.stop()

            with col_exp_b:
                st.info(f"Analyzing {len(expiries_to_use)} expiries")

            calls_filtered = all_calls[all_calls['expiry'].isin(expiries_to_use)].copy()
            puts_filtered = all_puts[all_puts['expiry'].isin(expiries_to_use)].copy()

            strike_range = st.slider(
                "Strike range around current price ($):",
                -50, 50, (-10, 10), 1
            )
            min_strike = current_price + strike_range[0]
            max_strike = current_price + strike_range[1]

            calls_filtered = calls_filtered[
                (calls_filtered['strike'] >= min_strike) & (calls_filtered['strike'] <= max_strike)
            ].copy()
            puts_filtered = puts_filtered[
                (puts_filtered['strike'] >= min_strike) & (puts_filtered['strike'] <= max_strike)
            ].copy()

            m_filter = st.multiselect(
                "Moneyness filter:",
                options=["ITM", "NTM", "ATM", "OTM"],
                default=["NTM", "ATM"]
            )

            if not calls_filtered.empty:
                calls_filtered['moneyness'] = calls_filtered['strike'].apply(
                    lambda s: classify_moneyness(s, current_price)
                )
                calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]

            if not puts_filtered.empty:
                puts_filtered['moneyness'] = puts_filtered['strike'].apply(
                    lambda s: classify_moneyness(s, current_price)
                )
                puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]

            st.write(f"Filtered: {len(calls_filtered)} calls, {len(puts_filtered)} puts")

            col_c, col_p = st.columns(2)
            # Calls
            with col_c:
                st.subheader("📈 Call Signals")
                if not calls_filtered.empty:
                    call_signals = process_options_batch(calls_filtered, "call", df, current_price)
                    if not call_signals.empty:
                        display_cols = [
                            'contractSymbol', 'strike', 'lastPrice', 'volume',
                            'delta', 'gamma', 'theta', 'moneyness',
                            'score_percentage', 'profit_target', 'stop_loss',
                            'holding_period', 'is_0dte'
                        ]
                        display_cols = [c for c in display_cols if c in call_signals.columns]
                        display_df = call_signals[display_cols].rename(columns={
                            'score_percentage': 'Score%',
                            'profit_target': 'Target',
                            'stop_loss': 'Stop',
                            'holding_period': 'Hold Period',
                            'is_0dte': '0DTE'
                        })
                        st.dataframe(display_df.round(3), use_container_width=True, hide_index=True)
                        avg_score = call_signals['score_percentage'].mean()
                        top_score = call_signals['score_percentage'].max()
                        st.success(f"{len(call_signals)} call signals | Avg: {avg_score:.1f}% | Best: {top_score:.1f}%")

                        best_call = call_signals.iloc[0]
                        with st.expander(f"🏆 Top Call: {best_call['contractSymbol']}"):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Score", f"{best_call['score_percentage']:.1f}%")
                                st.metric("Delta", f"{best_call['delta']:.3f}")
                                st.metric("OI", f"{best_call['open_interest']}")
                            with c2:
                                st.metric("Target", f"${best_call['profit_target']:.2f}")
                                st.metric("Gamma", f"{best_call['gamma']:.3f}")
                                st.metric("Volume", f"{best_call['volume']}")
                            with c3:
                                st.metric("Stop", f"${best_call['stop_loss']:.2f}")
                                st.metric("IV", f"{best_call['implied_volatility']*100:.1f}%")
                                st.metric("Hold", best_call['holding_period'])

                        with st.expander("🔬 Backtest (Simulated)", expanded=False):
                            bt = run_backtest(call_signals, df, 'call')
                            if bt is not None and not bt.empty:
                                st.dataframe(bt)
                                st.metric("Avg P&L", f"{bt['pnl_pct'].mean():.1f}%")
                                st.metric("Win Rate", f"{(bt['avg_pnl'] > 0).mean()*100:.1f}%")
                                st.metric("Sharpe", f"{bt['sharpe_ratio'].iloc[0]:.2f}")
                                st.metric("Max DD", f"{bt['max_drawdown_pct'].iloc[0]:.2f}%")
                            else:
                                st.info("No backtest results")
                    else:
                        st.info("No call signals for current filters.")
                else:
                    st.info("No call options with current filters.")

            # Puts
            with col_p:
                st.subheader("📉 Put Signals")
                if not puts_filtered.empty:
                    put_signals = process_options_batch(puts_filtered, "put", df, current_price)
                    if not put_signals.empty:
                        display_cols = [
                            'contractSymbol', 'strike', 'lastPrice', 'volume',
                            'delta', 'gamma', 'theta', 'moneyness',
                            'score_percentage', 'profit_target', 'stop_loss',
                            'holding_period', 'is_0dte'
                        ]
                        display_cols = [c for c in display_cols if c in put_signals.columns]
                        display_df = put_signals[display_cols].rename(columns={
                            'score_percentage': 'Score%',
                            'profit_target': 'Target',
                            'stop_loss': 'Stop',
                            'holding_period': 'Hold Period',
                            'is_0dte': '0DTE'
                        })
                        st.dataframe(display_df.round(3), use_container_width=True, hide_index=True)
                        avg_score = put_signals['score_percentage'].mean()
                        top_score = put_signals['score_percentage'].max()
                        st.success(f"{len(put_signals)} put signals | Avg: {avg_score:.1f}% | Best: {top_score:.1f}%")

                        best_put = put_signals.iloc[0]
                        with st.expander(f"🏆 Top Put: {best_put['contractSymbol']}"):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Score", f"{best_put['score_percentage']:.1f}%")
                                st.metric("Delta", f"{best_put['delta']:.3f}")
                                st.metric("OI", f"{best_put['open_interest']}")
                            with c2:
                                st.metric("Target", f"${best_put['profit_target']:.2f}")
                                st.metric("Gamma", f"{best_put['gamma']:.3f}")
                                st.metric("Volume", f"{best_put['volume']}")
                            with c3:
                                st.metric("Stop", f"${best_put['stop_loss']:.2f}")
                                st.metric("IV", f"{best_put['implied_volatility']*100:.1f}%")
                                st.metric("Hold", best_put['holding_period'])

                        with st.expander("🔬 Backtest (Simulated)", expanded=False):
                            bt = run_backtest(put_signals, df, 'put')
                            if bt is not None and not bt.empty:
                                st.dataframe(bt)
                                st.metric("Avg P&L", f"{bt['pnl_pct'].mean():.1f}%")
                                st.metric("Win Rate", f"{(bt['avg_pnl'] > 0).mean()*100:.1f}%")
                                st.metric("Sharpe", f"{bt['sharpe_ratio'].iloc[0]:.2f}")
                                st.metric("Max DD", f"{bt['max_drawdown_pct'].iloc[0]:.2f}%")
                            else:
                                st.info("No backtest results")
                    else:
                        st.info("No put signals for current filters.")
                else:
                    st.info("No put options with current filters.")

            # Greeks heatmap
            with st.expander("📊 Greeks Heatmap", expanded=False):
                import plotly.express as px
                combined = pd.concat([
                    calls_filtered.assign(type='Call'),
                    puts_filtered.assign(type='Put')
                ]) if (not calls_filtered.empty or not puts_filtered.empty) else pd.DataFrame()
                if not combined.empty:
                    fig = px.density_heatmap(
                        combined, x='strike', y='expiry', z='delta',
                        facet_col='type', color_continuous_scale='RdBu',
                        title='Delta Heatmap Across Strikes & Expiries'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data for heatmap.")

            call_score = calculate_scanner_score(df, 'call')
            put_score = calculate_scanner_score(df, 'put')
            st.markdown("---")
            st.subheader("🧠 Technical Scanner Scores")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Call Scanner", f"{call_score:.1f}%")
            with c2:
                st.metric("Put Scanner", f"{put_score:.1f}%")
            with c3:
                bias = "Bullish" if call_score > put_score else "Bearish" if put_score > call_score else "Neutral"
                diff = abs(call_score - put_score)
                st.metric("Directional Bias", bias)
                st.caption(f"Strength: {diff:.1f}% difference")
        except Exception as e:
            st.error(f"Error in signal analysis: {str(e)}")

    # ------------- TAB 2: Technical ------------- #
    with tab2:
        try:
            if 'df' not in locals():
                df = get_stock_data_with_indicators(ticker)
            if df.empty:
                st.info("No data.")
            else:
                st.subheader("📊 Technical Dashboard")
                latest = df.iloc[-1]
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    st.metric("Price", f"${latest['Close']:.2f}")
                with c2:
                    if not pd.isna(latest['EMA_9']):
                        st.metric("EMA 9", f"${latest['EMA_9']:.2f}")
                with c3:
                    if not pd.isna(latest['EMA_20']):
                        st.metric("EMA 20", f"${latest['EMA_20']:.2f}")
                with c4:
                    if not pd.isna(latest['RSI']):
                        st.metric("RSI", f"{latest['RSI']:.1f}")
                with c5:
                    if not pd.isna(latest['ATR_pct']):
                        st.metric("ATR%", f"{latest['ATR_pct']*100:.2f}%")
                with c6:
                    if not pd.isna(latest['avg_vol']):
                        ratio = latest['Volume'] / latest['avg_vol']
                        st.metric("Volume Ratio", f"{ratio:.1f}x")

                st.subheader("📋 Recent Candles")
                display = df.tail(10)[['Datetime', 'Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'ATR_pct', 'Volume', 'avg_vol']].copy()
                display['ATR_pct'] = display['ATR_pct'] * 100
                display['Volume Ratio'] = display['Volume'] / display['avg_vol']
                display['Time'] = display['Datetime'].dt.strftime('%H:%M')
                final_cols = ['Time', 'Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'ATR_pct', 'Volume Ratio']
                st.dataframe(display[final_cols].rename(columns={'ATR_pct': 'ATR%'}).round(2),
                             use_container_width=True, hide_index=True)

                st.subheader("📈 Chart")
                chart = create_stock_chart(df, st.session_state.sr_data)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Technical tab error: {str(e)}")

    # ------------- TAB 3: S/R ------------- #
    with tab3:
        st.subheader("📈 Multi-Timeframe Support/Resistance")
        if not st.session_state.sr_data:
            st.warning("No S/R data. Try refreshing.")
        else:
            fig = plot_sr_levels_enhanced(st.session_state.sr_data, current_price)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Key Levels (5m / 15m / 30m / 1h)")
            for tf in ['5min', '15min', '30min', '1h']:
                if tf in st.session_state.sr_data:
                    sr = st.session_state.sr_data[tf]
                    st.markdown(f"**{tf}**")
                    st.write("Support:")
                    for lvl in sr['support'][:3]:
                        dist = abs(lvl - current_price) / current_price * 100
                        st.write(f"- {lvl:.2f} ({dist:.1f}% away)")
                    st.write("Resistance:")
                    for lvl in sr['resistance'][:3]:
                        dist = abs(lvl - current_price) / current_price * 100
                        st.write(f"- {lvl:.2f} ({dist:.1f}% away)")
        st.info("Use S/R + VWAP + your levels to align call/put entries with clean moves.")

    # ------------- TAB 4: Explanations ------------- #
    with tab4:
        st.subheader("🔍 Signal Methodology")
        st.markdown("""
        - This app scores each contract based on: Delta, Gamma, Theta, Trend (EMAs), RSI, Volume, and VWAP.
        - Scores are normalized to a 0–100 scale based on your configured weights.
        - A contract only becomes a **signal** if ALL core conditions pass.
        - VWAP adds extra weight but is not required if you want to relax it.
        """)
        st.markdown("**Important:** This is NOT a holy grail. It helps you **filter** and **rank** setups.")

    # ------------- TAB 5: Market Context ------------- #
    with tab5:
        st.subheader("📰 Market Context")
        try:
            stock = yf.Ticker(ticker)
            with st.expander("🏢 Company Overview", expanded=True):
                try:
                    info = stock.info
                    if info:
                        st.write(f"**Name:** {info.get('longName', 'N/A')}")
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                except Exception:
                    st.write("No company info")
            with st.expander("📰 Recent News", expanded=False):
                try:
                    news = stock.news
                    if news:
                        for n in news[:5]:
                            st.markdown(f"**{n.get('title', 'No Title')}**")
                            st.write(n.get('publisher', 'Unknown'))
                            link = n.get('link', '')
                            if link:
                                st.markdown(f"[Read]({link})")
                            st.markdown("---")
                    else:
                        st.write("No recent news.")
                except Exception:
                    st.write("News unavailable.")
        except Exception as e:
            st.error(f"Context error: {str(e)}")

    # ------------- TAB 6: API Usage ------------- #
    with tab6:
        st.subheader("📊 Free Tier Usage")
        if not st.session_state.API_CALL_LOG:
            st.info("No API calls logged yet.")
        else:
            now = time.time()
            av_1min = len([t for t in st.session_state.API_CALL_LOG
                           if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 60])
            av_1hr = len([t for t in st.session_state.API_CALL_LOG
                          if t['source'] == "ALPHA_VANTAGE" and now - t['timestamp'] < 3600])

            fmp_1hr = len([t for t in st.session_state.API_CALL_LOG
                           if t['source'] == "FMP" and now - t['timestamp'] < 3600])
            fmp_24h = len([t for t in st.session_state.API_CALL_LOG
                           if t['source'] == "FMP" and now - t['timestamp'] < 86400])

            iex_1hr = len([t for t in st.session_state.API_CALL_LOG
                           if t['source'] == "IEX" and now - t['timestamp'] < 3600])
            iex_24h = len([t for t in st.session_state.API_CALL_LOG
                           if t['source'] == "IEX" and now - t['timestamp'] < 86400])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("AV last min", f"{av_1min}/5")
                st.metric("AV last hour", f"{av_1hr}/300")
            with c2:
                st.metric("FMP last hour", f"{fmp_1hr}/10")
                st.metric("FMP last 24h", f"{fmp_24h}/250")
            with c3:
                st.metric("IEX last hour", f"{iex_1hr}/69")
                st.metric("IEX last 24h", f"{iex_24h}/1667")

else:
    st.info("Enter a ticker to start scanning options plays (calls & puts).")

# ----- Auto-Refresh Engine ----- #
if st.session_state.get('auto_refresh_enabled', False) and ticker:
    now = time.time()
    elapsed = now - st.session_state.last_refresh
    min_interval = max(st.session_state.refresh_interval, CONFIG['MIN_REFRESH_INTERVAL'])
    if elapsed > min_interval:
        st.session_state.last_refresh = now
        st.session_state.refresh_counter += 1
        st.cache_data.clear()
        st.success(f"🔄 Auto-refreshed at {datetime.datetime.now().strftime('%H:%M:%S')}")
        time.sleep(0.5)
        st.rerun()
