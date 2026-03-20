import datetime
import math
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import pytz

from plotly.subplots import make_subplots
from scipy import signal
from streamlit_autorefresh import st_autorefresh
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, KeltnerChannel

try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except Exception:
    POLYGON_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 1,
    "DATA_TIMEOUT": 10,
    "MIN_DATA_POINTS": 50,
    "CACHE_TTL": 300,
    "STOCK_CACHE_TTL": 300,
    "OPTIONS_CACHE_TTL": 1800,
    "RATE_LIMIT_COOLDOWN": 180,
    "MIN_REFRESH_INTERVAL": 60,
    "MARKET_OPEN": datetime.time(9, 30),
    "MARKET_CLOSE": datetime.time(16, 0),
    "PREMARKET_START": datetime.time(4, 0),
    "VOLATILITY_THRESHOLDS": {
        "low": 0.015,
        "medium": 0.03,
        "high": 0.05,
    },
    "PROFIT_TARGETS": {
        "call": 0.15,
        "put": 0.15,
        "stop_loss": 0.08,
    },
    "TRADING_HOURS_PER_DAY": 6.5,
    "SR_LEVEL_SENSITIVITY": {
        "5min": 0.003,
        "15min": 0.004,
        "30min": 0.005,
        "1h": 0.006,
        "2h": 0.007,
        "4h": 0.008,
        "daily": 0.010,
    },
    "SR_SENSITIVITY": {
        "SR_WINDOW_SIZES": {
            "5min": 3,
            "15min": 5,
            "30min": 7,
            "1h": 10,
            "2h": 12,
            "4h": 6,
            "daily": 20,
        },
        "LIQUIDITY_THRESHOLDS": {
            "min_open_interest": 100,
            "min_volume": 100,
            "max_bid_ask_spread_pct": 0.10,
        },
    },
}

SIGNAL_THRESHOLDS = {
    "call": {
        "delta_base": 0.50,
        "gamma_base": 0.05,
        "theta_base": 0.05,
        "rsi_min": 50,
        "rsi_max": 50,
        "volume_min": 1000,
        "condition_weights": {
            "delta": 0.25,
            "gamma": 0.20,
            "theta": 0.15,
            "trend": 0.20,
            "momentum": 0.10,
            "volume": 0.10,
        },
    },
    "put": {
        "delta_base": -0.50,
        "gamma_base": 0.05,
        "theta_base": 0.05,
        "rsi_min": 50,
        "rsi_max": 50,
        "volume_min": 1000,
        "condition_weights": {
            "delta": 0.25,
            "gamma": 0.20,
            "theta": 0.15,
            "trend": 0.20,
            "momentum": 0.10,
            "volume": 0.10,
        },
    },
}


# =========================================================
# SESSION STATE INIT
# =========================================================
def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "API_CALL_LOG": [],
        "refresh_counter": 0,
        "last_refresh": time.time(),
        "refresh_interval": CONFIG["MIN_REFRESH_INTERVAL"],
        "auto_refresh_enabled": False,
        "sr_data": {},
        "last_ticker": "",
        "current_timeframe": "5m",
        "force_demo": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================================================
# STYLING
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
            div[data-stale="true"] { opacity: 1.0 !important; }
            .main { background-color: #131722; color: #d1d4dc; }
            section[data-testid="stSidebar"] { background-color: #1e222d; }
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
                background-color: #1e222d;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 20px;
                font-weight: bold;
                background-color: #1e222d;
                border-radius: 4px;
                color: #2962ff;
            }
            .stTabs [aria-selected="true"] {
                background-color: #2962ff;
                color: white;
            }
            .stButton button {
                background-color: #2962ff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            .stButton button:hover {
                background-color: #1e53e5;
                color: white;
            }
            .stTextInput input, .stSelectbox select {
                background-color: #1e222d;
                color: #d1d4dc;
                border: 1px solid #2a2e39;
            }
            [data-testid="stMetricValue"] {
                color: #d1d4dc;
                font-weight: bold;
            }
            [data-testid="stMetricLabel"] {
                color: #758696;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# AUTH
# =========================================================
def authenticate_user() -> None:
    app_username = st.secrets.get("APP_USERNAME", "")
    app_password = st.secrets.get("APP_PASSWORD", "")

    if st.session_state.authenticated:
        return

    st.title("🔒 Login to Options Analyzer Pro")
    st.markdown("Enter your credentials to access the application.")

    username = st.text_input("Username", value="")
    password = st.text_input("Password", type="password", value="")

    if st.button("Login"):
        if username == app_username and password == app_password and app_username and app_password:
            st.session_state.authenticated = True
            st.success("✅ Logged in successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

    st.stop()


# =========================================================
# TIME HELPERS
# =========================================================
def eastern_now() -> datetime.datetime:
    return datetime.datetime.now(pytz.timezone("US/Eastern"))


def is_market_open() -> bool:
    now = eastern_now()
    if now.weekday() >= 5:
        return False
    return CONFIG["MARKET_OPEN"] <= now.time() <= CONFIG["MARKET_CLOSE"]


def is_premarket() -> bool:
    now = eastern_now()
    if now.weekday() >= 5:
        return False
    return CONFIG["PREMARKET_START"] <= now.time() < CONFIG["MARKET_OPEN"]


def is_early_market() -> bool:
    if not is_market_open():
        return False
    now = eastern_now()
    market_open_dt = now.replace(
        hour=CONFIG["MARKET_OPEN"].hour,
        minute=CONFIG["MARKET_OPEN"].minute,
        second=0,
        microsecond=0,
    )
    return (now - market_open_dt).total_seconds() < 1800


def calculate_remaining_trading_hours() -> float:
    now = eastern_now()
    close_dt = now.replace(
        hour=CONFIG["MARKET_CLOSE"].hour,
        minute=CONFIG["MARKET_CLOSE"].minute,
        second=0,
        microsecond=0,
    )
    if now >= close_dt:
        return 0.0
    return max(0.0, (close_dt - now).total_seconds() / 3600.0)


# =========================================================
# API LOGGING
# =========================================================
def can_make_request(source: str) -> bool:
    now = time.time()
    st.session_state.API_CALL_LOG = [
        t for t in st.session_state.API_CALL_LOG if now - t["timestamp"] < 3600
    ]

    av_count = len(
        [t for t in st.session_state.API_CALL_LOG if t["source"] == "ALPHA_VANTAGE" and now - t["timestamp"] < 60]
    )
    fmp_count = len(
        [t for t in st.session_state.API_CALL_LOG if t["source"] == "FMP" and now - t["timestamp"] < 3600]
    )
    iex_count = len(
        [t for t in st.session_state.API_CALL_LOG if t["source"] == "IEX" and now - t["timestamp"] < 3600]
    )

    if source == "ALPHA_VANTAGE" and av_count >= 4:
        return False
    if source == "FMP" and fmp_count >= 9:
        return False
    if source == "IEX" and iex_count >= 29:
        return False
    return True


def log_api_request(source: str) -> None:
    st.session_state.API_CALL_LOG.append({"source": source, "timestamp": time.time()})


# =========================================================
# CURRENT PRICE
# =========================================================
@st.cache_data(ttl=5, show_spinner=False)
def get_current_price(ticker: str) -> float:
    polygon_api_key = st.secrets.get("POLYGON_API_KEY", "")
    alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    iex_key = st.secrets.get("IEX_API_KEY", "")

    if polygon_api_key and POLYGON_AVAILABLE:
        try:
            client = RESTClient(polygon_api_key, trace=False)
            trade = client.get_last_trade(ticker)
            if hasattr(trade, "price"):
                return float(trade.price)
        except Exception:
            pass

    if alpha_vantage_key and can_make_request("ALPHA_VANTAGE"):
        try:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={alpha_vantage_key}"
            )
            r = requests.get(url, timeout=2)
            r.raise_for_status()
            data = r.json()
            price = data.get("Global Quote", {}).get("05. price")
            if price:
                log_api_request("ALPHA_VANTAGE")
                return float(price)
        except Exception:
            pass

    if fmp_key and can_make_request("FMP"):
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote-short/{ticker}?apikey={fmp_key}"
            r = requests.get(url, timeout=2)
            r.raise_for_status()
            data = r.json()
            if data and isinstance(data, list) and "price" in data[0]:
                log_api_request("FMP")
                return float(data[0]["price"])
        except Exception:
            pass

    if iex_key and can_make_request("IEX"):
        try:
            url = f"https://cloud.iexapis.com/stable/stock/{ticker}/quote?token={iex_key}"
            r = requests.get(url, timeout=2)
            r.raise_for_status()
            data = r.json()
            if "latestPrice" in data:
                log_api_request("IEX")
                return float(data["latestPrice"])
        except Exception:
            pass

    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass

    return 0.0


# =========================================================
# DATA HELPERS
# =========================================================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Datetime" not in df.columns:
        candidates = ["Date", "date", "index", "time", "Time"]
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            df = df.rename(columns={found: "Datetime"})
    if "Datetime" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    return df


# =========================================================
# INDICATORS
# =========================================================
def calculate_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_vol"] = np.nan

    try:
        for date_, group in df.groupby(df["Datetime"].dt.date):
            regular = group[~group["premarket"]]
            if not regular.empty:
                df.loc[regular.index, "avg_vol"] = regular["Volume"].expanding(min_periods=1).mean()

            pre = group[group["premarket"]]
            if not pre.empty:
                df.loc[pre.index, "avg_vol"] = pre["Volume"].expanding(min_periods=1).mean()

        df["avg_vol"] = df["avg_vol"].fillna(df["Volume"].mean())
    except Exception:
        df["avg_vol"] = df["Volume"].mean()

    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)

    if df.empty:
        return df

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    for period in [9, 20, 50, 200]:
        if len(df) >= period:
            df[f"EMA_{period}"] = EMAIndicator(close=close, window=period).ema_indicator()
        else:
            df[f"EMA_{period}"] = np.nan

    if len(df) >= 14:
        df["RSI"] = RSIIndicator(close=close, window=14).rsi()
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["ATR"] = atr.average_true_range()
        df["ATR_pct"] = df["ATR"] / close.replace(0, np.nan)
    else:
        df["RSI"] = np.nan
        df["ATR"] = np.nan
        df["ATR_pct"] = np.nan

    if len(df) >= 26:
        macd = MACD(close=close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()

        kc = KeltnerChannel(high=high, low=low, close=close)
        df["KC_upper"] = kc.keltner_channel_hband()
        df["KC_middle"] = kc.keltner_channel_mband()
        df["KC_lower"] = kc.keltner_channel_lband()
    else:
        for col in ["MACD", "MACD_signal", "MACD_hist", "KC_upper", "KC_middle", "KC_lower"]:
            df[col] = np.nan

    df["VWAP"] = np.nan
    for _, group in df.groupby(df["Datetime"].dt.date):
        regular = group[~group["premarket"]]
        if not regular.empty:
            tp = (regular["High"] + regular["Low"] + regular["Close"]) / 3
            num = (tp * regular["Volume"]).cumsum()
            den = regular["Volume"].cumsum().replace(0, np.nan)
            df.loc[regular.index, "VWAP"] = num / den

        pre = group[group["premarket"]]
        if not pre.empty:
            tp = (pre["High"] + pre["Low"] + pre["Close"]) / 3
            num = (tp * pre["Volume"]).cumsum()
            den = pre["Volume"].cumsum().replace(0, np.nan)
            df.loc[pre.index, "VWAP"] = num / den

    df = calculate_volume_averages(df)
    return df


@st.cache_data(ttl=CONFIG["STOCK_CACHE_TTL"], show_spinner=False)
def get_stock_data_with_indicators(ticker: str) -> pd.DataFrame:
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=10)

        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="5m",
            auto_adjust=True,
            progress=False,
            prepost=True,
        )

        if df.empty:
            return pd.DataFrame()

        df = flatten_columns(df)
        df = df.reset_index()
        df = ensure_datetime_column(df)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return pd.DataFrame()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required)

        if len(df) < CONFIG["MIN_DATA_POINTS"]:
            return pd.DataFrame()

        eastern = pytz.timezone("US/Eastern")
        if df["Datetime"].dt.tz is None:
            df["Datetime"] = df["Datetime"].dt.tz_localize("UTC")
        df["Datetime"] = df["Datetime"].dt.tz_convert(eastern)

        df["premarket"] = (
            (df["Datetime"].dt.time >= CONFIG["PREMARKET_START"])
            & (df["Datetime"].dt.time < CONFIG["MARKET_OPEN"])
        )

        return compute_all_indicators(df)

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================
def find_peaks_valleys_robust(data: np.ndarray, order: int = 5, prominence: Optional[float] = None) -> Tuple[List[int], List[int]]:
    if len(data) < order * 2 + 1:
        return [], []

    try:
        if prominence is not None:
            peaks, _ = signal.find_peaks(data, distance=order, prominence=prominence)
            valleys, _ = signal.find_peaks(-data, distance=order, prominence=prominence)
            return peaks.tolist(), valleys.tolist()

        peaks: List[int] = []
        valleys: List[int] = []
        for i in range(order, len(data) - order):
            if all(data[i] > data[i - j] and data[i] > data[i + j] for j in range(1, order + 1)):
                peaks.append(i)
            if all(data[i] < data[i - j] and data[i] < data[i + j] for j in range(1, order + 1)):
                valleys.append(i)
        return peaks, valleys
    except Exception:
        return [], []


def calculate_dynamic_sensitivity(data: pd.DataFrame, base_sensitivity: float) -> float:
    try:
        if data.empty or len(data) < 10:
            return base_sensitivity

        current_price = float(data["Close"].iloc[-1])
        if current_price <= 0 or np.isnan(current_price):
            return base_sensitivity

        tr1 = data["High"] - data["Low"]
        tr2 = (data["High"] - data["Close"].shift(1)).abs()
        tr3 = (data["Low"] - data["Close"].shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=min(14, len(data))).mean().iloc[-1]

        if pd.notna(atr) and atr > 0:
            volatility_ratio = atr / current_price
            dynamic = base_sensitivity * (1 + volatility_ratio * 2)
            return min(max(dynamic, base_sensitivity * 0.5), base_sensitivity * 3)

        return base_sensitivity
    except Exception:
        return base_sensitivity


def cluster_levels_improved(levels: List[float], current_price: float, sensitivity: float, level_type: str) -> List[Dict]:
    if not levels:
        return []

    levels = sorted(levels)
    clustered: List[Dict] = []
    current_cluster: List[float] = []

    for level in levels:
        if not current_cluster:
            current_cluster.append(level)
            continue

        cluster_center = np.mean(current_cluster)
        distance_ratio = abs(level - cluster_center) / max(current_price, 1e-9)

        if distance_ratio <= sensitivity:
            current_cluster.append(level)
        else:
            cluster_price = float(np.mean(current_cluster))
            clustered.append({
                "price": cluster_price,
                "strength": len(current_cluster),
                "distance": abs(cluster_price - current_price) / max(current_price, 1e-9),
                "type": level_type,
                "raw_levels": current_cluster.copy(),
            })
            current_cluster = [level]

    if current_cluster:
        cluster_price = float(np.mean(current_cluster))
        clustered.append({
            "price": cluster_price,
            "strength": len(current_cluster),
            "distance": abs(cluster_price - current_price) / max(current_price, 1e-9),
            "type": level_type,
            "raw_levels": current_cluster.copy(),
        })

    clustered.sort(key=lambda x: (-x["strength"], x["distance"]))
    return clustered[:3]


def calculate_support_resistance_enhanced(data: pd.DataFrame, timeframe: str, current_price: float) -> Dict:
    if data.empty or len(data) < 20:
        return {
            "support": [],
            "resistance": [],
            "sensitivity": CONFIG["SR_LEVEL_SENSITIVITY"].get(timeframe, 0.005),
            "timeframe": timeframe,
            "data_points": len(data),
        }

    try:
        base_sensitivity = CONFIG["SR_LEVEL_SENSITIVITY"].get(timeframe, 0.005)
        window_size = CONFIG["SR_SENSITIVITY"]["SR_WINDOW_SIZES"].get(timeframe, 5)
        dynamic_sensitivity = calculate_dynamic_sensitivity(data, base_sensitivity)

        highs = data["High"].values
        lows = data["Low"].values
        closes = data["Close"].values

        prominence = max(np.std(closes) * 0.5, 0.01)

        resistance_idx1, support_idx1 = find_peaks_valleys_robust(highs, order=window_size, prominence=prominence)
        support_idx2, resistance_idx2 = find_peaks_valleys_robust(lows, order=window_size, prominence=prominence)

        all_resistance_indices = list(set(resistance_idx1 + resistance_idx2))
        all_support_indices = list(set(support_idx1 + support_idx2))

        resistance_levels = [float(highs[i]) for i in all_resistance_indices if i < len(highs)]
        support_levels = [float(lows[i]) for i in all_support_indices if i < len(lows)]

        close_peaks, close_valleys = find_peaks_valleys_robust(closes, order=max(3, window_size - 2))
        resistance_levels.extend([float(closes[i]) for i in close_peaks])
        support_levels.extend([float(closes[i]) for i in close_valleys])

        if "VWAP" in data.columns and pd.notna(data["VWAP"].iloc[-1]):
            vwap_value = float(data["VWAP"].iloc[-1])
            support_levels.append(vwap_value)
            resistance_levels.append(vwap_value)
        else:
            vwap_value = np.nan

        min_distance = current_price * 0.001
        resistance_levels = [x for x in set(resistance_levels) if abs(x - current_price) > min_distance and x > current_price]
        support_levels = [x for x in set(support_levels) if abs(x - current_price) > min_distance and x < current_price]

        clustered_resistance = cluster_levels_improved(resistance_levels, current_price, dynamic_sensitivity, "resistance")
        clustered_support = cluster_levels_improved(support_levels, current_price, dynamic_sensitivity, "support")

        return {
            "support": [x["price"] for x in clustered_support],
            "resistance": [x["price"] for x in clustered_resistance],
            "vwap": vwap_value,
            "sensitivity": dynamic_sensitivity,
            "timeframe": timeframe,
            "data_points": len(data),
            "support_details": clustered_support,
            "resistance_details": clustered_resistance,
            "stats": {
                "raw_support_count": len(support_levels),
                "raw_resistance_count": len(resistance_levels),
                "clustered_support_count": len(clustered_support),
                "clustered_resistance_count": len(clustered_resistance),
            },
        }
    except Exception as e:
        return {
            "support": [],
            "resistance": [],
            "sensitivity": CONFIG["SR_LEVEL_SENSITIVITY"].get(timeframe, 0.005),
            "timeframe": timeframe,
            "data_points": len(data),
            "error": str(e),
        }


@st.cache_data(ttl=300, show_spinner=False)
def get_multi_timeframe_data_enhanced(ticker: str) -> Tuple[Dict[str, pd.DataFrame], float]:
    timeframes = {
        "5min": {"interval": "5m", "period": "5d"},
        "15min": {"interval": "15m", "period": "15d"},
        "30min": {"interval": "30m", "period": "30d"},
        "1h": {"interval": "60m", "period": "60d"},
        "2h": {"interval": "60m", "period": "90d", "resample": "2H"},
        "4h": {"interval": "60m", "period": "180d", "resample": "4H"},
        "daily": {"interval": "1d", "period": "1y"},
    }

    data: Dict[str, pd.DataFrame] = {}
    current_price: Optional[float] = None

    for tf, params in timeframes.items():
        try:
            df = yf.download(
                ticker,
                period=params["period"],
                interval=params["interval"],
                progress=False,
                prepost=True,
                auto_adjust=True,
            )

            if df.empty:
                continue

            df = flatten_columns(df).dropna()

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in required_cols):
                continue

            df = df[df["High"] >= df["Low"]]
            df = df[df["Volume"] >= 0]

            if "resample" in params:
                df.index = pd.to_datetime(df.index)
                df = (
                    df.resample(params["resample"])
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna()
                )

            if len(df) < 20:
                continue

            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            denom = df["Volume"].cumsum().replace(0, np.nan)
            df["VWAP"] = (typical_price * df["Volume"]).cumsum() / denom

            data[tf] = df[["Open", "High", "Low", "Close", "Volume", "VWAP"]]

            if current_price is None and tf == "5min":
                current_price = float(df["Close"].iloc[-1])

        except Exception:
            continue

    if current_price is None:
        for tf in ["15min", "30min", "1h", "2h", "4h", "daily"]:
            if tf in data:
                current_price = float(data[tf]["Close"].iloc[-1])
                break

    if current_price is None:
        current_price = get_current_price(ticker) or 100.0

    return data, float(current_price)


def analyze_support_resistance_enhanced(ticker: str) -> Dict:
    tf_data, current_price = get_multi_timeframe_data_enhanced(ticker)
    if not tf_data:
        return {}

    results = {}
    for timeframe, data in tf_data.items():
        results[timeframe] = calculate_support_resistance_enhanced(data, timeframe, current_price)
    return results


def plot_sr_levels_enhanced(data: Dict, current_price: float) -> go.Figure:
    fig = go.Figure()

    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color="blue",
        line_width=3,
        annotation_text=f"Current Price: ${current_price:.2f}",
        annotation_position="top right",
    )

    timeframe_colors = {
        "5min": "rgba(255,165,0,0.8)",
        "15min": "rgba(255,255,0,0.8)",
        "30min": "rgba(0,255,0,0.8)",
        "1h": "rgba(0,0,255,0.8)",
        "2h": "rgba(128,0,128,0.8)",
        "4h": "rgba(165,42,42,0.8)",
        "daily": "rgba(255,255,255,0.8)",
    }

    support_data = []
    resistance_data = []

    for tf, sr in data.items():
        color = timeframe_colors.get(tf, "gray")
        for level in sr.get("support_details", []):
            support_data.append({
                "timeframe": tf,
                "price": level["price"],
                "strength": level["strength"],
                "color": color,
                "distance_pct": level["distance"] * 100,
            })
        for level in sr.get("resistance_details", []):
            resistance_data.append({
                "timeframe": tf,
                "price": level["price"],
                "strength": level["strength"],
                "color": color,
                "distance_pct": level["distance"] * 100,
            })

    if support_data:
        sdf = pd.DataFrame(support_data)
        for tf in sdf["timeframe"].unique():
            x = sdf[sdf["timeframe"] == tf]
            fig.add_trace(
                go.Scatter(
                    x=x["timeframe"],
                    y=x["price"],
                    mode="markers",
                    marker=dict(
                        color=x["color"].iloc[0],
                        size=np.clip(x["strength"] * 8, a_min=6, a_max=20),
                        symbol="triangle-up",
                    ),
                    name=f"Support ({tf})",
                    customdata=np.stack((x["strength"], x["distance_pct"]), axis=1),
                    hovertemplate="<b>Support</b><br>Price: $%{y:.2f}<br>Strength: %{customdata[0]}<br>Distance: %{customdata[1]:.2f}%<extra></extra>",
                )
            )

    if resistance_data:
        rdf = pd.DataFrame(resistance_data)
        for tf in rdf["timeframe"].unique():
            x = rdf[rdf["timeframe"] == tf]
            fig.add_trace(
                go.Scatter(
                    x=x["timeframe"],
                    y=x["price"],
                    mode="markers",
                    marker=dict(
                        color=x["color"].iloc[0],
                        size=np.clip(x["strength"] * 8, a_min=6, a_max=20),
                        symbol="triangle-down",
                    ),
                    name=f"Resistance ({tf})",
                    customdata=np.stack((x["strength"], x["distance_pct"]), axis=1),
                    hovertemplate="<b>Resistance</b><br>Price: $%{y:.2f}<br>Strength: %{customdata[0]}<br>Distance: %{customdata[1]:.2f}%<extra></extra>",
                )
            )

    fig.update_layout(
        title="Enhanced Support & Resistance Analysis",
        template="plotly_dark",
        height=700,
        xaxis=dict(
            title="Timeframe",
            categoryorder="array",
            categoryarray=["5min", "15min", "30min", "1h", "2h", "4h", "daily"],
        ),
        yaxis=dict(
            title="Price ($)",
            range=[current_price * 0.92, current_price * 1.08],
        ),
        legend=dict(orientation="v", y=1, x=1.02),
        margin=dict(r=150),
    )
    return fig


# =========================================================
# OPTIONS DATA
# =========================================================
def clear_rate_limit() -> None:
    if "yf_rate_limited_until" in st.session_state:
        del st.session_state["yf_rate_limited_until"]
        st.success("✅ Rate limit status cleared")
        st.rerun()


@st.cache_data(ttl=CONFIG["OPTIONS_CACHE_TTL"], show_spinner=False)
def get_real_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    if "yf_rate_limited_until" in st.session_state:
        if st.session_state["yf_rate_limited_until"] > time.time():
            return [], pd.DataFrame(), pd.DataFrame()
        del st.session_state["yf_rate_limited_until"]

    try:
        stock = yf.Ticker(ticker)
        expiries = list(stock.options) if stock.options else []
        if not expiries:
            return [], pd.DataFrame(), pd.DataFrame()

        nearest_expiry = expiries[0]
        time.sleep(1)

        chain = stock.option_chain(nearest_expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        if calls.empty and puts.empty:
            return [], pd.DataFrame(), pd.DataFrame()

        calls["expiry"] = nearest_expiry
        puts["expiry"] = nearest_expiry

        required_cols = ["strike", "lastPrice", "volume", "openInterest", "bid", "ask"]
        if not all(c in calls.columns for c in required_cols) or not all(c in puts.columns for c in required_cols):
            return [], pd.DataFrame(), pd.DataFrame()

        for df in [calls, puts]:
            for greek in ["delta", "gamma", "theta", "impliedVolatility"]:
                if greek not in df.columns:
                    df[greek] = np.nan

        return [nearest_expiry], calls, puts

    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ["429", "rate limit", "too many requests", "quota"]):
            st.session_state["yf_rate_limited_until"] = time.time() + CONFIG["RATE_LIMIT_COOLDOWN"]
        return [], pd.DataFrame(), pd.DataFrame()


def get_full_options_chain(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    return get_real_options_data(ticker)


def get_fallback_options_data(ticker: str) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    current_price = get_current_price(ticker)
    if current_price <= 0:
        defaults = {
            "SPY": 550, "QQQ": 480, "IWM": 215, "AAPL": 230,
            "TSLA": 250, "NVDA": 125, "MSFT": 420, "GOOGL": 175,
            "AMZN": 185, "META": 520,
        }
        current_price = defaults.get(ticker, 100)

    strike_range = max(5, current_price * 0.10)
    increment = 1 if current_price < 50 else 5 if current_price < 200 else 10
    start_strike = int((current_price - strike_range) / increment) * increment
    end_strike = int((current_price + strike_range) / increment) * increment
    strikes = [s for s in range(start_strike, end_strike + increment, increment) if s > 0]

    today = datetime.date.today()
    expiries = []
    if today.weekday() < 5:
        expiries.append(today.strftime("%Y-%m-%d"))
    next_friday = today + datetime.timedelta(days=(4 - today.weekday()) % 7 or 7)
    expiries.append(next_friday.strftime("%Y-%m-%d"))
    expiries.append((next_friday + datetime.timedelta(days=7)).strftime("%Y-%m-%d"))

    calls_data = []
    puts_data = []

    for expiry in expiries:
        expiry_date = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()
        days_to_expiry = (expiry_date - today).days
        is_0dte = days_to_expiry == 0

        for strike in strikes:
            moneyness = current_price / strike

            if moneyness > 1.05:
                call_delta = 0.7 + (moneyness - 1) * 0.2
                put_delta = call_delta - 1
                gamma = 0.02
            elif moneyness > 0.95:
                call_delta = 0.5
                put_delta = -0.5
                gamma = 0.08 if is_0dte else 0.05
            else:
                call_delta = 0.3 - (1 - moneyness) * 0.2
                put_delta = call_delta - 1
                gamma = 0.02

            theta = 0.05 if is_0dte else 0.03 if days_to_expiry <= 7 else 0.02

            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = 5 if is_0dte else 10 if days_to_expiry <= 7 else 15

            call_price = max(0.1, intrinsic_call + time_value * gamma)
            put_price = max(0.1, intrinsic_put + time_value * gamma)
            volume = 1000 if abs(moneyness - 1) < 0.05 else 500

            calls_data.append({
                "contractSymbol": f"{ticker}{expiry.replace('-', '')}C{strike*1000:08.0f}",
                "strike": strike,
                "expiry": expiry,
                "lastPrice": round(call_price, 2),
                "volume": volume,
                "openInterest": volume // 2,
                "impliedVolatility": 0.25,
                "delta": round(call_delta, 3),
                "gamma": round(gamma, 3),
                "theta": round(theta, 3),
                "bid": round(call_price * 0.98, 2),
                "ask": round(call_price * 1.02, 2),
            })

            puts_data.append({
                "contractSymbol": f"{ticker}{expiry.replace('-', '')}P{strike*1000:08.0f}",
                "strike": strike,
                "expiry": expiry,
                "lastPrice": round(put_price, 2),
                "volume": volume,
                "openInterest": volume // 2,
                "impliedVolatility": 0.25,
                "delta": round(put_delta, 3),
                "gamma": round(gamma, 3),
                "theta": round(theta, 3),
                "bid": round(put_price * 0.98, 2),
                "ask": round(put_price * 1.02, 2),
            })

    st.warning("⚠️ Demo options data in use. Not real market data.")
    return expiries, pd.DataFrame(calls_data), pd.DataFrame(puts_data)


# =========================================================
# SIGNALS
# =========================================================
def classify_moneyness(strike: float, spot: float) -> str:
    diff_pct = abs(strike - spot) / max(spot, 1e-9)
    if diff_pct < 0.01:
        return "ATM"
    if strike < spot:
        return "NTM" if diff_pct < 0.03 else "ITM"
    return "NTM" if diff_pct < 0.03 else "OTM"


def calculate_approximate_greeks(option: Dict, spot_price: float) -> Tuple[float, float, float]:
    try:
        strike = float(option["strike"])
        moneyness = spot_price / max(strike, 1e-9)
        is_call = "C" in option.get("contractSymbol", "")

        if is_call:
            if moneyness > 1.03:
                delta, gamma = 0.95, 0.01
            elif moneyness > 1.00:
                delta, gamma = 0.65, 0.05
            elif moneyness > 0.97:
                delta, gamma = 0.50, 0.08
            else:
                delta, gamma = 0.35, 0.05
        else:
            if moneyness < 0.97:
                delta, gamma = -0.95, 0.01
            elif moneyness < 1.00:
                delta, gamma = -0.65, 0.05
            elif moneyness < 1.03:
                delta, gamma = -0.50, 0.08
            else:
                delta, gamma = -0.35, 0.05

        theta = 0.05 if option["expiry"] == datetime.date.today().strftime("%Y-%m-%d") else 0.02
        return delta, gamma, theta
    except Exception:
        return 0.5, 0.05, 0.02


def validate_option_data(option: pd.Series, spot_price: float) -> bool:
    try:
        required_fields = ["strike", "lastPrice", "volume", "openInterest", "bid", "ask"]
        for field in required_fields:
            if field not in option or pd.isna(option[field]):
                return False

        if float(option["lastPrice"]) <= 0:
            return False

        liq = CONFIG["SR_SENSITIVITY"]["LIQUIDITY_THRESHOLDS"]
        if float(option["openInterest"]) < liq["min_open_interest"]:
            return False
        if float(option["volume"]) < liq["min_volume"]:
            return False

        bid_ask_spread = abs(float(option["ask"]) - float(option["bid"]))
        spread_pct = bid_ask_spread / float(option["lastPrice"]) if float(option["lastPrice"]) > 0 else float("inf")
        if spread_pct > liq["max_bid_ask_spread_pct"]:
            return False

        if pd.isna(option.get("delta")) or pd.isna(option.get("gamma")) or pd.isna(option.get("theta")):
            delta, gamma, theta = calculate_approximate_greeks(option.to_dict(), spot_price)
            option["delta"] = delta
            option["gamma"] = gamma
            option["theta"] = theta

        return True
    except Exception:
        return False


def calculate_dynamic_thresholds(stock_row: pd.Series, side: str, is_0dte: bool) -> Dict[str, float]:
    thresholds = SIGNAL_THRESHOLDS[side].copy()
    volatility = stock_row.get("ATR_pct", 0.02)
    if pd.isna(volatility):
        volatility = 0.02

    vol_multiplier = 1 + (float(volatility) * 100)

    if side == "call":
        thresholds["delta_min"] = max(0.3, min(0.8, thresholds["delta_base"] * vol_multiplier))
    else:
        thresholds["delta_max"] = min(-0.3, max(-0.8, thresholds["delta_base"] * vol_multiplier))

    thresholds["gamma_min"] = thresholds["gamma_base"] * (1 + float(volatility) * 5)
    thresholds["volume_multiplier"] = max(0.8, min(2.5, 1.0 + float(volatility) * 10))

    if is_premarket() or is_early_market():
        if side == "call":
            thresholds["delta_min"] = max(0.35, thresholds.get("delta_min", 0.5))
        else:
            thresholds["delta_max"] = min(-0.35, thresholds.get("delta_max", -0.5))
        thresholds["gamma_min"] *= 0.8

    if is_0dte:
        thresholds["gamma_min"] *= 0.7
        if side == "call":
            thresholds["delta_min"] = max(0.4, thresholds.get("delta_min", 0.5))
        else:
            thresholds["delta_max"] = min(-0.4, thresholds.get("delta_max", -0.5))

    return thresholds


def generate_enhanced_signal(option: pd.Series, side: str, stock_df: pd.DataFrame, is_0dte: bool) -> Dict:
    if stock_df.empty:
        return {"signal": False, "score": 0.0, "explanations": []}

    latest = stock_df.iloc[-1]
    current_price = float(latest["Close"])

    if not validate_option_data(option, current_price):
        return {"signal": False, "score": 0.0, "explanations": []}

    thresholds = calculate_dynamic_thresholds(latest, side, is_0dte)
    weights = thresholds["condition_weights"]

    delta = float(option["delta"])
    gamma = float(option["gamma"])
    theta = float(option["theta"])
    option_volume = float(option["volume"])

    close = float(latest["Close"])
    ema_9 = latest.get("EMA_9")
    ema_20 = latest.get("EMA_20")
    rsi = latest.get("RSI")
    vwap = latest.get("VWAP")

    weighted_score = 0.0
    explanations = []
    core_conditions = []

    if side == "call":
        delta_pass = delta >= thresholds.get("delta_min", 0.5)
        trend_pass = pd.notna(ema_9) and pd.notna(ema_20) and close > float(ema_9) > float(ema_20)
        momentum_pass = pd.notna(rsi) and float(rsi) > thresholds["rsi_min"]
        vwap_pass = pd.notna(vwap) and close > float(vwap)
    else:
        delta_pass = delta <= thresholds.get("delta_max", -0.5)
        trend_pass = pd.notna(ema_9) and pd.notna(ema_20) and close < float(ema_9) < float(ema_20)
        momentum_pass = pd.notna(rsi) and float(rsi) < thresholds["rsi_max"]
        vwap_pass = pd.notna(vwap) and close < float(vwap)

    gamma_pass = gamma >= thresholds.get("gamma_min", 0.05)
    theta_pass = theta <= thresholds["theta_base"]
    volume_pass = option_volume >= thresholds["volume_min"]

    checks = [
        ("Delta", delta_pass, weights["delta"]),
        ("Gamma", gamma_pass, weights["gamma"]),
        ("Theta", theta_pass, weights["theta"]),
        ("Trend", trend_pass, weights["trend"]),
        ("Momentum", momentum_pass, weights["momentum"]),
        ("Volume", volume_pass, weights["volume"]),
        ("VWAP", vwap_pass, 0.15),
    ]

    for name, passed, weight in checks:
        if passed:
            weighted_score += weight
        explanations.append({
            "condition": name,
            "passed": bool(passed),
            "weight": float(weight),
        })

    core_conditions = [delta_pass, gamma_pass, theta_pass, trend_pass]
    signal_ok = all(core_conditions)

    profit_target = None
    stop_loss = None
    holding_period = None
    est_hourly_decay = 0.0
    est_remaining_decay = 0.0

    if signal_ok:
        entry_price = float(option["lastPrice"])
        slippage_pct = 0.005
        commission_per_contract = 0.65
        adjusted_entry = entry_price * (1 + slippage_pct) + commission_per_contract
        option_type = "call" if side == "call" else "put"

        profit_target = adjusted_entry * (1 + CONFIG["PROFIT_TARGETS"][option_type])
        stop_loss = adjusted_entry * (1 - CONFIG["PROFIT_TARGETS"]["stop_loss"])

        expiry_date = datetime.datetime.strptime(option["expiry"], "%Y-%m-%d").date()
        days_to_expiry = (expiry_date - datetime.date.today()).days
        if days_to_expiry == 0:
            holding_period = "Intraday (Exit before 3:30 PM)"
        elif days_to_expiry <= 3:
            holding_period = "1-2 days"
        else:
            holding_period = "3-7 days"

        if is_0dte and theta:
            est_hourly_decay = -theta / CONFIG["TRADING_HOURS_PER_DAY"]
            est_remaining_decay = est_hourly_decay * calculate_remaining_trading_hours()

    return {
        "signal": signal_ok,
        "score": weighted_score,
        "max_score": 1.0,
        "score_percentage": weighted_score * 100,
        "explanations": explanations,
        "thresholds": thresholds,
        "profit_target": profit_target,
        "stop_loss": stop_loss,
        "holding_period": holding_period,
        "est_hourly_decay": est_hourly_decay,
        "est_remaining_decay": est_remaining_decay,
        "passed_conditions": [e["condition"] for e in explanations if e["passed"]],
        "failed_conditions": [e["condition"] for e in explanations if not e["passed"]],
        "open_interest": option["openInterest"],
        "volume": option["volume"],
        "implied_volatility": option.get("impliedVolatility", np.nan),
    }


def process_options_batch(options_df: pd.DataFrame, side: str, stock_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    if options_df.empty or stock_df.empty:
        return pd.DataFrame()

    df = options_df.copy()
    df = df[df["lastPrice"] > 0]
    df = df.dropna(subset=["strike", "lastPrice", "volume", "openInterest"])
    if df.empty:
        return pd.DataFrame()

    today = datetime.date.today()
    df["is_0dte"] = df["expiry"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date() == today)
    df["moneyness"] = df["strike"].apply(lambda x: classify_moneyness(float(x), current_price))

    for idx, row in df.iterrows():
        if pd.isna(row.get("delta")) or pd.isna(row.get("gamma")) or pd.isna(row.get("theta")):
            delta, gamma, theta = calculate_approximate_greeks(row.to_dict(), current_price)
            df.at[idx, "delta"] = delta
            df.at[idx, "gamma"] = gamma
            df.at[idx, "theta"] = theta

    signals = []
    for _, row in df.iterrows():
        result = generate_enhanced_signal(row, side, stock_df, bool(row["is_0dte"]))
        if result["signal"]:
            payload = row.to_dict()
            payload.update(result)
            signals.append(payload)

    if not signals:
        return pd.DataFrame()

    out = pd.DataFrame(signals)
    return out.sort_values("score_percentage", ascending=False)


def calculate_scanner_score(stock_df: pd.DataFrame, side: str) -> float:
    if stock_df.empty:
        return 0.0

    latest = stock_df.iloc[-1]
    score = 0.0
    max_score = 5.0

    close = latest.get("Close", np.nan)
    ema_9 = latest.get("EMA_9", np.nan)
    ema_20 = latest.get("EMA_20", np.nan)
    ema_50 = latest.get("EMA_50", np.nan)
    ema_200 = latest.get("EMA_200", np.nan)
    rsi = latest.get("RSI", np.nan)
    macd = latest.get("MACD", np.nan)
    macd_signal = latest.get("MACD_signal", np.nan)
    vwap = latest.get("VWAP", np.nan)

    if side == "call":
        if pd.notna(ema_9) and pd.notna(ema_20) and close > ema_9 > ema_20:
            score += 1
        if pd.notna(ema_50) and pd.notna(ema_200) and ema_50 > ema_200:
            score += 1
        if pd.notna(rsi) and rsi > 50:
            score += 1
        if pd.notna(macd) and pd.notna(macd_signal) and macd > macd_signal:
            score += 1
        if pd.notna(vwap) and close > vwap:
            score += 1
    else:
        if pd.notna(ema_9) and pd.notna(ema_20) and close < ema_9 < ema_20:
            score += 1
        if pd.notna(ema_50) and pd.notna(ema_200) and ema_50 < ema_200:
            score += 1
        if pd.notna(rsi) and rsi < 50:
            score += 1
        if pd.notna(macd) and pd.notna(macd_signal) and macd < macd_signal:
            score += 1
        if pd.notna(vwap) and close < vwap:
            score += 1

    return (score / max_score) * 100


# =========================================================
# CHART
# =========================================================
def create_stock_chart(df: pd.DataFrame, sr_levels: Optional[Dict] = None, timeframe: str = "5m") -> Optional[go.Figure]:
    if df.empty:
        return None

    df = flatten_columns(df.copy()).reset_index()
    df = ensure_datetime_column(df)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return None

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)

    if df.empty:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    for period in [9, 20, 50, 200]:
        if len(df) >= period:
            df[f"EMA_{period}"] = EMAIndicator(close=close, window=period).ema_indicator()
        else:
            df[f"EMA_{period}"] = np.nan

    if len(df) >= 14:
        df["RSI"] = RSIIndicator(close=close, window=14).rsi()
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["ATR"] = atr.average_true_range()
        df["ATR_pct"] = df["ATR"] / close.replace(0, np.nan)
    else:
        df["RSI"] = np.nan
        df["ATR"] = np.nan
        df["ATR_pct"] = np.nan

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)

    if len(df) >= 26:
        macd = MACD(close=close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
        kc = KeltnerChannel(high=high, low=low, close=close)
        df["KC_upper"] = kc.keltner_channel_hband()
        df["KC_middle"] = kc.keltner_channel_mband()
        df["KC_lower"] = kc.keltner_channel_lband()
    else:
        for col in ["MACD", "MACD_signal", "MACD_hist", "KC_upper", "KC_middle", "KC_lower"]:
            df[col] = np.nan

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.15, 0.15, 0.15],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Datetime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
            increasing_fillcolor="green",
            decreasing_fillcolor="red",
        ),
        row=1,
        col=1,
    )

    for color, period in zip(["lime", "cyan", "magenta", "yellow"], [9, 20, 50, 200]):
        col_name = f"EMA_{period}"
        if not df[col_name].isna().all():
            fig.add_trace(
                go.Scatter(x=df["Datetime"], y=df[col_name], name=f"EMA {period}", line=dict(color=color)),
                row=1,
                col=1,
            )

    for col_name, color, name in [
        ("KC_upper", "red", "KC Upper"),
        ("KC_middle", "green", "KC Middle"),
        ("KC_lower", "red", "KC Lower"),
    ]:
        if not df[col_name].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df["Datetime"],
                    y=df[col_name],
                    name=name,
                    line=dict(color=color, dash="dash" if col_name != "KC_middle" else "solid"),
                ),
                row=1,
                col=1,
            )

    if not df["VWAP"].isna().all():
        fig.add_trace(
            go.Scatter(x=df["Datetime"], y=df["VWAP"], name="VWAP", line=dict(color="cyan", width=2)),
            row=1,
            col=1,
        )

    volume_colors = ["green" if c >= o else "red" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df["Datetime"], y=df["Volume"], name="Volume", marker_color=volume_colors), row=2, col=1)

    if not df["MACD"].isna().all():
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD"], name="MACD", line=dict(color="blue")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD_signal"], name="Signal", line=dict(color="orange")), row=3, col=1)
        hist_colors = ["green" if x >= 0 else "red" for x in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df["Datetime"], y=df["MACD_hist"], name="Histogram", marker_color=hist_colors), row=3, col=1)

    if not df["RSI"].isna().all():
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], name="RSI", line=dict(color="purple")), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    tf_mapping = {
        "5m": "5min", "15m": "15min", "30m": "30min",
        "1H": "1h", "1D": "daily", "1W": "daily", "1M": "daily",
    }
    tf_key = tf_mapping.get(timeframe, "5min")

    if sr_levels and tf_key in sr_levels:
        for level in sr_levels[tf_key].get("support", []):
            fig.add_hline(y=level, line_dash="dash", line_color="green", row=1, col=1,
                          annotation_text=f"S: {level:.2f}", annotation_position="bottom right")
        for level in sr_levels[tf_key].get("resistance", []):
            fig.add_hline(y=level, line_dash="dash", line_color="red", row=1, col=1,
                          annotation_text=f"R: {level:.2f}", annotation_position="top right")

    fig.update_layout(
        height=800,
        title=f"Price Chart - {timeframe}",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template="plotly_dark",
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
    )

    for row in [1, 2, 3, 4]:
        fig.update_yaxes(side="right", row=row, col=1)

    return fig


# =========================================================
# BACKTEST
# =========================================================
def run_backtest(signals_df: pd.DataFrame, stock_df: pd.DataFrame, side: str) -> Optional[pd.DataFrame]:
    if signals_df.empty or stock_df.empty:
        return None

    results = []
    returns = []

    for _, row in signals_df.iterrows():
        entry_price = float(row["lastPrice"])
        recent_closes = stock_df["Close"].tail(10).values
        pnls = []

        for exit_price in recent_closes:
            if side == "call":
                pnl = max(0, exit_price - row["strike"]) - entry_price
            else:
                pnl = max(0, row["strike"] - exit_price) - entry_price
            pnl *= 0.95
            pnls.append(pnl)

        avg_pnl = float(np.mean(pnls)) if pnls else 0.0
        pnl_pct = (avg_pnl / entry_price) * 100 if entry_price > 0 else 0.0
        returns.append(pnl_pct / 100.0)

        results.append({
            "contract": row["contractSymbol"],
            "entry_price": entry_price,
            "avg_pnl": avg_pnl,
            "pnl_pct": pnl_pct,
            "score": row["score_percentage"],
        })

    out = pd.DataFrame(results).sort_values("pnl_pct", ascending=False)
    if len(returns) > 1:
        arr = np.array(returns)
        mean_ret = np.mean(arr)
        std_ret = np.std(arr)
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / np.where(peak == 0, 1, peak)
        max_dd = float(np.min(dd) * 100) if len(dd) else 0.0
        gross_profit = np.sum(arr[arr > 0])
        gross_loss = abs(np.sum(arr[arr < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        out["sharpe_ratio"] = sharpe
        out["max_drawdown_pct"] = max_dd
        out["profit_factor"] = profit_factor

    return out


# =========================================================
# PERFORMANCE
# =========================================================
def measure_performance() -> None:
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = {
            "start_time": time.time(),
            "api_calls": 0,
            "data_points_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_usage": 0.0,
        }

    try:
        import psutil
        process = psutil.Process()
        st.session_state.performance_metrics["memory_usage"] = process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    with st.expander("⚡ Performance Metrics", expanded=False):
        elapsed = time.time() - st.session_state.performance_metrics["start_time"]
        st.metric("Uptime", f"{elapsed:.1f}s")
        st.metric("API Calls", st.session_state.performance_metrics["api_calls"])
        st.metric("Data Points Processed", st.session_state.performance_metrics["data_points_processed"])
        ratio = (
            st.session_state.performance_metrics["cache_hits"] /
            max(1, st.session_state.performance_metrics["cache_hits"] + st.session_state.performance_metrics["cache_misses"])
        ) * 100
        st.metric("Cache Hit Ratio", f"{ratio:.1f}%")
        st.metric("Memory Usage", f"{st.session_state.performance_metrics['memory_usage']:.1f} MB")


# =========================================================
# NEWS / ECONOMIC CALENDAR
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_benzinga_news(symbol: str, token: str, limit: int = 5) -> List[Dict]:
    if not token:
        return []

    url = "https://api.benzinga.com/api/v2/news"
    params = {
        "token": token,
        "symbols": symbol,
        "size": max(1, min(limit, 20)),
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    articles = data.get("articles") if isinstance(data, dict) else data
    if not isinstance(articles, list):
        return []

    normalized = []
    for a in articles[:limit]:
        normalized.append({
            "title": a.get("title", "No title"),
            "source": a.get("author") or a.get("source") or "Benzinga",
            "time": a.get("created") or a.get("created_at") or a.get("pubDate") or "",
            "summary": a.get("teaser") or a.get("description") or a.get("body") or "",
            "url": a.get("url") or a.get("link") or "",
        })
    return normalized


def to_local_timestr(dt_like) -> str:
    local_tz = pytz.timezone("Africa/Casablanca")

    try:
        if isinstance(dt_like, (int, float)) or (isinstance(dt_like, str) and str(dt_like).isdigit()):
            ts = int(float(dt_like))
            return datetime.datetime.fromtimestamp(ts, pytz.UTC).astimezone(local_tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    try:
        s = str(dt_like).replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(local_tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown time"


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔑 API Settings")
        st.info("API keys are loaded from Streamlit secrets, not typed into the app.")
        st.caption("Required keys if used: POLYGON_API_KEY, ALPHA_VANTAGE_API_KEY, FMP_API_KEY, IEX_API_KEY, BENZINGA_API_KEY")

        st.subheader("🔄 Smart Auto-Refresh")
        enable_auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.auto_refresh_enabled,
            key="auto_refresh_enabled_checkbox",
        )
        st.session_state.auto_refresh_enabled = enable_auto_refresh

        if enable_auto_refresh:
            refresh_interval = st.selectbox(
                "Refresh Interval",
                options=[60, 120, 300, 600],
                index=1,
                format_func=lambda x: f"{x} seconds" if x < 60 else f"{x // 60} minute{'s' if x > 60 else ''}",
            )
            st.session_state.refresh_interval = refresh_interval

        with st.expander("📊 Signal Thresholds & Weights", expanded=False):
            st.markdown("#### 📈 Calls")
            SIGNAL_THRESHOLDS["call"]["condition_weights"]["delta"] = st.slider("Call Delta Weight", 0.1, 0.4, 0.25, 0.05)
            SIGNAL_THRESHOLDS["call"]["condition_weights"]["gamma"] = st.slider("Call Gamma Weight", 0.1, 0.3, 0.20, 0.05)
            SIGNAL_THRESHOLDS["call"]["condition_weights"]["trend"] = st.slider("Call Trend Weight", 0.1, 0.3, 0.20, 0.05)
            SIGNAL_THRESHOLDS["call"]["delta_base"] = st.slider("Call Delta Base", 0.1, 1.0, 0.5, 0.1)
            SIGNAL_THRESHOLDS["call"]["gamma_base"] = st.slider("Call Gamma Base", 0.01, 0.2, 0.05, 0.01)
            SIGNAL_THRESHOLDS["call"]["volume_min"] = st.slider("Call Min Volume", 100, 5000, 1000, 100)

            st.markdown("#### 📉 Puts")
            SIGNAL_THRESHOLDS["put"]["condition_weights"]["delta"] = st.slider("Put Delta Weight", 0.1, 0.4, 0.25, 0.05)
            SIGNAL_THRESHOLDS["put"]["condition_weights"]["gamma"] = st.slider("Put Gamma Weight", 0.1, 0.3, 0.20, 0.05)
            SIGNAL_THRESHOLDS["put"]["condition_weights"]["trend"] = st.slider("Put Trend Weight", 0.1, 0.3, 0.20, 0.05)
            SIGNAL_THRESHOLDS["put"]["delta_base"] = st.slider("Put Delta Base", -1.0, -0.1, -0.5, 0.1)
            SIGNAL_THRESHOLDS["put"]["gamma_base"] = st.slider("Put Gamma Base", 0.01, 0.2, 0.05, 0.01)
            SIGNAL_THRESHOLDS["put"]["volume_min"] = st.slider("Put Min Volume", 100, 5000, 1000, 100)

        with st.expander("🎯 Risk Management", expanded=False):
            CONFIG["PROFIT_TARGETS"]["call"] = st.slider("Call Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
            CONFIG["PROFIT_TARGETS"]["put"] = st.slider("Put Profit Target (%)", 0.05, 0.50, 0.15, 0.01)
            CONFIG["PROFIT_TARGETS"]["stop_loss"] = st.slider("Stop Loss (%)", 0.03, 0.20, 0.08, 0.01)

        st.subheader("🕐 Market Status")
        if is_market_open():
            st.success("🟢 Market OPEN")
        elif is_premarket():
            st.warning("🟡 Premarket")
        else:
            st.info("🔴 Market CLOSED")

        st.caption(f"ET: {eastern_now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.markdown("---")
        st.subheader("🗑️ Cache Management")

        if st.button("🧹 Clear All Cache"):
            st.cache_data.clear()
            for key in ["sr_data", "last_ticker", "yf_rate_limited_until"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ All cache cleared.")
            st.rerun()

        measure_performance()


# =========================================================
# TABS
# =========================================================
def render_general_tab(ticker: str) -> pd.DataFrame:
    st.header("🎯 Enhanced Options Signals")

    col1, col2, col3, col4, col5 = st.columns(5)
    current_price = get_current_price(ticker)
    cache_age = int(time.time() - st.session_state.get("last_refresh", 0))

    with col1:
        if is_market_open():
            st.success("🟢 OPEN")
        elif is_premarket():
            st.warning("🟡 PRE")
        else:
            st.info("🔴 CLOSED")

    with col2:
        st.metric("Price", f"${current_price:.2f}" if current_price > 0 else "N/A")
    with col3:
        st.metric("Cache Age", f"{cache_age}s")
    with col4:
        st.metric("Refreshes", st.session_state.refresh_counter)
    with col5:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.session_state.refresh_counter += 1
            st.rerun()

    if not st.session_state.sr_data or st.session_state.last_ticker != ticker:
        with st.spinner("🔍 Analyzing support/resistance..."):
            st.session_state.sr_data = analyze_support_resistance_enhanced(ticker)
            st.session_state.last_ticker = ticker

    df = get_stock_data_with_indicators(ticker)
    if df.empty:
        st.error("❌ Unable to fetch stock data.")
        return pd.DataFrame()

    current_price = float(df.iloc[-1]["Close"])
    st.success(f"✅ {ticker} - ${current_price:.2f}")

    atr_pct = df.iloc[-1].get("ATR_pct", np.nan)
    if pd.notna(atr_pct):
        if atr_pct > CONFIG["VOLATILITY_THRESHOLDS"]["high"]:
            vol_status, icon = "Extreme", "🔴"
        elif atr_pct > CONFIG["VOLATILITY_THRESHOLDS"]["medium"]:
            vol_status, icon = "High", "🟡"
        elif atr_pct > CONFIG["VOLATILITY_THRESHOLDS"]["low"]:
            vol_status, icon = "Medium", "🟠"
        else:
            vol_status, icon = "Low", "🟢"
        st.info(f"{icon} Volatility: {atr_pct * 100:.2f}% ({vol_status})")

    with st.spinner("📥 Fetching options data..."):
        expiries, all_calls, all_puts = get_full_options_chain(ticker)

    show_demo = False
    if not expiries:
        st.error("❌ Unable to fetch real options data.")
        with st.expander("💡 Solutions", expanded=True):
            remaining = 0
            if "yf_rate_limited_until" in st.session_state:
                remaining = max(0, int(st.session_state["yf_rate_limited_until"] - time.time()))
            if remaining > 0:
                st.warning(f"⏳ Rate limited for {remaining} more seconds.")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔄 Clear Rate Limit"):
                    clear_rate_limit()
            with c2:
                if st.button("⏰ Force Retry"):
                    if "yf_rate_limited_until" in st.session_state:
                        del st.session_state["yf_rate_limited_until"]
                    st.cache_data.clear()
                    st.rerun()
            with c3:
                show_demo = st.button("📊 Show Demo Data")

        if show_demo:
            expiries, all_calls, all_puts = get_fallback_options_data(ticker)
        else:
            st.info("Use Technical or Chart tabs while options data is unavailable.")
            return df

    if not expiries:
        return df

    st.success(f"✅ Options loaded: {len(all_calls)} calls, {len(all_puts)} puts")

    colA, colB = st.columns(2)
    with colA:
        expiry_mode = st.radio("📅 Expiration Filter", ["0DTE Only", "This Week", "All Near-Term"], index=1)
    with colB:
        st.info(f"Total expiries returned: {len(expiries)}")

    today = datetime.date.today()
    if expiry_mode == "0DTE Only":
        expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
    elif expiry_mode == "This Week":
        week_end = today + datetime.timedelta(days=7)
        expiries_to_use = [e for e in expiries if today <= datetime.datetime.strptime(e, "%Y-%m-%d").date() <= week_end]
    else:
        expiries_to_use = expiries[:5]

    if not expiries_to_use:
        st.warning("⚠️ No expiries available for selected mode.")
        return df

    calls_filtered = all_calls[all_calls["expiry"].isin(expiries_to_use)].copy()
    puts_filtered = all_puts[all_puts["expiry"].isin(expiries_to_use)].copy()

    strike_range = st.slider("🎯 Strike Range Around Current Price ($)", -50, 50, (-10, 10), 1)
    min_strike = current_price + strike_range[0]
    max_strike = current_price + strike_range[1]

    calls_filtered = calls_filtered[(calls_filtered["strike"] >= min_strike) & (calls_filtered["strike"] <= max_strike)].copy()
    puts_filtered = puts_filtered[(puts_filtered["strike"] >= min_strike) & (puts_filtered["strike"] <= max_strike)].copy()

    m_filter = st.multiselect("💰 Moneyness Filter", ["ITM", "NTM", "ATM", "OTM"], default=["NTM", "ATM"])
    if not calls_filtered.empty:
        calls_filtered["moneyness"] = calls_filtered["strike"].apply(lambda x: classify_moneyness(float(x), current_price))
        calls_filtered = calls_filtered[calls_filtered["moneyness"].isin(m_filter)]
    if not puts_filtered.empty:
        puts_filtered["moneyness"] = puts_filtered["strike"].apply(lambda x: classify_moneyness(float(x), current_price))
        puts_filtered = puts_filtered[puts_filtered["moneyness"].isin(m_filter)]

    st.write(f"🔍 Filtered Options: {len(calls_filtered)} calls, {len(puts_filtered)} puts")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Enhanced Call Signals")
        if not calls_filtered.empty:
            call_signals_df = process_options_batch(calls_filtered, "call", df, current_price)
            if not call_signals_df.empty:
                display_cols = [
                    "contractSymbol", "strike", "lastPrice", "volume", "delta", "gamma",
                    "theta", "moneyness", "score_percentage", "profit_target",
                    "stop_loss", "holding_period", "is_0dte",
                ]
                show = call_signals_df[[c for c in display_cols if c in call_signals_df.columns]].copy()
                show = show.rename(columns={
                    "score_percentage": "Score%",
                    "profit_target": "Target",
                    "stop_loss": "Stop",
                    "holding_period": "Hold Period",
                    "is_0dte": "0DTE",
                })
                st.dataframe(show.round(3), use_container_width=True, hide_index=True)
                st.success(f"✅ {len(call_signals_df)} call signals | Avg: {call_signals_df['score_percentage'].mean():.1f}%")

                best = call_signals_df.iloc[0]
                with st.expander(f"🏆 Best Call Signal ({best['contractSymbol']})"):
                    a, b, c = st.columns(3)
                    a.metric("Score", f"{best['score_percentage']:.1f}%")
                    a.metric("Delta", f"{best['delta']:.3f}")
                    a.metric("Open Interest", f"{best['open_interest']}")
                    b.metric("Profit Target", f"${best['profit_target']:.2f}" if pd.notna(best["profit_target"]) else "N/A")
                    b.metric("Gamma", f"{best['gamma']:.3f}")
                    b.metric("Volume", f"{best['volume']}")
                    c.metric("Stop Loss", f"${best['stop_loss']:.2f}" if pd.notna(best["stop_loss"]) else "N/A")
                    iv = best["implied_volatility"]
                    c.metric("Implied Vol", f"{float(iv) * 100:.1f}%" if pd.notna(iv) else "N/A")
                    c.metric("Holding", best["holding_period"])

                bt = run_backtest(call_signals_df, df, "call")
                if bt is not None and not bt.empty:
                    with st.expander("🔬 Backtest Results"):
                        st.dataframe(bt, use_container_width=True, hide_index=True)
            else:
                st.info("No call signals found.")
        else:
            st.info("No call options available for selected filters.")

    with col2:
        st.subheader("📉 Enhanced Put Signals")
        if not puts_filtered.empty:
            put_signals_df = process_options_batch(puts_filtered, "put", df, current_price)
            if not put_signals_df.empty:
                display_cols = [
                    "contractSymbol", "strike", "lastPrice", "volume", "delta", "gamma",
                    "theta", "moneyness", "score_percentage", "profit_target",
                    "stop_loss", "holding_period", "is_0dte",
                ]
                show = put_signals_df[[c for c in display_cols if c in put_signals_df.columns]].copy()
                show = show.rename(columns={
                    "score_percentage": "Score%",
                    "profit_target": "Target",
                    "stop_loss": "Stop",
                    "holding_period": "Hold Period",
                    "is_0dte": "0DTE",
                })
                st.dataframe(show.round(3), use_container_width=True, hide_index=True)
                st.success(f"✅ {len(put_signals_df)} put signals | Avg: {put_signals_df['score_percentage'].mean():.1f}%")

                best = put_signals_df.iloc[0]
                with st.expander(f"🏆 Best Put Signal ({best['contractSymbol']})"):
                    a, b, c = st.columns(3)
                    a.metric("Score", f"{best['score_percentage']:.1f}%")
                    a.metric("Delta", f"{best['delta']:.3f}")
                    a.metric("Open Interest", f"{best['open_interest']}")
                    b.metric("Profit Target", f"${best['profit_target']:.2f}" if pd.notna(best["profit_target"]) else "N/A")
                    b.metric("Gamma", f"{best['gamma']:.3f}")
                    b.metric("Volume", f"{best['volume']}")
                    c.metric("Stop Loss", f"${best['stop_loss']:.2f}" if pd.notna(best["stop_loss"]) else "N/A")
                    iv = best["implied_volatility"]
                    c.metric("Implied Vol", f"{float(iv) * 100:.1f}%" if pd.notna(iv) else "N/A")
                    c.metric("Holding", best["holding_period"])

                bt = run_backtest(put_signals_df, df, "put")
                if bt is not None and not bt.empty:
                    with st.expander("🔬 Backtest Results"):
                        st.dataframe(bt, use_container_width=True, hide_index=True)
            else:
                st.info("No put signals found.")
        else:
            st.info("No put options available for selected filters.")

    st.markdown("---")
    st.subheader("🧠 Technical Scanner Scores")
    call_score = calculate_scanner_score(df, "call")
    put_score = calculate_scanner_score(df, "put")

    c1, c2, c3 = st.columns(3)
    c1.metric("📈 Call Scanner", f"{call_score:.1f}%")
    c2.metric("📉 Put Scanner", f"{put_score:.1f}%")
    bias = "Bullish" if call_score > put_score else "Bearish" if put_score > call_score else "Neutral"
    c3.metric("🎯 Directional Bias", bias)

    return df


def render_chart_tab(ticker: str) -> None:
    st.header("📊 Professional Chart")

    timeframes = ["5m", "15m", "30m", "1H", "1D", "1W", "1M"]
    selected_timeframe = st.selectbox("Select Timeframe:", timeframes, index=0)
    st.session_state.current_timeframe = selected_timeframe

    tf_mapping = {
        "5m": "5m", "15m": "15m", "30m": "30m",
        "1H": "60m", "1D": "1d", "1W": "1wk", "1M": "1mo",
    }
    period_mapping = {
        "5m": "5d", "15m": "15d", "30m": "30d",
        "1H": "60d", "1D": "1y", "1W": "2y", "1M": "5y",
    }

    with st.spinner(f"Loading {selected_timeframe} chart..."):
        df = yf.download(
            ticker,
            period=period_mapping[selected_timeframe],
            interval=tf_mapping[selected_timeframe],
            prepost=True,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            st.error("No chart data available.")
            return

        fig = create_stock_chart(df, st.session_state.get("sr_data", {}), selected_timeframe)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to create chart.")


def render_news_tab(ticker: str, df: pd.DataFrame) -> None:
    st.header("📰 Market News & Analysis")

    stock = yf.Ticker(ticker)
    news_fetched = False

    benzinga_key = st.secrets.get("BENZINGA_API_KEY", "")
    alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    fmp_key = st.secrets.get("FMP_API_KEY", "")

    try:
        bz_items = fetch_benzinga_news(ticker, benzinga_key, limit=5)
        if bz_items:
            st.subheader(f"Latest News for {ticker} (Benzinga)")
            for item in bz_items:
                st.markdown(f"### {item['title']}")
                st.caption(f"Source: {item['source']} | {to_local_timestr(item['time'])}")
                if item["summary"]:
                    st.write(item["summary"])
                if item["url"]:
                    st.markdown(f"[Read more]({item['url']})")
                st.divider()
            news_fetched = True
    except Exception:
        pass

    if not news_fetched and alpha_vantage_key:
        try:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_vantage_key}&limit=5"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if "feed" in data and data["feed"]:
                st.subheader(f"Latest News for {ticker} (Alpha Vantage)")
                for item in data["feed"][:5]:
                    st.markdown(f"### {item.get('title', 'No title')}")
                    st.caption(f"Source: {item.get('source', 'Unknown')} | {to_local_timestr(item.get('time_published', ''))}")
                    if item.get("summary"):
                        st.write(item["summary"])
                    if item.get("url"):
                        st.markdown(f"[Read more]({item['url']})")
                    st.divider()
                news_fetched = True
        except Exception:
            pass

    if not news_fetched:
        try:
            news = stock.news
            if news:
                st.subheader(f"Latest News for {ticker} (Yahoo Finance)")
                for item in news[:5]:
                    st.markdown(f"### {item.get('title', 'No title')}")
                    st.caption(f"Publisher: {item.get('publisher', 'Unknown')} | {to_local_timestr(item.get('providerPublishTime', time.time()))}")
                    if item.get("summary"):
                        st.write(item["summary"])
                    if item.get("link"):
                        st.markdown(f"[Read more]({item['link']})")
                    st.divider()
                news_fetched = True
        except Exception:
            pass

    if not news_fetched:
        st.info("News data is temporarily unavailable.")

    st.subheader("Market Analysis")
    with st.expander("Technical Analysis Summary", expanded=True):
        if not df.empty:
            latest = df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("RSI", f"{latest['RSI']:.1f}" if pd.notna(latest["RSI"]) else "N/A")
                st.metric("MACD", f"{latest['MACD']:.3f}" if pd.notna(latest["MACD"]) else "N/A")
            with c2:
                trend = "Bullish" if latest["Close"] > latest["EMA_20"] else "Bearish" if latest["Close"] < latest["EMA_20"] else "Neutral"
                vol_vs_avg = latest["Volume"] / max(latest.get("avg_vol", 1), 1)
                st.metric("Trend", trend)
                st.metric("Volume vs Avg", f"{vol_vs_avg:.1f}x")
            with c3:
                sr = st.session_state.get("sr_data", {})
                st.metric("Support Levels", len(sr.get("5min", {}).get("support", [])))
                st.metric("Resistance Levels", len(sr.get("5min", {}).get("resistance", [])))

    st.subheader("📅 US Economic Calendar (Upcoming)")
    if fmp_key:
        try:
            start_date = datetime.date.today()
            end_date = start_date + datetime.timedelta(days=7)
            url = (
                "https://financialmodelingprep.com/api/v3/economic_calendar"
                f"?from={start_date.isoformat()}&to={end_date.isoformat()}&apikey={fmp_key}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()

            us_events = [e for e in data if e.get("country") == "US"]
            if us_events:
                cal = pd.DataFrame(us_events)[["date", "event", "actual", "previous", "change", "estimate", "impact"]]
                cal = cal.rename(columns={
                    "date": "Date (UTC)", "event": "Event", "actual": "Actual",
                    "previous": "Previous", "change": "Change",
                    "estimate": "Estimate", "impact": "Impact",
                })
                st.dataframe(cal, use_container_width=True, height=300)
            else:
                st.info("No upcoming US economic events in the next 7 days.")
        except Exception as e:
            st.warning(f"Error fetching economic calendar: {e}")
    else:
        st.warning("Add FMP_API_KEY in Streamlit secrets to enable the economic calendar.")


def render_financials_tab(ticker: str) -> None:
    st.header("💼 Financial Analysis")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            market_cap = info.get("marketCap")
            if market_cap:
                if market_cap > 1e12:
                    st.metric("Market Cap", f"${market_cap/1e12:.2f}T")
                elif market_cap > 1e9:
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", f"${market_cap/1e6:.2f}M")
            st.metric("P/E Ratio", f"{info.get('trailingPE', np.nan):.2f}" if info.get("trailingPE") else "N/A")
        with c2:
            pm = info.get("profitMargins")
            roe = info.get("returnOnEquity")
            st.metric("Profit Margin", f"{pm*100:.2f}%" if pm is not None else "N/A")
            st.metric("ROE", f"{roe*100:.2f}%" if roe is not None else "N/A")
        with c3:
            dte = info.get("debtToEquity")
            cr = info.get("currentRatio")
            st.metric("Debt/Equity", f"{dte:.2f}" if dte is not None else "N/A")
            st.metric("Current Ratio", f"{cr:.2f}" if cr is not None else "N/A")
        with c4:
            dy = info.get("dividendYield")
            beta = info.get("beta")
            st.metric("Dividend Yield", f"{dy*100:.2f}%" if dy is not None else "N/A")
            st.metric("Beta", f"{beta:.2f}" if beta is not None else "N/A")

        st.subheader("Financial Statements")
        statement_type = st.selectbox("Select Statement", ["Income Statement", "Balance Sheet", "Cash Flow"])
        if statement_type == "Income Statement" and not financials.empty:
            st.dataframe(financials.head(10), use_container_width=True)
        elif statement_type == "Balance Sheet" and not balance_sheet.empty:
            st.dataframe(balance_sheet.head(10), use_container_width=True)
        elif statement_type == "Cash Flow" and not cashflow.empty:
            st.dataframe(cashflow.head(10), use_container_width=True)
        else:
            st.info("Financial data not available.")
    except Exception as e:
        st.error(f"Error loading financial data: {e}")


def render_technical_tab(ticker: str) -> None:
    st.header("📈 Technical Analysis")

    current_price = get_current_price(ticker)
    tf_data, _ = get_multi_timeframe_data_enhanced(ticker)
    sr_results = {}

    for tf in ["5min", "15min", "30min", "1h", "2h", "4h", "daily"]:
        if tf in tf_data and not tf_data[tf].empty:
            sr_results[tf] = calculate_support_resistance_enhanced(tf_data[tf], tf, current_price)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Strong Support Levels")
        for tf in sr_results:
            if sr_results[tf]["support"]:
                strongest = min(sr_results[tf]["support"], key=lambda x: abs(x - current_price))
                st.write(f"**{tf}**: ${strongest:.2f}")

    with c2:
        st.subheader("Strong Resistance Levels")
        for tf in sr_results:
            if sr_results[tf]["resistance"]:
                strongest = min(sr_results[tf]["resistance"], key=lambda x: abs(x - current_price))
                st.write(f"**{tf}**: ${strongest:.2f}")

    if sr_results:
        fig = plot_sr_levels_enhanced(sr_results, current_price)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Technical Studies")
    study_type = st.selectbox("Select Study", ["Moving Averages", "Oscillators", "Volatility", "Volume"])
    if study_type == "Moving Averages":
        st.info("Golden Cross: 50MA > 200MA | Death Cross: 50MA < 200MA")
    elif study_type == "Oscillators":
        oscillator = st.selectbox("Select Oscillator", ["RSI", "Stochastic", "MACD", "CCI"])
        if oscillator == "RSI":
            st.info("RSI > 70 = Overbought | RSI < 30 = Oversold")


def render_forum_tab() -> None:
    st.header("💬 Trading Community")
    st.info(
        "Community features coming later: chat, strategy sharing, trade ideas, and educational threads."
    )
    with st.expander("Sample Discussion Threads"):
        threads = [
            {"title": "SPY 0DTE Strategy Discussion", "replies": 42, "last_post": "2 hours ago"},
            {"title": "Weekly Options Trading Tips", "replies": 18, "last_post": "5 hours ago"},
            {"title": "Volatility Analysis for Next Week", "replies": 7, "last_post": "1 day ago"},
            {"title": "Earnings Plays Discussion", "replies": 23, "last_post": "2 days ago"},
        ]
        for thread in threads:
            st.write(f"**{thread['title']}**")
            st.caption(f"Replies: {thread['replies']} | Last post: {thread['last_post']}")
            st.divider()


# =========================================================
# APP
# =========================================================
def render_app() -> None:
    st.title("📈 Options Analyzer Pro")
    st.markdown("**TradingView-Style Layout** • **Professional Analysis** • **Real-time Signals**")

    render_sidebar()

    ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="").upper().strip()
    if not ticker:
        st.info("👋 Enter a ticker above to begin.")
        return

    st_autorefresh(interval=1000, limit=None, key="price_refresh")

    tabs = st.tabs(["General", "Chart", "News & Analysis", "Financials", "Technical", "Forum"])

    df = pd.DataFrame()
    with tabs[0]:
        df = render_general_tab(ticker)
    with tabs[1]:
        render_chart_tab(ticker)
    with tabs[2]:
        render_news_tab(ticker, df)
    with tabs[3]:
        render_financials_tab(ticker)
    with tabs[4]:
        render_technical_tab(ticker)
    with tabs[5]:
        render_forum_tab()

    if st.session_state.get("auto_refresh_enabled", False) and ticker:
        current_time = time.time()
        elapsed = current_time - st.session_state.last_refresh
        min_interval = max(st.session_state.refresh_interval, CONFIG["MIN_REFRESH_INTERVAL"])

        if elapsed > min_interval:
            st.session_state.last_refresh = current_time
            st.session_state.refresh_counter += 1
            st.cache_data.clear()
            st.success(f"🔄 Auto-refreshed at {datetime.datetime.now().strftime('%H:%M:%S')}")
            time.sleep(0.5)
            st.rerun()

