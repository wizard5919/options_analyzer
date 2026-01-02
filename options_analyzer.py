
import os
import math
import time
import json
import warnings
import datetime as dt
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Technical indicators (ta)
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, KeltnerChannel

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional: scipy for faster/accurate normal CDF/PDF
try:
    from scipy.stats import norm
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    norm = None

# Optional: st_autorefresh (prevents TypeError / missing import issues)
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
except Exception:
    _st_autorefresh = None

warnings.filterwarnings("ignore")


# -------------------------
# Configuration
# -------------------------
CONFIG: Dict[str, Any] = {
    "APP": {
        "TITLE": "Options Analyzer Pro",
        "LAYOUT": "wide",
        "DEFAULT_TICKER": "QQQ",
        "DEFAULT_TIMEFRAME": "15m",
        "DEFAULT_PERIOD": "5d",
        "MIN_REFRESH_SECONDS": 30,      # safety: avoid API hammering
        "DEFAULT_REFRESH_SECONDS": 60,  # sane default
    },
    "DATA": {
        "YF_TIMEOUT": 12,
        "REQUESTS_TIMEOUT": 12,
        "CACHE_TTL_PRICE": 60,     # seconds
        "CACHE_TTL_CHAIN": 120,    # seconds
        "CACHE_TTL_NEWS": 300,     # seconds
    },
    # FIXED: now at top-level, not nested (this was breaking your code)
    "LIQUIDITY_THRESHOLDS": {
        "min_open_interest": 50,
        "min_volume": 50,
        "max_bid_ask_spread_pct": 0.15,  # 15%
    },
    # FIXED: actual sensitivity map (your code was calling .get(timeframe) on wrong dict)
    "SR_SENSITIVITY_MAP": {
        "5m": 0.003,
        "15m": 0.004,
        "30m": 0.005,
        "60m": 0.006,
        "1h": 0.006,
        "2h": 0.007,
        "4h": 0.008,
        "1d": 0.010,
        "daily": 0.010,
    },
    "SR_WINDOW_SIZES": {
        "5m": 40,
        "15m": 60,
        "30m": 80,
        "60m": 120,
        "1h": 120,
        "2h": 160,
        "4h": 220,
        "1d": 200,
        "daily": 200,
    },
    "SIGNALS": {
        "MIN_SCORE_TO_SHOW": 60,
        "MAX_CONTRACTS_SHOWN": 25,
    }
}


# -------------------------
# Secrets / Credentials
# -------------------------
def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    # prefers Streamlit secrets, falls back to env vars
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def login_gate() -> bool:
    """
    A) REAL login gate:
    - If not authenticated, show login UI and stop.
    - No other app code should run before this returns True.
    """
    st.session_state.setdefault("authenticated", False)

    # Already logged in
    if st.session_state.authenticated:
        return True

    st.title("🔒 Login to Options Analyzer Pro")

    # Recommended: put these in .streamlit/secrets.toml
    # APP_USER="..."
    # APP_PASS="..."
    valid_user = _get_secret("APP_USER", "")
    valid_pass = _get_secret("APP_PASS", "")

    if not valid_user or not valid_pass:
        st.warning(
            "Login is not configured. Add APP_USER and APP_PASS to Streamlit secrets "
            "(.streamlit/secrets.toml) or environment variables."
        )
        st.stop()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Login", use_container_width=True):
            if username == valid_user and password == valid_pass:
                st.session_state.authenticated = True
                st.success("✅ Logged in.")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
    with col2:
        st.caption("Tip: store credentials in Streamlit secrets — never hardcode.")

    return False


# -------------------------
# Safe autorefresh wrapper
# -------------------------
def safe_autorefresh(seconds: int, key: str) -> None:
    """
    C) stable refresh:
    - only runs if streamlit-autorefresh is installed
    - enforce minimum interval
    """
    if _st_autorefresh is None:
        return

    seconds = int(max(CONFIG["APP"]["MIN_REFRESH_SECONDS"], seconds))
    # streamlit-autorefresh expects milliseconds
    _st_autorefresh(interval=seconds * 1000, limit=None, key=key)


# -------------------------
# Cached data fetchers
# -------------------------
@st.cache_data(ttl=CONFIG["DATA"]["CACHE_TTL_PRICE"])
def fetch_price_history(ticker: str, timeframe: str, period: str, cache_buster: int) -> pd.DataFrame:
    """
    C) caching is controlled via TTL + cache_buster, no global clears.
    """
    t = yf.Ticker(ticker)
    # yfinance uses interval: '1m','2m','5m','15m','30m','60m','90m','1h','1d',...
    df = t.history(period=period, interval=timeframe, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Normalize columns
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Time"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "Time"}, inplace=True)
    return df


@st.cache_data(ttl=CONFIG["DATA"]["CACHE_TTL_CHAIN"])
def fetch_option_chain(ticker: str, expiry: str, cache_buster: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (calls_df, puts_df) from yfinance.
    """
    t = yf.Ticker(ticker)
    chain = t.option_chain(expiry)
    calls = chain.calls.copy() if chain and hasattr(chain, "calls") else pd.DataFrame()
    puts = chain.puts.copy() if chain and hasattr(chain, "puts") else pd.DataFrame()
    return calls, puts


@st.cache_data(ttl=CONFIG["DATA"]["CACHE_TTL_NEWS"])
def fetch_news_unified(ticker: str, cache_buster: int) -> List[Dict[str, Any]]:
    """
    D) unified news:
    1) Benzinga (if BENZINGA_API_KEY)
    2) AlphaVantage (if ALPHAVANTAGE_API_KEY)
    3) Yahoo via yfinance .news
    """
    timeout = CONFIG["DATA"]["REQUESTS_TIMEOUT"]

    benz_key = _get_secret("BENZINGA_API_KEY", "")
    av_key = _get_secret("ALPHAVANTAGE_API_KEY", "")

    # 1) Benzinga
    if benz_key:
        try:
            url = "https://api.benzinga.com/api/v2/news"
            params = {
                "token": benz_key,
                "symbols": ticker,
                "pagesize": 20,
                "displayOutput": "full",
                "sort": "updated",
            }
            r = requests.get(url, params=params, timeout=timeout)
            if r.ok:
                data = r.json()
                items = data if isinstance(data, list) else data.get("news", [])
                out = []
                for it in items[:20]:
                    out.append({
                        "source": "Benzinga",
                        "title": it.get("title") or "Untitled",
                        "url": it.get("url") or "",
                        "published": it.get("created") or it.get("updated") or "",
                        "summary": it.get("teaser") or it.get("content") or "",
                    })
                if out:
                    return out
        except Exception:
            pass

    # 2) AlphaVantage
    if av_key:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "apikey": av_key,
                "limit": 20,
            }
            r = requests.get(url, params=params, timeout=timeout)
            if r.ok:
                data = r.json()
                feed = data.get("feed", [])
                out = []
                for it in feed[:20]:
                    out.append({
                        "source": "AlphaVantage",
                        "title": it.get("title") or "Untitled",
                        "url": it.get("url") or "",
                        "published": it.get("time_published") or "",
                        "summary": it.get("summary") or "",
                    })
                if out:
                    return out
        except Exception:
            pass

    # 3) Yahoo fallback
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", None) or []
        out = []
        for it in news[:20]:
            out.append({
                "source": "Yahoo",
                "title": it.get("title") or "Untitled",
                "url": it.get("link") or "",
                "published": it.get("providerPublishTime") or "",
                "summary": it.get("publisher") or "",
            })
        return out
    except Exception:
        return []


# -------------------------
# Indicators / Helpers
# -------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    d = df.copy()
    # Ensure numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    close = d["Close"]
    high = d["High"]
    low = d["Low"]

    # EMAs
    d["EMA9"] = EMAIndicator(close=close, window=9).ema_indicator()
    d["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
    d["EMA50"] = EMAIndicator(close=close, window=50).ema_indicator()
    d["EMA200"] = EMAIndicator(close=close, window=200).ema_indicator()

    # RSI / MACD
    d["RSI"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    d["MACD"] = macd.macd()
    d["MACDSignal"] = macd.macd_signal()
    d["MACDHist"] = macd.macd_diff()

    # ATR / Keltner
    d["ATR"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    kc = KeltnerChannel(high=high, low=low, close=close, window=20, window_atr=10, original_version=False)
    d["KC_MID"] = kc.keltner_channel_mband()
    d["KC_UP"] = kc.keltner_channel_hband()
    d["KC_LOW"] = kc.keltner_channel_lband()

    # VWAP (simple session-based approximation)
    tp = (high + low + close) / 3.0
    v = d["Volume"].replace(0, np.nan)
    d["VWAP"] = (tp.mul(v).cumsum() / v.cumsum()).fillna(method="ffill")

    return d


def compute_sr_levels(df: pd.DataFrame, timeframe: str) -> Dict[str, float]:
    """
    B) clean SR:
    - use rolling high/low window based on timeframe
    - return only key levels (recent swing high/low)
    """
    if df is None or df.empty:
        return {}

    w = int(CONFIG["SR_WINDOW_SIZES"].get(timeframe, 80))
    d = df.copy()
    if "High" not in d or "Low" not in d:
        return {}

    d["swing_high"] = d["High"].rolling(window=w, min_periods=max(10, w // 4)).max()
    d["swing_low"] = d["Low"].rolling(window=w, min_periods=max(10, w // 4)).min()

    last = d.dropna().iloc[-1] if not d.dropna().empty else None
    if last is None:
        return {}

    return {
        "resistance": float(last["swing_high"]),
        "support": float(last["swing_low"]),
    }


# -------------------------
# Black-Scholes Greeks (approx)
# -------------------------
def _norm_cdf(x: float) -> float:
    if SCIPY_OK and norm is not None:
        return float(norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    if SCIPY_OK and norm is not None:
        return float(norm.pdf(x))
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> Dict[str, float]:
    """
    Computes Delta/Gamma/Theta (per day) approximate.
    - T in years
    - sigma in decimal
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan}

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    pdf = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)

    if option_type.lower() == "call":
        delta = cdf_d1
        theta = (
            -(S * pdf * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * cdf_d2
        )
    else:
        delta = cdf_d1 - 1.0
        theta = (
            -(S * pdf * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        )

    gamma = pdf / (S * sigma * math.sqrt(T))

    # theta per day (rough)
    theta_per_day = theta / 365.0
    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta_per_day)}


def enrich_chain_with_greeks(chain: pd.DataFrame, spot: float, expiry: str, side: str) -> pd.DataFrame:
    """
    Adds columns delta/gamma/theta if missing (approx).
    Uses impliedVolatility if available, else a conservative fallback.
    """
    if chain is None or chain.empty:
        return pd.DataFrame()

    d = chain.copy()

    # time to expiry (at market close approx)
    try:
        exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d")
        now = dt.datetime.now()
        T = max(1e-6, (exp_date - now).total_seconds() / (365.0 * 24 * 3600))
    except Exception:
        T = 7 / 365.0

    r = 0.02  # simple constant; you can replace with live risk-free if you want

    if "impliedVolatility" in d.columns:
        iv = pd.to_numeric(d["impliedVolatility"], errors="coerce").fillna(0.35)
        iv = iv.clip(lower=0.05, upper=3.0)
    else:
        iv = pd.Series([0.35] * len(d), index=d.index)

    # Compute greeks row-wise (fast enough after liquidity filters)
    deltas, gammas, thetas = [], [], []
    for i, row in d.iterrows():
        K = float(row.get("strike", np.nan))
        sig = float(iv.loc[i]) if i in iv.index else 0.35
        g = bs_greeks(spot, K, T, r, sig, option_type=side)
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        thetas.append(g["theta"])

    d["delta"] = deltas
    d["gamma"] = gammas
    d["theta"] = thetas
    d["side"] = side
    d["expiry"] = expiry
    return d


def liquidity_filter(chain: pd.DataFrame) -> pd.DataFrame:
    if chain is None or chain.empty:
        return pd.DataFrame()

    liq = CONFIG["LIQUIDITY_THRESHOLDS"]
    d = chain.copy()

    # Ensure numeric
    for c in ["openInterest", "volume", "bid", "ask", "lastPrice", "strike"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["mid"] = (d["bid"] + d["ask"]) / 2.0
    d["spread_pct"] = np.where(
        d["mid"] > 0, (d["ask"] - d["bid"]) / d["mid"], np.nan
    )

    d = d[
        (d["openInterest"].fillna(0) >= liq["min_open_interest"])
        & (d["volume"].fillna(0) >= liq["min_volume"])
        & (d["spread_pct"].fillna(1) <= liq["max_bid_ask_spread_pct"])
    ].copy()

    return d


def score_contract(row: pd.Series, trend: Dict[str, float], side: str) -> int:
    """
    Simple “institutional-ish” scoring:
    - Liquidity (OI/Vol/spread)
    - Greeks (prefer higher gamma, reasonable theta)
    - Trend alignment (EMA/VWAP/RSI)
    Score: 0..100
    """
    score = 0

    # Liquidity
    oi = float(row.get("openInterest", 0) or 0)
    vol = float(row.get("volume", 0) or 0)
    spread = float(row.get("spread_pct", 1) if row.get("spread_pct", None) is not None else 1)

    score += min(25, int(np.log1p(oi) * 4))
    score += min(20, int(np.log1p(vol) * 4))
    score += max(0, int(15 * (1 - min(spread, 1.0))))  # tighter spread => higher

    # Greeks
    delta = float(row.get("delta", np.nan))
    gamma = float(row.get("gamma", np.nan))
    theta = float(row.get("theta", np.nan))

    if not np.isnan(delta):
        # prefer delta in 0.25-0.55 for calls, -0.55 to -0.25 for puts (scalping-friendly)
        if side == "call":
            score += 15 if 0.25 <= delta <= 0.60 else 6
        else:
            score += 15 if -0.60 <= delta <= -0.25 else 6
    if not np.isnan(gamma):
        score += min(10, int(gamma * 500))  # gamma scaling
    if not np.isnan(theta):
        # theta is per day (negative). less negative is better for holding; for scalp we still prefer not too ugly
        score += 10 if theta >= -0.20 else 4

    # Trend alignment
    # trend dict: {"bias": "bull"/"bear", "rsi":..., "above_vwap":...}
    if trend.get("bias") == ("bull" if side == "call" else "bear"):
        score += 15
    if trend.get("above_vwap") is True and side == "call":
        score += 5
    if trend.get("above_vwap") is False and side == "put":
        score += 5

    rsi = trend.get("rsi")
    if isinstance(rsi, (int, float)) and not np.isnan(rsi):
        if side == "call":
            score += 5 if rsi >= 50 else 0
        else:
            score += 5 if rsi <= 50 else 0

    return int(min(100, max(0, score)))


def derive_trend(last_row: pd.Series) -> Dict[str, Any]:
    """
    Determines trend bias from EMA stack + VWAP + MACD histogram.
    """
    ema9 = last_row.get("EMA9", np.nan)
    ema20 = last_row.get("EMA20", np.nan)
    ema50 = last_row.get("EMA50", np.nan)
    vwap = last_row.get("VWAP", np.nan)
    close = last_row.get("Close", np.nan)
    rsi = last_row.get("RSI", np.nan)
    macdh = last_row.get("MACDHist", np.nan)

    bull_stack = (close > ema9 > ema20 > ema50) if all(map(np.isfinite, [close, ema9, ema20, ema50])) else False
    bear_stack = (close < ema9 < ema20 < ema50) if all(map(np.isfinite, [close, ema9, ema20, ema50])) else False

    if bull_stack and (np.isfinite(macdh) and macdh >= 0):
        bias = "bull"
    elif bear_stack and (np.isfinite(macdh) and macdh <= 0):
        bias = "bear"
    else:
        bias = "neutral"

    above_vwap = (close > vwap) if all(map(np.isfinite, [close, vwap])) else None

    return {
        "bias": bias,
        "rsi": float(rsi) if np.isfinite(rsi) else np.nan,
        "above_vwap": above_vwap,
    }


def plot_price(df: pd.DataFrame, sr: Dict[str, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Time"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    for col in ["EMA9", "EMA20", "EMA50", "EMA200", "VWAP"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Time"], y=df[col], mode="lines", name=col))

    if sr.get("support") is not None:
        fig.add_hline(y=sr["support"], line_dash="dot", annotation_text=f"Support {sr['support']:.2f}")
    if sr.get("resistance") is not None:
        fig.add_hline(y=sr["resistance"], line_dash="dot", annotation_text=f"Resistance {sr['resistance']:.2f}")

    fig.update_layout(height=520, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -------------------------
# Main App
# -------------------------
def run_app():
    # Session defaults
    st.session_state.setdefault("cache_buster", 0)
    st.session_state.setdefault("auto_refresh_enabled", False)
    st.session_state.setdefault("refresh_seconds", CONFIG["APP"]["DEFAULT_REFRESH_SECONDS"])

    # Sidebar
    st.sidebar.header("⚙️ Controls")

    ticker = st.sidebar.text_input("Ticker", value=CONFIG["APP"]["DEFAULT_TICKER"]).strip().upper()

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=["5m", "15m", "30m", "60m", "1h", "2h", "4h", "1d"],
        index=1
    )

    # map 60m <-> 1h for yfinance
    yf_interval = timeframe
    if timeframe == "1h":
        yf_interval = "60m"
    elif timeframe == "2h":
        yf_interval = "120m"
    elif timeframe == "4h":
        yf_interval = "240m"

    period = st.sidebar.selectbox("Period", options=["1d", "5d", "1mo", "3mo"], index=1)

    st.sidebar.divider()

    st.session_state.auto_refresh_enabled = st.sidebar.toggle("Auto Refresh", value=st.session_state.auto_refresh_enabled)
    st.session_state.refresh_seconds = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=CONFIG["APP"]["MIN_REFRESH_SECONDS"],
        max_value=600,
        value=int(st.session_state.refresh_seconds),
        step=10
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.cache_buster += 1
            st.rerun()
    with c2:
        if st.button("🧹 Reset Session", use_container_width=True):
            for k in ["cache_buster", "auto_refresh_enabled", "refresh_seconds"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # C) auto refresh AFTER login + AFTER ticker exists
    if st.session_state.auto_refresh_enabled and ticker:
        safe_autorefresh(st.session_state.refresh_seconds, key="price_refresh")

    # Header
    st.title("📈 Options Analyzer Pro")
    st.caption("Greeks + Liquidity + Trend Filters (designed for scalping calls/puts).")

    # Load price data
    with st.spinner("Fetching price data..."):
        price_df = fetch_price_history(ticker, yf_interval, period, st.session_state.cache_buster)

    if price_df.empty:
        st.error("No price data returned. Check the ticker or try a different timeframe/period.")
        st.stop()

    price_df = add_indicators(price_df)
    sr = compute_sr_levels(price_df, timeframe=yf_interval)

    last = price_df.dropna().iloc[-1] if not price_df.dropna().empty else price_df.iloc[-1]
    spot = float(last["Close"])
    trend = derive_trend(last)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Chart", "🧠 Options Scanner", "📰 News"])

    with tab1:
        st.subheader(f"{ticker} — {timeframe} | Last: {spot:.2f}")
        st.write(f"**Trend Bias:** `{trend['bias']}` | **RSI:** `{trend.get('rsi', np.nan):.1f}`")
        fig = plot_price(price_df, sr)
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.metric("Support", f"{sr.get('support', np.nan):.2f}" if sr else "—")
        with cols[1]:
            st.metric("Resistance", f"{sr.get('resistance', np.nan):.2f}" if sr else "—")

    with tab2:
        st.subheader("Options Scanner")

        # expiries
        try:
            t = yf.Ticker(ticker)
            expiries = list(getattr(t, "options", []) or [])
        except Exception:
            expiries = []

        if not expiries:
            st.error("No option expirations found for this ticker.")
            st.stop()

        expiry = st.selectbox("Expiration", options=expiries, index=0)

        # strike range filter
        pct = st.slider("Strike Range around spot (%)", 5, 50, 20, 5)
        low_strike = spot * (1 - pct / 100.0)
        high_strike = spot * (1 + pct / 100.0)

        side_choice = st.radio("Side", ["Calls", "Puts", "Both"], horizontal=True)

        liq = CONFIG["LIQUIDITY_THRESHOLDS"]
        st.caption(
            f"Liquidity filters: OI≥{liq['min_open_interest']} | Vol≥{liq['min_volume']} | Spread≤{int(liq['max_bid_ask_spread_pct']*100)}%"
        )

        with st.spinner("Fetching option chain..."):
            calls_raw, puts_raw = fetch_option_chain(ticker, expiry, st.session_state.cache_buster)

        calls = liquidity_filter(calls_raw)
        puts = liquidity_filter(puts_raw)

        # strike filter
        if not calls.empty:
            calls = calls[(calls["strike"] >= low_strike) & (calls["strike"] <= high_strike)].copy()
        if not puts.empty:
            puts = puts[(puts["strike"] >= low_strike) & (puts["strike"] <= high_strike)].copy()

        # enrich greeks (approx)
        if side_choice in ("Calls", "Both") and not calls.empty:
            calls = enrich_chain_with_greeks(calls, spot=spot, expiry=expiry, side="call")
        if side_choice in ("Puts", "Both") and not puts.empty:
            puts = enrich_chain_with_greeks(puts, spot=spot, expiry=expiry, side="put")

        # build scored table
        rows = []
        if side_choice in ("Calls", "Both") and not calls.empty:
            for _, r in calls.iterrows():
                rows.append(r.to_dict())
        if side_choice in ("Puts", "Both") and not puts.empty:
            for _, r in puts.iterrows():
                rows.append(r.to_dict())

        if not rows:
            st.warning("No contracts pass filters. Try widening strike range or lowering liquidity thresholds.")
            st.stop()

        df_all = pd.DataFrame(rows)

        # score
        scores = []
        for _, r in df_all.iterrows():
            scores.append(score_contract(r, trend=trend, side=r.get("side", "")))
        df_all["score"] = scores

        # label idea
        def label(row):
            if row["side"] == "call" and trend["bias"] == "bull" and row["score"] >= 75:
                return "LONG CALL BREAKOUT"
            if row["side"] == "put" and trend["bias"] == "bear" and row["score"] >= 75:
                return "LONG PUT BREAKDOWN"
            return ""

        df_all["label"] = df_all.apply(label, axis=1)

        # sort and show
        df_all = df_all.sort_values(["score", "openInterest", "volume"], ascending=[False, False, False]).head(
            CONFIG["SIGNALS"]["MAX_CONTRACTS_SHOWN"]
        )

        min_score = st.slider("Minimum score to display", 0, 100, CONFIG["SIGNALS"]["MIN_SCORE_TO_SHOW"], 5)
        shown = df_all[df_all["score"] >= min_score].copy()

        st.write(f"Showing **{len(shown)}** contracts (min score {min_score}).")

        # pretty columns
        keep_cols = [
            "side", "expiry", "contractSymbol", "strike",
            "lastPrice", "bid", "ask", "mid", "spread_pct",
            "volume", "openInterest",
            "impliedVolatility", "delta", "gamma", "theta",
            "score", "label"
        ]
        for c in keep_cols:
            if c not in shown.columns:
                shown[c] = np.nan

        shown["spread_pct"] = (shown["spread_pct"] * 100.0).round(2)
        shown["impliedVolatility"] = (shown["impliedVolatility"] * 100.0).round(2)

        st.dataframe(
            shown[keep_cols].rename(columns={
                "spread_pct": "spread_%", "impliedVolatility": "IV_%"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Greeks are approximations (Black-Scholes using impliedVolatility when available). "
            "For scalping, focus on tight spreads + strong liquidity + trend alignment."
        )

    with tab3:
        st.subheader("Market News & Analysis")
        st.caption("Sources: Benzinga → AlphaVantage → Yahoo (fallback)")

        with st.spinner("Fetching news..."):
            news = fetch_news_unified(ticker, st.session_state.cache_buster)

        if not news:
            st.warning("No news found (or API keys missing). Add BENZINGA_API_KEY and/or ALPHAVANTAGE_API_KEY in secrets for richer feed.")
        else:
            for it in news[:15]:
                title = it.get("title", "Untitled")
                url = it.get("url", "")
                src = it.get("source", "")
                published = it.get("published", "")
                summary = it.get("summary", "")

                st.markdown(f"**{title}**")
                if url:
                    st.markdown(f"- Source: `{src}` | [Open article]({url})")
                else:
                    st.markdown(f"- Source: `{src}`")
                if published:
                    st.caption(str(published))
                if summary:
                    st.write(summary[:400] + ("..." if len(summary) > 400 else ""))
                st.divider()


def main():
    st.set_page_config(
        page_title=CONFIG["APP"]["TITLE"],
        layout=CONFIG["APP"]["LAYOUT"],
    )

    # A) login wall blocks everything below
    if not login_gate():
        st.stop()

    run_app()


if __name__ == "__main__":
    main()
