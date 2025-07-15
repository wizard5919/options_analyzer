import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import requests
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from typing import Tuple
try:
    from polygon import RESTClient
except ImportError:
    st.error("Polygon API client not found. Please install with 'pip install polygon-api-client'.")
    RESTClient = None
try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    st.error("Alpha Vantage client not found. Please install with 'pip install alpha-vantage'.")
    TimeSeries = None
try:
    import iexfinance as iex
except ImportError:
    st.error("IEX Finance client not found. Please install with 'pip install iexfinance'.")
    iex = None
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(page_title="Options Greeks Buy Signal Analyzer", layout="wide")

# API Configuration (Replace with your actual keys)
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
ALPHA_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
IEX_API_KEY = "YOUR_IEX_CLOUD_API_KEY"

polygon_client = RESTClient(POLYGON_API_KEY) if RESTClient else None
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas') if TimeSeries else None

# Configuration
CONFIG = {
    'CACHE_TTL': 300,
    'RATE_LIMIT_COOLDOWN': 180,
}

SIGNAL_THRESHOLDS = {
    'call': {'delta_min': 0.6, 'gamma_min': 0.08, 'theta_max': 0.05, 'vega_min': 0.01, 'rsi_min': 50, 'volume_multiplier': 1.5},
    'put': {'delta_max': -0.6, 'gamma_min': 0.08, 'theta_max': 0.05, 'vega_min': 0.01, 'rsi_max': 50, 'volume_multiplier': 1.5}
}

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data_polygon(ticker: str) -> pd.DataFrame:
    if not polygon_client:
        st.error("Polygon client not initialized. Check API key and installation.")
        return pd.DataFrame()
    try:
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=10)
        data = polygon_client.get_aggs(ticker, 1, "minute", start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'), limit=50000)
        df = pd.DataFrame(data)
        if df.empty:
            st.warning(f"No data from Polygon for {ticker}")
            return pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        return df[['timestamp', 'Close', 'High', 'Low', 'Volume']]
    except Exception as e:
        st.error(f"Polygon error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data_alpha(ticker: str) -> pd.DataFrame:
    if not ts:
        st.error("Alpha Vantage client not initialized. Check API key and installation.")
        return pd.DataFrame()
    try:
        data, meta = ts.get_intraday(symbol=ticker, interval='1min', outputsize='full')
        if data.empty:
            st.warning(f"No data from Alpha Vantage for {ticker}")
            return pd.DataFrame()
        data.index = pd.to_datetime(data.index)
        return data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
    except Exception as e:
        st.error(f"Alpha Vantage error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data_iex(ticker: str) -> pd.DataFrame:
    if not iex:
        st.error("IEX Finance client not initialized. Check API key and installation.")
        return pd.DataFrame()
    try:
        data = iex.Stock(ticker, token=IEX_API_KEY).get_chart(range='1m', chartByDay=True)
        df = pd.DataFrame(data)
        if df.empty:
            st.warning(f"No data from IEX Cloud for {ticker}")
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        return df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
    except Exception as e:
        st.error(f"IEX Cloud error: {str(e)}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'Close' not in df.columns:
        return df
    df = df.copy()
    close = df['Close'].astype(float)
    if len(close) >= 9:
        df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
    if len(close) >= 20:
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
    if len(close) >= 14:
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP'] = vwap
    df['avg_vol'] = df['Volume'].rolling(window=min(20, len(df)), min_periods=1).mean()
    return df

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_data_polygon(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not polygon_client:
        st.error("Polygon client not initialized. Check API key and installation.")
        return pd.DataFrame(), pd.DataFrame()
    try:
        calls = pd.DataFrame(columns=['strike_price', 'delta', 'gamma', 'theta', 'vega', 'last_price', 'volume', 'open_interest'])
        puts = pd.DataFrame(columns=['strike_price', 'delta', 'gamma', 'theta', 'vega', 'last_price', 'volume', 'open_interest'])
        options = polygon_client.get_options_contracts(ticker, limit=1000)
        for opt in options:
            data = {
                'strike_price': opt.get('strike_price', np.nan),
                'delta': opt.get('greeks', {}).get('delta', np.nan),
                'gamma': opt.get('greeks', {}).get('gamma', np.nan),
                'theta': opt.get('greeks', {}).get('theta', np.nan),
                'vega': opt.get('greeks', {}).get('vega', np.nan),
                'last_price': opt.get('last_price', np.nan),
                'volume': opt.get('volume', 0),
                'open_interest': opt.get('open_interest', 0)
            }
            if opt.get('contract_type') == 'call':
                calls = pd.concat([calls, pd.DataFrame([data])], ignore_index=True)
            elif opt.get('contract_type') == 'put':
                puts = pd.concat([puts, pd.DataFrame([data])], ignore_index=True)
        return calls, puts
    except Exception as e:
        st.error(f"Polygon options error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame) -> Dict:
    if stock_df.empty or any(pd.isna(option.get(greek)) for greek in ['delta', 'gamma', 'theta', 'vega']):
        return {'signal': False, 'reason': 'Insufficient data'}
    latest = stock_df.iloc[-1]
    delta, gamma, theta, vega = [float(option.get(g, np.nan)) for g in ['delta', 'gamma', 'theta', 'vega']]
    close, ema_9, ema_20, rsi, vwap, volume, avg_vol = [float(latest.get(col, np.nan)) for col in ['Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'Volume', 'avg_vol']]
    thresholds = SIGNAL_THRESHOLDS[side]
    conditions = [
        (delta >= thresholds['delta_min'] if side == 'call' else delta <= thresholds['delta_max'], f"Delta {'>=' if side == 'call' else '<='} {thresholds['delta_min' if side == 'call' else 'delta_max']}", delta),
        (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']}", gamma),
        (theta <= thresholds['theta_max'], f"Theta <= {thresholds['theta_max']}", theta),
        (vega >= thresholds['vega_min'], f"Vega >= {thresholds['vega_min']}", vega),
        (rsi > thresholds['rsi_min'] if side == 'call' else rsi < thresholds['rsi_max'], f"RSI {'>' if side == 'call' else '<'} {thresholds['rsi_min' if side == 'call' else 'rsi_max']}", rsi),
        (volume > thresholds['volume_multiplier'] * avg_vol, f"Volume > {thresholds['volume_multiplier']}x avg", volume)
    ]
    signal = all(passed for passed, _, _ in conditions)
    return {'signal': signal, 'conditions': [(desc, val) for passed, desc, val in conditions]}

def fetch_headlines(ticker: str) -> str:
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return "\n".join([h.get_text() for h in soup.find_all('h3')[:3]])
    except Exception:
        return "Unable to fetch news."

def color_signal(val):
    return f"color: {'green' if val else 'red'}"

# Interface
st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown(f"**Real-time analysis as of {datetime.datetime.now().strftime('%I:%M %p %z on %B %d, %Y')}. Enhanced with Polygon.io, Alpha Vantage, and IEX Cloud.**")

if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = time.time()

if 'rate_limited_until' in st.session_state and time.time() < st.session_state['rate_limited_until']:
    st.warning(f"Rate limit active. Wait {int(st.session_state['rate_limited_until'] - time.time())}s.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuration")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    refresh_interval = st.selectbox("Refresh Interval", [60, 120, 300], format_func=lambda x: f"{x}s") if enable_auto_refresh else None
    for side in ['call', 'put']:
        st.subheader(f"**{side.capitalize()} Thresholds**")
        SIGNAL_THRESHOLDS[side]['delta_min' if side == 'call' else 'delta_max'] = st.slider(f"Min/Max Delta", -1.0, 1.0, SIGNAL_THRESHOLDS[side]['delta_min' if side == 'call' else 'delta_max'], 0.1)
        SIGNAL_THRESHOLDS[side]['gamma_min'] = st.slider("Min Gamma", 0.01, 0.2, SIGNAL_THRESHOLDS[side]['gamma_min'], 0.01)
        SIGNAL_THRESHOLDS[side]['theta_max'] = st.slider("Max Theta", 0.01, 0.1, SIGNAL_THRESHOLDS[side]['theta_max'], 0.01)
        SIGNAL_THRESHOLDS[side]['vega_min'] = st.slider("Min Vega", 0.01, 0.1, SIGNAL_THRESHOLDS[side]['vega_min'], 0.01)
        SIGNAL_THRESHOLDS[side]['rsi_min' if side == 'call' else 'rsi_max'] = st.slider(f"Min/Max RSI", 30, 70, SIGNAL_THRESHOLDS[side]['rsi_min' if side == 'call' else 'rsi_max'], 5)
        SIGNAL_THRESHOLDS[side]['volume_multiplier'] = st.slider("Volume Multiplier", 1.0, 3.0, SIGNAL_THRESHOLDS[side]['volume_multiplier'], 0.1)

ticker = st.text_input("Enter Ticker", value="QQQ").upper()
if ticker:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader(f"📊 {ticker} Analysis")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col3:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    if enable_auto_refresh and time.time() - st.session_state.last_auto_refresh >= refresh_interval:
        st.session_state.last_auto_refresh = time.time()
        st.session_state.refresh_counter += 1
        st.rerun()

    st.caption(f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🔄 {st.session_state.refresh_counter}")

    # Dashboard Layout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Signals Dashboard")
        # Aggregate data from multiple sources
        df_poly = get_stock_data_polygon(ticker)
        df_alpha = get_stock_data_alpha(ticker)
        df_iex = get_stock_data_iex(ticker)
        df = pd.concat([df for df in [df_poly, df_alpha, df_iex] if not df.empty]).drop_duplicates(subset=['timestamp' if 'timestamp' in df.columns else 'date']).sort_index()
        if df.empty:
            st.error("No stock data available from any source.")
        else:
            df = compute_indicators(df)
            current_price = df['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")

            calls, puts = get_options_data_polygon(ticker)
            if calls.empty and puts.empty:
                st.error("No options data available.")
            else:
                # Process all options to capture all opportunities
                for side, df_options in [('call', calls), ('put', puts)]:
                    if not df_options.empty:
                        signals = [generate_signal(row, side, df) for _, row in df_options.iterrows()]
                        signal_df = pd.DataFrame(signals, index=df_options.index)
                        signal_df['signal'] = signal_df['signal']
                        if not signal_df.empty:
                            st.dataframe(signal_df[signal_df['signal']][['signal'] + [c[0] for c in signal_df['conditions'].iloc[0]]].style.applymap(color_signal, subset=['signal']), use_container_width=True)
                            st.success(f"{sum(signal_df['signal'])} {side} signals found across all opportunities!")
                        else:
                            st.info(f"No {side} signals generated.")

    with col2:
        st.subheader("📈 Visualizations")
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9'))
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP'))
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap for signal strength (if signals exist)
            if 'signal_df' in locals() and not signal_df.empty:
                heat_data = signal_df[['signal'] + [c[0] for c in signal_df['conditions'].iloc[0]]].mean()
                fig_heat = go.Figure(data=go.Heatmap(z=heat_data.values, x=heat_data.index, colorscale='Viridis'))
                st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(fetch_headlines(ticker))
    if st.button("📩 Feedback"):
        st.success("Feedback noted!")

else:
    st.info("Enter a ticker to start.")
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        **Steps:**
        1. Enter a ticker (e.g., QQQ, SPY)
        2. Adjust thresholds in sidebar
        3. Review signals for all opportunities
        **Signals:** Based on Delta, Gamma, Theta, Vega, RSI, Volume
        **Note:** Install required packages (polygon-api-client, alpha-vantage, iexfinance) using 'pip install'.
        """)
