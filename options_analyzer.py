import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import warnings
from typing import Optional, Tuple, Dict, List
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests

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
}

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_min': 0.6,
        'gamma_min': 0.08,
        'theta_max': 0.05,
        'rsi_min': 50,
        'volume_multiplier': 1.5
    },
    'put': {
        'delta_max': -0.6,
        'gamma_min': 0.08,
        'theta_max': 0.05,
        'rsi_max': 50,
        'volume_multiplier': 1.5
    }
}

# =============================
# UTILITY FUNCTIONS
# =============================

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    """Safely call API functions with retry logic"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
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
def get_stock_data(ticker: str, days: int = 10) -> pd.DataFrame:
    """Fetch stock data with extended hours using prepost=True"""
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)
        
        data = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval="1m",
            prepost=True,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        data = data.dropna(how='all')
        for col in required_cols:
            if col in data.columns:
                if hasattr(data[col].iloc[0], '__len__') and not isinstance(data[col].iloc[0], str):
                    data[col] = data[col].apply(lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x)
                data[col] = pd.to_numeric(data[col], errors='coerce')

        data = data.dropna(subset=required_cols)
        
        if len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient data points ({len(data)}). Need at least {CONFIG['MIN_DATA_POINTS']}.")
            return pd.DataFrame()
        
        return data.reset_index()
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators with comprehensive error handling"""
    if df.empty:
        return df
    
    try:
        df = df.copy()
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
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
            ema_9 = EMAIndicator(close=close, window=9)
            df['EMA_9'] = ema_9.ema_indicator()
        else:
            df['EMA_9'] = np.nan
            
        if len(close) >= 20:
            ema_20 = EMAIndicator(close=close, window=20)
            df['EMA_20'] = ema_20.ema_indicator()
        else:
            df['EMA_20'] = np.nan
            
        if len(close) >= 14:
            rsi = RSIIndicator(close=close, window=14)
            df['RSI'] = rsi.rsi()
        else:
            df['RSI'] = np.nan

        typical_price = (high + low + close) / 3
        vwap_cumsum = (volume * typical_price).cumsum()
        volume_cumsum = volume.cumsum()
        df['VWAP'] = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
        
        window_size = min(20, len(volume))
        if window_size > 1:
            df['avg_vol'] = volume.rolling(window=window_size, min_periods=1).mean()
        else:
            df['avg_vol'] = volume.mean()
            
        return df
        
    except Exception as e:
        st.error(f"Error in compute_indicators: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> List[str]:
    """Get options expiries with error handling"""
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
    """Fetch options data with comprehensive error handling"""
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
            
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            
            required_cols = ['strike', 'lastPrice', 'volume', 'openInterest']
            
            for df_name, df in [('calls', calls), ('puts', puts)]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing columns in {df_name} for {expiry}: {missing_cols}")
                    continue
            
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
            
            time.sleep(1)
            
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

def classify_moneyness(strike: float, spot: float, tolerance: float = 0.01) -> str:
    """Classify option moneyness with tolerance"""
    ratio = strike / spot
    if ratio < (1 - tolerance):
        return 'ITM'
    elif ratio > (1 + tolerance):
        return 'OTM'
    else:
        return 'ATM'

def validate_option_data(option: pd.Series) -> bool:
    """Validate that option has required data for analysis"""
    required_fields = ['delta', 'gamma', 'theta', 'strike', 'lastPrice', 'volume', 'openInterest']
    
    for field in required_fields:
        if field not in option or pd.isna(option[field]):
            return False
    
    if option['lastPrice'] <= 0:
        return False
    
    return True

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame) -> Dict:
    """Generate trading signal with detailed analysis and ROI"""
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available'}
    
    if not validate_option_data(option):
        return {'signal': False, 'reason': 'Insufficient option data'}
    
    latest = stock_df.iloc[-1]
    
    try:
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume
        
        thresholds = SIGNAL_THRESHOLDS[side]
        
        conditions = []
        
        if side == "call":
            conditions = [
                (delta >= thresholds['delta_min'], f"Delta >= {thresholds['delta_min']}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']}", gamma),
                (theta <= thresholds['theta_max'], f"Theta <= {thresholds['theta_max']}", theta),
                (ema_9 is not None and ema_20 is not None and close > ema_9 > ema_20, "Price > EMA9 > EMA20", f"{close:.2f} > {ema_9:.2f} > {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi > thresholds['rsi_min'], f"RSI > {thresholds['rsi_min']}", rsi),
                (vwap is not None and close > vwap, "Price > VWAP", f"{close:.2f} > {vwap:.2f}" if vwap else "N/A"),
                (volume > thresholds['volume_multiplier'] * avg_vol, f"Volume > {thresholds['volume_multiplier']}x avg", f"{volume:.0f} > {avg_vol:.0f}")
            ]
            break_even = option['strike'] + option['lastPrice']
        else:
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']}", gamma),
                (theta <= thresholds['theta_max'], f"Theta <= {thresholds['theta_max']}", theta),
                (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']}", rsi),
                (vwap is not None and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (volume > thresholds['volume_multiplier'] * avg_vol, f"Volume > {thresholds['volume_multiplier']}x avg", f"{volume:.0f} > {avg_vol:.0f}")
            ]
            break_even = option['strike'] - option['lastPrice']
        
        passed_conditions = [desc for passed, desc, val in conditions if passed]
        failed_conditions = [f"{desc} (got {val})" for passed, desc, val in conditions if not passed]
        
        signal = all(passed for passed, desc, val in conditions)
        
        # Calculate ROI
        potential_return = ((option['lastPrice'] - break_even) / break_even) * 100 if break_even > 0 else 0
        
        return {
            'signal': signal,
            'passed_conditions': passed_conditions,
            'failed_conditions': failed_conditions,
            'score': len(passed_conditions) / len(conditions),
            'estimated_roi': potential_return
        }
        
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}'}

def fetch_headlines(ticker: str) -> str:
    """Fetch recent news headlines for the ticker"""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3')
        return "\n".join([h.get_text() for h in headlines[:3]])
    except Exception:
        return "Unable to fetch news at this time."

def color_roi(val):
    """Color-code ROI values"""
    color = "green" if val > 0 else "red"
    return f"color: {color}"

# =============================
# STREAMLIT INTERFACE
# =============================

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Enhanced robust version** with comprehensive error handling, detailed analysis, and real-time refresh capabilities.")

if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = time.time()

if 'rate_limited_until' in st.session_state:
    if time.time() < st.session_state['rate_limited_until']:
        remaining = int(st.session_state['rate_limited_until'] - time.time())
        st.warning(f"Yahoo Finance API rate limited. Please wait {remaining} seconds before retrying.")
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

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🔄 Auto-Refresh Settings")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    
    if enable_auto_refresh:
        min_interval = 60
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[60, 120, 300],
            index=0,
            format_func=lambda x: f"{x} seconds"
        )
        st.info(f"Data will refresh every {refresh_interval} seconds (minimum enforced)")
        if refresh_interval < 300:
            st.warning("Frequent auto-refreshes may lead to Yahoo Finance rate limiting. Consider increasing the refresh interval.")
    else:
        refresh_interval = None
    
    st.subheader("Signal Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Calls**")
        SIGNAL_THRESHOLDS['call']['delta_min'] = st.slider("Min Delta", 0.1, 1.0, 0.6, 0.1)
        SIGNAL_THRESHOLDS['call']['gamma_min'] = st.slider("Min Gamma", 0.01, 0.2, 0.08, 0.01)
        SIGNAL_THRESHOLDS['call']['rsi_min'] = st.slider("Min RSI", 30, 70, 50, 5)
    
    with col2:
        st.write("**Puts**")
        SIGNAL_THRESHOLDS['put']['delta_max'] = st.slider("Max Delta", -1.0, -0.1, -0.6, 0.1)
        SIGNAL_THRESHOLDS['put']['gamma_min'] = st.slider("Min Gamma ", 0.01, 0.2, 0.08, 0.01)
        SIGNAL_THRESHOLDS['put']['rsi_max'] = st.slider("Max RSI", 30, 70, 50, 5)
    
    SIGNAL_THRESHOLDS['call']['theta_max'] = SIGNAL_THRESHOLDS['put']['theta_max'] = st.slider("Max Theta", 0.01, 0.1, 0.05, 0.01)
    SIGNAL_THRESHOLDS['call']['volume_multiplier'] = SIGNAL_THRESHOLDS['put']['volume_multiplier'] = st.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)

ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="QQQ").upper()

if ticker:
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.subheader(f"📊 {ticker} Options Analysis")
    
    with col2:
        manual_refresh = st.button("🔄 Refresh Now")
    
    with col3:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    with col4:
        if enable_auto_refresh:
            current_time = time.time()
            time_elapsed = current_time - st.session_state.last_auto_refresh
            remaining = max(0, refresh_interval - int(time_elapsed))
            if remaining > 0:
                st.info(f"⏱️ {remaining}s")
            else:
                st.success("🔄 Refreshing...")

    if enable_auto_refresh:
        current_time = time.time()
        time_elapsed = current_time - st.session_state.last_auto_refresh
        if time_elapsed >= refresh_interval:
            st.session_state.last_auto_refresh = current_time
            st.session_state.refresh_counter += 1
            st.rerun()
    
    if manual_refresh:
        st.cache_data.clear()
        st.session_state.last_auto_refresh = time.time()
        st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"📅 Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")

    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Stock Data", "⚙️ Analysis Details"])
    
    with tab1:
        with st.spinner("Fetching and analyzing data..."):
            df = get_stock_data(ticker)
            
            if df.empty:
                st.error("Unable to fetch stock data. Please check the ticker symbol.")
                st.stop()
            
            df = compute_indicators(df)
            
            if df.empty:
                st.error("Unable to compute technical indicators.")
                st.stop()
            
            current_price = df.iloc[-1]['Close']
            st.success(f"✅ **{ticker}** - Current Price: **${current_price:.2f}**")
            
            expiries = get_options_expiries(ticker)
            
            if not expiries:
                st.error("No options expiries available for this ticker.")
                st.stop()
            
            expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"])
            
            today = datetime.date.today()
            if expiry_mode == "0DTE Only":
                expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
            else:
                expiries_to_use = expiries[:5]
            
            if not expiries_to_use:
                st.warning("No options expiries available for the selected mode.")
                st.stop()
            
            st.info(f"Analyzing {len(expiries_to_use)} expiries: {', '.join(expiries_to_use)}")
            
            calls, puts = fetch_options_data(ticker, expiries_to_use)
            
            if calls.empty and puts.empty:
                st.error("No options data available.")
                st.stop()
            
            strike_range = st.slider("Strike Range Around Current Price ($):", -50, 50, (-10, 10), 1)
            min_strike = current_price + strike_range[0]
            max_strike = current_price + strike_range[1]
            
            calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)].copy()
            puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)].copy()
            
            if not calls_filtered.empty:
                calls_filtered['moneyness'] = calls_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
            if not puts_filtered.empty:
                puts_filtered['moneyness'] = puts_filtered['strike'].apply(lambda x: classify_moneyness(x, current_price))
            
            m_filter = st.multiselect("Filter by Moneyness:", options=["ITM", "ATM", "OTM"], default=["ITM", "ATM", "OTM"])
            
            if not calls_filtered.empty:
                calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
            if not puts_filtered.empty:
                puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Call Option Signals")
                if not calls_filtered.empty:
                    call_signals = []
                    for _, row in calls_filtered.iterrows():
                        signal_result = generate_signal(row, "call", df)
                        if signal_result['signal']:
                            row_dict = row.to_dict()
                            row_dict.update({
                                'signal_score': signal_result['score'],
                                'estimated_roi': signal_result['estimated_roi']
                            })
                            call_signals.append(row_dict)
                    
                    if call_signals:
                        signals_df = pd.DataFrame(call_signals)
                        signals_df = signals_df[(signals_df['signal_score'] > 0.8) & 
                                              (signals_df['volume'] > 5000) & 
                                              (signals_df['openInterest'] > 1000) & 
                                              (signals_df['estimated_roi'] > 20)]
                        
                        if not signals_df.empty:
                            st.markdown("### 🚀 Top Profit Opportunities")
                            st.dataframe(
                                signals_df[['contractSymbol', 'moneyness', 'strike', 'lastPrice', 'estimated_roi', 'signal_score']],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        signals_df = pd.DataFrame(call_signals).sort_values('signal_score', ascending=False)
                        display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'moneyness', 'estimated_roi', 'signal_score']
                        available_cols = [col for col in display_cols if col in signals_df.columns]
                        styled_df = signals_df[available_cols].round(4).style.applymap(color_roi, subset=['estimated_roi'])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        st.success(f"Found {len(call_signals)} call signals!")
                    else:
                        st.info("No call signals found matching criteria.")
                else:
                    st.info("No call options available for selected filters.")
            
            with col2:
                st.subheader("📉 Put Option Signals")
                if not puts_filtered.empty:
                    put_signals = []
                    for _, row in puts_filtered.iterrows():
                        signal_result = generate_signal(row, "put", df)
                        if signal_result['signal']:
                            row_dict = row.to_dict()
                            row_dict.update({
                                'signal_score': signal_result['score'],
                                'estimated_roi': signal_result['estimated_roi']
                            })
                            put_signals.append(row_dict)
                    
                    if put_signals:
                        signals_df = pd.DataFrame(put_signals)
                        signals_df = signals_df[(signals_df['signal_score'] > 0.8) & 
                                              (signals_df['volume'] > 5000) & 
                                              (signals_df['openInterest'] > 1000) & 
                                              (signals_df['estimated_roi'] > 20)]
                        
                        if not signals_df.empty:
                            st.markdown("### 🚀 Top Profit Opportunities")
                            st.dataframe(
                                signals_df[['contractSymbol', 'moneyness', 'strike', 'lastPrice', 'estimated_roi', 'signal_score']],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        signals_df = pd.DataFrame(put_signals).sort_values('signal_score', ascending=False)
                        display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'moneyness', 'estimated_roi', 'signal_score']
                        available_cols = [col for col in display_cols if col in signals_df.columns]
                        styled_df = signals_df[available_cols].round(4).style.applymap(color_roi, subset=['estimated_roi'])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        st.success(f"Found {len(put_signals)} put signals!")
                    else:
                        st.info("No put signals found matching criteria.")
                else:
                    st.info("No put options available for selected filters.")
            
            if st.button("📩 Report This Signal Was Profitable"):
                st.success("Thanks for your feedback!")
    
    with tab2:
        if 'df' in locals() and not df.empty:
            st.subheader("📊 Stock Data & Indicators")
            
            latest = df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${latest['Close']:.2f}")
            with col2:
                ema_9 = latest['EMA_9']
                st.metric("EMA 9", f"${ema_9:.2f}" if not pd.isna(ema_9) else "N/A")
            with col3:
                ema_20 = latest['EMA_20']
                st.metric("EMA 20", f"${ema_20:.2f}" if not pd.isna(ema_20) else "N/A")
            with col4:
                rsi = latest['RSI']
                st.metric("RSI", f"{rsi:.1f}" if not pd.isna(rsi) else "N/A")
            
            st.subheader("Recent Data")
            display_df = df.tail(10)[['Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'Volume']].round(2)
            st.dataframe(display_df, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_9'], name='EMA 9'))
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA_20'], name='EMA 20'))
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['VWAP'], name='VWAP'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Analysis Details")
        
        if enable_auto_refresh:
            st.info(f"🔄 Auto-refresh enabled: Every {refresh_interval} seconds")
        else:
            st.info("🔄 Auto-refresh disabled")
        
        if 'calls_filtered' in locals() and not calls_filtered.empty:
            st.write("**Sample Call Analysis:**")
            sample_call = calls_filtered.iloc[0]
            if 'df' in locals():
                result = generate_signal(sample_call, "call", df)
                st.json(result)
        
        st.write("**Current Signal Thresholds:**")
        st.json(SIGNAL_THRESHOLDS)
        
        st.write("**System Configuration:**")
        st.json(CONFIG)
        
        st.markdown("📢 Recent News")
        st.write(fetch_headlines(ticker))

    with st.expander("ℹ️ About Rate Limiting"):
        st.markdown("""
        Yahoo Finance may restrict how often data can be retrieved. If you see a "rate limited" warning, please:
        - Wait a few minutes before refreshing again
        - Avoid setting auto-refresh intervals lower than 1 minute
        - Use the app with one ticker at a time to reduce load
        """)

else:
    st.info("Please enter a stock ticker to begin analysis.")
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Steps to analyze options:**
        1. Enter a stock ticker (e.g., SPY, QQQ, AAPL)
        2. Configure auto-refresh settings in the sidebar (optional)
        3. Select expiration filter (0DTE for same-day, or near-term)
        4. Adjust strike range around current price
        5. Filter by moneyness (ITM, ATM, OTM)
        6. Review generated signals
        
        **Signal Criteria:**
        - **Calls:** High delta, sufficient gamma, low theta, bullish technicals
        - **Puts:** Low delta, sufficient gamma, low theta, bearish technicals
        
        **Technical Indicators:**
        - EMA crossovers for trend direction
        - RSI for momentum
        - VWAP for intraday sentiment
        - Volume analysis for confirmation
        
        **Refresh Features:**
        - **Auto-refresh:** Automatically updates data at set intervals
        - **Manual refresh:** Click "Refresh Now" to update immediately
        - **Clear cache:** Force fresh data retrieval
        """)
