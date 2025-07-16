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
    'PREMARKET_START': datetime.time(4, 0),  # 4:00 AM Eastern
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

def is_after_hours() -> bool:
    """Check if we're in after-hours trading"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    now_time = now.time()
    
    # Only consider weekdays
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    return CONFIG['MARKET_CLOSE'] < now_time <= datetime.time(20, 0)  # 8 PM Eastern

def is_early_market() -> bool:
    """Check if we're in the first 30 minutes of market open"""
    if not is_market_open():
        return False
    
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    market_open_today = datetime.datetime.combine(now.date(), CONFIG['MARKET_OPEN'])
    market_open_today = eastern.localize(market_open_today)
    
    return (now - market_open_today).total_seconds() < 1800  # First 30 minutes

# UPDATED REAL-TIME PRICE FUNCTION
def get_current_price(ticker: str) -> float:
    """Get the most current price with improved real-time accuracy"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        
        # Use appropriate price based on market session
        if is_market_open():
            price = info.get('last_price') or info.get('regular_market_price')
        elif is_premarket():
            price = info.get('pre_market_price') or info.get('regular_market_price')
        elif is_after_hours():
            price = info.get('post_market_price') or info.get('regular_market_price')
        else:
            price = info.get('regular_market_price') or info.get('previous_close')
        
        if price is None:
            # Fallback to minute data if fast_info fails
            data = stock.history(period='1d', interval='1m', prepost=True)
            if not data.empty:
                return data['Close'].iloc[-1]
        
        return price or 0.0
    except Exception as e:
        st.error(f"Error getting current price: {str(e)}")
        return 0.0

# REST OF THE CODE REMAINS THE SAME AS ORIGINAL...
# [Keep all other functions and logic unchanged]
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
if 'last_price_fetch' not in st.session_state:
    st.session_state.last_price_fetch = 0
if 'current_price' not in st.session_state:
    st.session_state.current_price = 0.0

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
        elif is_after_hours():
            st.warning("🌙 AFTER-HOURS Session")
        else:
            st.info("💤 Market is CLOSED")
    
    # REAL-TIME PRICE UPDATES
    with col2:
        # Only fetch price if market is open or we need to update
        current_time = time.time()
        price_refresh_rate = 30  # Update price every 30 seconds
        
        if (current_time - st.session_state.last_price_fetch > price_refresh_rate or
            st.session_state.current_price == 0.0):
            
            st.session_state.current_price = get_current_price(ticker)
            st.session_state.last_price_fetch = current_time
            
        st.metric("Current Price", f"${st.session_state.current_price:.2f}")
    
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
        st.session_state.last_price_fetch = 0  # Force price refresh
        st.rerun()
    
    # Add refresh counter display
    st.caption(f"🔄 Refresh count: {st.session_state.refresh_counter}")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Stock Data", "⚙️ Analysis Details"])

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
