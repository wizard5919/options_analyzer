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

# Suppress future warnings (e.g., from pandas or yfinance upcoming changes)
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
    'DATA_TIMEOUT': 30, # Timeout for individual API calls
    'MIN_DATA_POINTS': 50, # Minimum data points required for technical indicators
    'CACHE_TTL': 3600,  # Cache Time-To-Live in seconds (1 hour) to prevent frequent API calls
}

SIGNAL_THRESHOLDS = {
    'call': {
        'delta_min': 0.6,
        'gamma_min': 0.08,
        'theta_max': 0.05, # Theta is typically negative, but we consider its absolute value or a "less negative" value. Max positive value for theta
        'rsi_min': 50,
        'volume_multiplier': 1.5
    },
    'put': {
        'delta_max': -0.6, # Delta for puts is negative, so we check for a "more negative" value
        'gamma_min': 0.08,
        'theta_max': 0.05,
        'rsi_max': 50, # For puts, we look for RSI below this threshold
        'volume_multiplier': 1.5
    }
}

# =============================
# UTILITY FUNCTIONS
# =============================

def safe_api_call(func, *args, max_retries=CONFIG['MAX_RETRIES'], **kwargs):
    """Safely call API functions with retry logic and error reporting."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "Too Many Requests" in str(e):
                st.error("API rate limit hit. Please wait a few minutes before trying again or increase the refresh interval.")
                return None # Immediately stop if rate limited
            if attempt == max_retries - 1:
                st.error(f"API call failed after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(CONFIG['RETRY_DELAY'] * (attempt + 1))
    return None

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data(ticker: str, days: int = 10) -> pd.DataFrame:
    """Fetch stock data with caching and error handling."""
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)

        # Use auto_adjust=True to get adjusted close prices and suppress warnings
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="5m", # 5-minute intervals for intraday analysis
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.warning(f"No stock data found for ticker **{ticker}** in the last {days} days.")
            return pd.DataFrame()

        # Handle multi-level columns if present (common with older yfinance versions or certain queries)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required stock data columns for {ticker}: {missing_cols}. Data may be incomplete.")
            return pd.DataFrame()

        # Convert to numeric and handle any non-numeric data (e.g., from initial parsing)
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Remove rows with NaN in essential columns
        data = data.dropna(subset=required_cols)

        if len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient stock data points ({len(data)}) for **{ticker}**. Need at least {CONFIG['MIN_DATA_POINTS']} for reliable indicator calculation.")
            return pd.DataFrame()

        return data.reset_index(drop=True)

    except Exception as e:
        st.error(f"Error fetching stock data for **{ticker}**: {str(e)}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators (EMA, RSI, VWAP, Average Volume) with robust error handling."""
    if df.empty:
        return df

    try:
        # Make a copy to avoid modifying original DataFrame directly
        df = df.copy()

        # Validate required columns exist and are numeric
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column for indicator calculation: {col}. Cannot compute indicators.")
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any rows where essential columns became NaN after conversion
        df = df.dropna(subset=required_cols)

        if df.empty:
            st.warning("Stock data became empty after cleaning for indicator computation.")
            return df

        # Extract series for calculations
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)

        # Calculate indicators with checks for sufficient data
        # EMA indicators
        if len(close) >= 9:
            ema_9 = EMAIndicator(close=close, window=9)
            df['EMA_9'] = ema_9.ema_indicator()
        else:
            df['EMA_9'] = np.nan # Assign NaN if not enough data

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

        # VWAP (Volume Weighted Average Price)
        # Avoid division by zero if volume_cumsum is 0
        typical_price = (high + low + close) / 3
        vwap_cumsum = (volume * typical_price).cumsum()
        volume_cumsum = volume.cumsum()
        df['VWAP'] = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)

        # Average Volume (e.g., 20-period moving average of volume)
        window_size_vol = min(20, len(volume)) # Adjust window if data is sparse
        if window_size_vol > 0: # Ensure window_size is positive
            df['avg_vol'] = volume.rolling(window=window_size_vol, min_periods=1).mean()
        else:
            df['avg_vol'] = np.nan # Not enough data for average volume

        return df

    except Exception as e:
        st.error(f"Error computing technical indicators: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_expiries(ticker: str) -> List[str]:
    """Get options expiries for a given ticker with error handling."""
    try:
        stock = yf.Ticker(ticker)
        # safe_api_call is used to handle retries for initial fetch
        expiries = safe_api_call(lambda: stock.options)
        return list(expiries) if expiries else []
    except Exception as e:
        st.error(f"Error fetching options expiries for **{ticker}**: {str(e)}")
        return []

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def fetch_options_data(ticker: str, expiries: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch options data (calls and puts) for multiple expiries.
    Includes comprehensive error handling and rate limit consideration.
    """
    all_calls = pd.DataFrame()
    all_puts = pd.DataFrame()
    failed_expiries = []

    stock = yf.Ticker(ticker)

    for expiry in expiries:
        try:
            # Use safe_api_call for fetching option_chain
            chain = safe_api_call(stock.option_chain, expiry, max_retries=1) # Limit retries for chains to avoid compounding rate limits
            if chain is None:
                failed_expiries.append(expiry)
                continue # Skip this expiry if fetching failed

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # Add expiry information to each option row
            calls['expiry'] = expiry
            puts['expiry'] = expiry

            # Define essential columns for options data
            # 'impliedVolatility' is crucial for Greeks calculation, 'lastPrice' for signal, 'volume', 'openInterest' for liquidity
            required_cols_options = ['strike', 'lastPrice', 'volume', 'openInterest', 'delta', 'gamma', 'theta', 'impliedVolatility']

            # Validate and clean columns for calls
            for col in required_cols_options:
                if col not in calls.columns:
                    calls[col] = np.nan # Add as NaN if missing
                calls[col] = pd.to_numeric(calls[col], errors='coerce')
            calls = calls.dropna(subset=['strike', 'lastPrice', 'delta', 'gamma', 'theta']) # Drop rows if key Greeks or price are missing

            # Validate and clean columns for puts
            for col in required_cols_options:
                if col not in puts.columns:
                    puts[col] = np.nan # Add as NaN if missing
                puts[col] = pd.to_numeric(puts[col], errors='coerce')
            puts = puts.dropna(subset=['strike', 'lastPrice', 'delta', 'gamma', 'theta']) # Drop rows if key Greeks or price are missing


            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)

        except Exception as e:
            if "Too Many Requests" in str(e):
                st.error("API rate limit hit while fetching options chain. Please wait a few minutes before refreshing.")
                # If rate-limited, it's better to stop further option chain fetches for this refresh cycle
                return pd.DataFrame(), pd.DataFrame()
            st.warning(f"Failed to fetch options for expiry {expiry} due to: {str(e)}. Skipping this expiry.")
            failed_expiries.append(expiry)
            continue

    if failed_expiries:
        st.info(f"Could not retrieve complete options data for the following expiries: {', '.join(failed_expiries)}. This might be due to temporary network issues or missing data for certain expiries.")

    return all_calls, all_puts

def classify_moneyness(strike: float, spot: float, tolerance: float = 0.005) -> str:
    """
    Classify option moneyness (ITM, ATM, OTM) with a small tolerance for ATM.
    Tolerance is a percentage of the spot price.
    """
    if spot == 0: # Avoid division by zero
        return 'N/A'

    # Calculate difference as a percentage of spot price
    diff_percent = abs(strike - spot) / spot

    if diff_percent <= tolerance:
        return 'ATM'
    elif strike < spot: # For calls ITM, for puts OTM
        return 'ITM' if strike < spot else 'OTM' # For calls: strike < spot means ITM. For puts: strike > spot means ITM. The function needs to be aware of option type if used generically.
    else: # strike > spot
        return 'OTM' if strike > spot else 'ITM' # For calls: strike > spot means OTM. For puts: strike < spot means OTM.

# Corrected classify_moneyness for specific call/put use
def classify_option_moneyness(strike: float, spot: float, option_type: str, tolerance: float = 0.005) -> str:
    """Classify option moneyness based on option type."""
    if spot == 0:
        return 'N/A'

    if option_type.lower() == 'call':
        if strike < spot * (1 - tolerance):
            return 'ITM'
        elif strike > spot * (1 + tolerance):
            return 'OTM'
        else:
            return 'ATM'
    elif option_type.lower() == 'put':
        if strike > spot * (1 + tolerance):
            return 'ITM'
        elif strike < spot * (1 - tolerance):
            return 'OTM'
        else:
            return 'ATM'
    else:
        return 'Invalid Type'


def validate_option_data(option: pd.Series) -> bool:
    """Validate that option has required data for analysis and reasonable values."""
    required_fields = ['delta', 'gamma', 'theta', 'strike', 'lastPrice', 'volume', 'openInterest']

    for field in required_fields:
        if field not in option or pd.isna(option[field]):
            return False

    # Check for reasonable numerical values
    if not (option['lastPrice'] > 0 and option['strike'] > 0 and option['volume'] >= 0 and option['openInterest'] >= 0):
        return False
    
    # Greeks should generally be within plausible ranges
    if not (-1.0 <= option['delta'] <= 1.0 and option['gamma'] >= 0 and option['theta'] <= 0): # Theta is usually negative
        return False

    return True

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame) -> Dict:
    """Generate trading signal with detailed analysis based on Greeks and technical indicators."""
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available for technical analysis'}

    if not validate_option_data(option):
        return {'signal': False, 'reason': 'Insufficient or invalid option data'}

    latest_stock_data = stock_df.iloc[-1]

    try:
        # Extract option Greeks (ensure they are float and not None)
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta']) # Theta is typically negative, representing time decay

        # Extract latest stock technical data
        close = float(latest_stock_data['Close'])
        ema_9 = float(latest_stock_data['EMA_9']) if not pd.isna(latest_stock_data['EMA_9']) else None
        ema_20 = float(latest_stock_data['EMA_20']) if not pd.isna(latest_stock_data['EMA_20']) else None
        rsi = float(latest_stock_data['RSI']) if not pd.isna(latest_stock_data['RSI']) else None
        vwap = float(latest_stock_data['VWAP']) if not pd.isna(latest_stock_data['VWAP']) else None
        volume = float(latest_stock_data['Volume'])
        avg_vol = float(latest_stock_data['avg_vol']) if not pd.isna(latest_stock_data['avg_vol']) else volume # Fallback to current volume

        # Get dynamic thresholds for the specific option side (call/put)
        thresholds = SIGNAL_THRESHOLDS[side]

        # Define conditions for signal generation
        # Each condition is (boolean_result, description_string, actual_value)
        conditions = []

        if side == "call":
            conditions.extend([
                (delta >= thresholds['delta_min'], f"Delta >= {thresholds['delta_min']}", f"{delta:.2f}"),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']}", f"{gamma:.2f}"),
                (theta >= -thresholds['theta_max'], f"Theta >= -{thresholds['theta_max']}", f"{theta:.2f}"), # Theta should be less negative (closer to zero or positive)
                (ema_9 is not None and ema_20 is not None and close > ema_9 and ema_9 > ema_20, "Price > EMA9 > EMA20", f"P:{close:.2f} > E9:{ema_9:.2f} > E20:{ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi >= thresholds['rsi_min'], f"RSI >= {thresholds['rsi_min']}", f"{rsi:.1f}"),
                (vwap is not None and close > vwap, "Price > VWAP", f"P:{close:.2f} > VWAP:{vwap:.2f}" if vwap else "N/A"),
                (volume > thresholds['volume_multiplier'] * avg_vol, f"Volume > {thresholds['volume_multiplier']}x Avg Vol", f"Vol:{volume:.0f} > Avg:{avg_vol:.0f}")
            ])
        else:  # put side
            conditions.extend([
                (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']}", f"{delta:.2f}"),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']}", f"{gamma:.2f}"),
                (theta >= -thresholds['theta_max'], f"Theta >= -{thresholds['theta_max']}", f"{theta:.2f}"), # Theta should be less negative
                (ema_9 is not None and ema_20 is not None and close < ema_9 and ema_9 < ema_20, "Price < EMA9 < EMA20", f"P:{close:.2f} < E9:{ema_9:.2f} < E20:{ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi <= thresholds['rsi_max'], f"RSI <= {thresholds['rsi_max']}", f"{rsi:.1f}"),
                (vwap is not None and close < vwap, "Price < VWAP", f"P:{close:.2f} < VWAP:{vwap:.2f}" if vwap else "N/A"),
                (volume > thresholds['volume_multiplier'] * avg_vol, f"Volume > {thresholds['volume_multiplier']}x Avg Vol", f"Vol:{volume:.0f} > Avg:{avg_vol:.0f}")
            ])

        # Evaluate all conditions
        passed_conditions = [desc for passed, desc, val in conditions if passed]
        failed_conditions = [f"{desc} (Current: {val})" for passed, desc, val in conditions if not passed]

        signal = all(passed for passed, desc, val in conditions)
        score = len(passed_conditions) / len(conditions) if conditions else 0 # Avoid division by zero

        return {
            'signal': signal,
            'passed_conditions': passed_conditions,
            'failed_conditions': failed_conditions,
            'score': score
        }

    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation for {side} option: {str(e)}'}

# =============================
# STREAMLIT INTERFACE
# =============================

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Enhanced robust version** with comprehensive error handling, detailed analysis, and real-time refresh capabilities.")

# Initialize session state for refresh functionality
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = time.time()

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Auto-refresh settings
    st.subheader("🔄 Auto-Refresh Settings")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh", value=False, help="Automatically refresh data at set intervals. Be mindful of API rate limits.")

    if enable_auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[300, 600, 900, 1800], # Options in seconds (5 min, 10 min, 15 min, 30 min)
            index=0, # Default to 5 minutes
            format_func=lambda x: f"{x // 60} minutes" if x >= 60 else f"{x} seconds",
            help="Select how often the data should refresh. Higher intervals reduce the chance of hitting API rate limits."
        )
        st.info(f"Data will refresh every {refresh_interval // 60} minutes.")
    else:
        refresh_interval = 0 # No auto-refresh

    # Signal thresholds
    st.subheader("Signal Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Calls (Bullish Signals)")
        SIGNAL_THRESHOLDS['call']['delta_min'] = st.slider("Min Delta (Call)", 0.1, 1.0, SIGNAL_THRESHOLDS['call']['delta_min'], 0.05)
        SIGNAL_THRESHOLDS['call']['gamma_min'] = st.slider("Min Gamma (Call)", 0.01, 0.2, SIGNAL_THRESHOLDS['call']['gamma_min'], 0.01)
        SIGNAL_THRESHOLDS['call']['rsi_min'] = st.slider("Min RSI (Call)", 30, 70, SIGNAL_THRESHOLDS['call']['rsi_min'], 5)

    with col2:
        st.write("#### Puts (Bearish Signals)")
        SIGNAL_THRESHOLDS['put']['delta_max'] = st.slider("Max Delta (Put)", -1.0, -0.1, SIGNAL_THRESHOLDS['put']['delta_max'], 0.05)
        SIGNAL_THRESHOLDS['put']['gamma_min'] = st.slider("Min Gamma (Put)", 0.01, 0.2, SIGNAL_THRESHOLDS['put']['gamma_min'], 0.01)
        SIGNAL_THRESHOLDS['put']['rsi_max'] = st.slider("Max RSI (Put)", 30, 70, SIGNAL_THRESHOLDS['put']['rsi_max'], 5)

    # Common thresholds
    st.write("#### Common Thresholds")
    SIGNAL_THRESHOLDS['call']['theta_max'] = SIGNAL_THRESHOLDS['put']['theta_max'] = st.slider("Max Theta (Time Decay, abs value)", 0.01, 0.1, SIGNAL_THRESHOLDS['call']['theta_max'], 0.01, help="Maximum acceptable absolute value of Theta. Lower is better as it implies less time decay.")
    SIGNAL_THRESHOLDS['call']['volume_multiplier'] = SIGNAL_THRESHOLDS['put']['volume_multiplier'] = st.slider("Volume Multiplier (vs. Avg)", 1.0, 3.0, SIGNAL_THRESHOLDS['call']['volume_multiplier'], 0.1, help="Current volume must be X times the average volume to signal high interest.")

# Main interface
ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper().strip()

if ticker:
    # Real-time data refresh controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.subheader(f"📊 {ticker} Options Analysis")

    with col2:
        manual_refresh = st.button("🔄 Refresh Now", help="Click to immediately refresh all data (stock, options).")

    with col3:
        if st.button("🗑️ Clear Cache", help="Clears all cached data, forcing a fresh download. Use if data seems stale."):
            st.cache_data.clear()
            st.session_state.refresh_counter = 0 # Reset refresh counter
            st.success("Cache cleared! Data will be re-fetched.")
            st.session_state.last_auto_refresh = time.time() # Reset last refresh time
            st.rerun() # Rerun to reflect cache clear and start fresh

    with col4:
        # Show refresh status / countdown
        if enable_auto_refresh and refresh_interval > 0:
            current_time = time.time()
            time_elapsed = current_time - st.session_state.last_auto_refresh
            remaining = max(0, refresh_interval - int(time_elapsed))
            if remaining > 0:
                st.info(f"⏱️ Next refresh in {remaining}s")
            else:
                st.success("🔄 Refreshing now...")
        else:
            st.info("Auto-refresh disabled.")


    # Auto-refresh logic (triggered only if enabled and time elapsed)
    if enable_auto_refresh and refresh_interval > 0:
        current_time = time.time()
        time_elapsed = current_time - st.session_state.last_auto_refresh

        if time_elapsed >= refresh_interval:
            st.session_state.last_auto_refresh = current_time
            st.session_state.refresh_counter += 1
            st.cache_data.clear() # Clear cache to force fresh data
            st.rerun() # Rerun the app

    # Manual refresh logic
    if manual_refresh:
        st.cache_data.clear() # Clear cache to force fresh data
        st.session_state.last_auto_refresh = time.time() # Reset auto-refresh timer too
        st.session_state.refresh_counter += 1 # Increment manual refresh counter
        st.rerun() # Rerun the app to fetch new data

    # Show last update timestamp and refresh count
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"📅 Last data update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.caption(f"🔄 Total refreshes this session: {st.session_state.refresh_counter}")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📈 Stock Data", "⚙️ Analysis Details"])

    with tab1:
        try:
            with st.spinner("Fetching and analyzing data... This may take a moment, especially on first load or refresh."):
                # Get stock data
                df = get_stock_data(ticker)

                if df.empty:
                    st.warning("Please ensure the ticker symbol is correct and valid data is available.")
                    st.stop()

                # Compute indicators
                df = compute_indicators(df)

                if df.empty:
                    st.error("Unable to compute technical indicators from stock data. Please check data availability.")
                    st.stop()

                # Display current stock info
                current_price = df.iloc[-1]['Close']
                st.success(f"✅ **{ticker}** - Current Price: **${current_price:.2f}**")

                # Get options expiries
                expiries = get_options_expiries(ticker)

                if not expiries:
                    st.error("No options expiries available for this ticker. This might be due to incorrect ticker or API limitations.")
                    st.stop()

                # Expiry selection
                expiry_mode = st.radio(
                    "Select Expiration Filter:",
                    ["0DTE Only", "Near-Term Expiries (Next 5)"],
                    help="0DTE: Only options expiring today. Near-Term: Options for the next 5 available expiration dates."
                )

                today = datetime.date.today()
                expiries_to_use = []
                if expiry_mode == "0DTE Only":
                    # Filter for expiries that match today's date
                    expiries_to_use = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]
                else: # "Near-Term Expiries"
                    # Take the first 5 available expiries, sorted chronologically
                    expiries_to_use = sorted([e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() >= today])[:5]

                if not expiries_to_use:
                    st.warning("No options expiries found for the selected filter criteria. Try changing the 'Expiration Filter'.")
                    st.stop()

                st.info(f"Analyzing {len(expiries_to_use)} expiries: {', '.join(expiries_to_use)}")

                # Fetch options data for selected expiries
                calls, puts = fetch_options_data(ticker, expiries_to_use)

                if calls.empty and puts.empty:
                    st.error("No options data available for the selected expiries. This could be due to API rate limits or data issues.")
                    st.stop()

                # Strike range filter
                min_strike_default = current_price - 10 if current_price > 10 else 0
                max_strike_default = current_price + 10
                strike_range = st.slider(
                    "Strike Range Around Current Price ($):",
                    float(current_price - 50), # Lower bound of slider
                    float(current_price + 50), # Upper bound of slider
                    (min_strike_default, max_strike_default), # Default range
                    step=0.5, # Small step for precision
                    format="$%.2f",
                    help="Filter options by strike price within this range relative to the current stock price."
                )
                min_strike_filter = strike_range[0]
                max_strike_filter = strike_range[1]

                # Filter options by strike
                calls_filtered = calls[(calls['strike'] >= min_strike_filter) & (calls['strike'] <= max_strike_filter)].copy()
                puts_filtered = puts[(puts['strike'] >= min_strike_filter) & (puts['strike'] <= max_strike_filter)].copy()

                # Add moneyness classification
                if not calls_filtered.empty:
                    calls_filtered['moneyness'] = calls_filtered['strike'].apply(lambda x: classify_option_moneyness(x, current_price, 'call'))
                if not puts_filtered.empty:
                    puts_filtered['moneyness'] = puts_filtered['strike'].apply(lambda x: classify_option_moneyness(x, current_price, 'put'))

                # Moneyness filter
                m_filter = st.multiselect(
                    "Filter by Moneyness (ITM/ATM/OTM):",
                    options=["ITM", "ATM", "OTM"],
                    default=["ITM", "ATM", "OTM"],
                    help="Select which moneyness categories to include in the signal analysis."
                )

                if not calls_filtered.empty:
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
                if not puts_filtered.empty:
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]


                # Generate and display signals
                col_calls_signals, col_puts_signals = st.columns(2)

                with col_calls_signals:
                    st.subheader("📈 Call Option Signals")
                    if not calls_filtered.empty:
                        call_signals_list = []
                        # Apply signal generation only to valid options
                        valid_calls = calls_filtered[calls_filtered.apply(validate_option_data, axis=1)]
                        for _, row in valid_calls.iterrows():
                            signal_result = generate_signal(row, "call", df)
                            if signal_result['signal']:
                                # Create a dictionary for the signal, adding signal_score and filtering relevant columns
                                signal_data = row[['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'impliedVolatility', 'volume', 'openInterest', 'expiry', 'moneyness']].to_dict()
                                signal_data['signal_score'] = signal_result['score']
                                call_signals_list.append(signal_data)

                        if call_signals_list:
                            signals_df = pd.DataFrame(call_signals_list)
                            signals_df = signals_df.sort_values(['signal_score', 'gamma'], ascending=[False, False]) # Sort by score, then gamma
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'moneyness', 'signal_score', 'expiry']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            st.dataframe(signals_df[available_cols].round(4), use_container_width=True, hide_index=True)
                            st.success(f"Found **{len(call_signals_list)}** Call signals matching criteria!")
                        else:
                            st.info("No Call options found matching all signal criteria for the selected filters.")
                    else:
                        st.info("No Call options available for the selected strike and moneyness filters.")

                with col_puts_signals:
                    st.subheader("📉 Put Option Signals")
                    if not puts_filtered.empty:
                        put_signals_list = []
                        valid_puts = puts_filtered[puts_filtered.apply(validate_option_data, axis=1)]
                        for _, row in valid_puts.iterrows():
                            signal_result = generate_signal(row, "put", df)
                            if signal_result['signal']:
                                signal_data = row[['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'impliedVolatility', 'volume', 'openInterest', 'expiry', 'moneyness']].to_dict()
                                signal_data['signal_score'] = signal_result['score']
                                put_signals_list.append(signal_data)

                        if put_signals_list:
                            signals_df = pd.DataFrame(put_signals_list)
                            signals_df = signals_df.sort_values(['signal_score', 'gamma'], ascending=[False, False])
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'moneyness', 'signal_score', 'expiry']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            st.dataframe(signals_df[available_cols].round(4), use_container_width=True, hide_index=True)
                            st.success(f"Found **{len(put_signals_list)}** Put signals matching criteria!")
                        else:
                            st.info("No Put options found matching all signal criteria for the selected filters.")
                    else:
                        st.info("No Put options available for the selected strike and moneyness filters.")

        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {str(e)}")
            st.error("Please try refreshing the page or checking the ticker symbol.")

    ---

    with tab2:
        # Stock data visualization
        if 'df' in locals() and not df.empty:
            st.subheader("📊 Stock Data & Indicators (Latest)")

            # Display latest values using metrics
            latest_stock_data = df.iloc[-1]

            col_p, col_ema9, col_ema20, col_rsi, col_vwap, col_vol = st.columns(6)

            with col_p:
                st.metric("Current Price", f"${latest_stock_data['Close']:.2f}")

            with col_ema9:
                ema_9 = latest_stock_data.get('EMA_9')
                st.metric("EMA 9", f"${ema_9:.2f}" if pd.notna(ema_9) else "N/A")

            with col_ema20:
                ema_20 = latest_stock_data.get('EMA_20')
                st.metric("EMA 20", f"${ema_20:.2f}" if pd.notna(ema_20) else "N/A")

            with col_rsi:
                rsi = latest_stock_data.get('RSI')
                st.metric("RSI", f"{rsi:.1f}" if pd.notna(rsi) else "N/A")

            with col_vwap:
                vwap = latest_stock_data.get('VWAP')
                st.metric("VWAP", f"${vwap:.2f}" if pd.notna(vwap) else "N/A")
            
            with col_vol:
                volume_latest = latest_stock_data.get('Volume')
                avg_vol_latest = latest_stock_data.get('avg_vol')
                if pd.notna(volume_latest) and pd.notna(avg_vol_latest):
                    st.metric("Volume / Avg Vol", f"{volume_latest:.0f} / {avg_vol_latest:.0f}")
                elif pd.notna(volume_latest):
                    st.metric("Volume", f"{volume_latest:.0f}")
                else:
                    st.metric("Volume", "N/A")

            st.subheader("Recent Price and Indicator History (Last 20 bars)")
            display_df = df.tail(20)[['Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'Volume', 'avg_vol']]
            st.dataframe(display_df.round(2), use_container_width=True)
        else:
            st.info("No stock data available to display in this tab.")

    ---

    with tab3:
        st.subheader("🔍 Analysis Details & Debugging Information")

        st.markdown("This tab provides insights into the internal state and the specific criteria used for signal generation.")

        if enable_auto_refresh:
            st.info(f"🔄 Auto-refresh is **enabled**: Data refreshes every {refresh_interval // 60} minutes.")
        else:
            st.info("🔄 Auto-refresh is **disabled**.")

        st.write("#### Current Signal Thresholds:")
        st.json(SIGNAL_THRESHOLDS)

        st.write("#### System Configuration:")
        st.json(CONFIG)

        if 'df' in locals() and not df.empty:
            st.write("#### Latest Stock Data Point (for signal evaluation):")
            st.json(df.iloc[-1].to_dict())

            if 'calls_filtered' in locals() and not calls_filtered.empty:
                st.write("#### Sample Call Option Signal Evaluation (First found valid call):")
                sample_call = calls_filtered[calls_filtered.apply(validate_option_data, axis=1)].iloc[0] if not calls_filtered[calls_filtered.apply(validate_option_data, axis=1)].empty else None
                if sample_call is not None:
                    result = generate_signal(sample_call, "call", df)
                    st.json(result)
                else:
                    st.info("No valid call option available for sample analysis.")

            if 'puts_filtered' in locals() and not puts_filtered.empty:
                st.write("#### Sample Put Option Signal Evaluation (First found valid put):")
                sample_put = puts_filtered[puts_filtered.apply(validate_option_data, axis=1)].iloc[0] if not puts_filtered[puts_filtered.apply(validate_option_data, axis=1)].empty else None
                if sample_put is not None:
                    result = generate_signal(sample_put, "put", df)
                    st.json(result)
                else:
                    st.info("No valid put option available for sample analysis.")

else:
    st.info("Please enter a stock ticker to begin analysis.")

    # Display help information
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Steps to analyze options:**
        1.  **Enter a stock ticker** (e.g., `SPY`, `QQQ`, `AAPL`) in the input box above.
        2.  **Configure auto-refresh** settings in the sidebar (optional). This automatically updates data at your chosen interval.
        3.  **Select expiration filter** to focus on 0DTE (Zero Days To Expiration) options or a range of near-term expiries.
        4.  **Adjust the strike range** around the current stock price to narrow down the options displayed.
        5.  **Filter by moneyness** (In-The-Money, At-The-Money, Out-of-The-Money) to include or exclude certain options.
        6.  **Review generated signals** in the "Signals" tab. The app highlights options that meet the configured buy criteria.

        **Signal Criteria (Customizable in Sidebar):**
        * **Calls (Bullish):** High Delta (strong price sensitivity), sufficient Gamma (high convexity), low (less negative) Theta (less time decay), bullish technical indicators (Price > EMA9 > EMA20, RSI > threshold, Price > VWAP), and higher than average volume.
        * **Puts (Bearish):** Low (more negative) Delta, sufficient Gamma, low (less negative) Theta, bearish technical indicators (Price < EMA9 < EMA20, RSI < threshold, Price < VWAP), and higher than average volume.

        **Technical Indicators Used:**
        * **EMA (Exponential Moving Average):** 9-period and 20-period EMAs are used to confirm short-term trend direction and crossovers.
        * **RSI (Relative Strength Index):** 14-period RSI measures momentum, indicating overbought/oversold conditions.
        * **VWAP (Volume Weighted Average Price):** Shows the average price of a security adjusted for its volume, useful for intraday sentiment.
        * **Volume Analysis:** Checks if current trading volume is significantly higher than average, indicating strong interest.

        **Data Refresh & Cache Management:**
        * **Auto-refresh:** Automatically updates data at set intervals (configurable in sidebar).
        * **Manual refresh:** Click "Refresh Now" to update immediately.
        * **Clear cache:** Forces the app to download all data fresh, useful for troubleshooting stale data.

        ---

        **Important Note on Data Limits:**
        This application fetches real-time financial data from public sources (like Yahoo Finance). These sources often have **rate limits**, meaning you can only make a certain number of requests in a given time. If you encounter **"Too Many Requests"** errors, please:
        * **Wait a few minutes** before trying again. The service might temporarily block your requests.
        * **Increase the "Refresh Interval"** in the sidebar to reduce the frequency of data requests.
        * Use the **"Clear Cache"** button only when absolutely necessary to force a fresh data retrieval, as this counts as new requests.
        """)
