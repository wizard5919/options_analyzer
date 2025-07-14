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
            if attempt == max_retries - 1:
                st.error(f"API call failed after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(CONFIG['RETRY_DELAY'] * (attempt + 1))
    return None

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data(ticker: str, days: int = 10) -> pd.DataFrame:
    """Fetch stock data with caching and error handling"""
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)
        
        # Use auto_adjust=True to suppress warnings
        data = yf.download(
            ticker, 
            start=start, 
            end=end, 
            interval="5m",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.warning(f"No data found for ticker {ticker}")
            return pd.DataFrame()

        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure we have required columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Clean and validate data
        data = data.dropna(how='all')
        
        # Convert to numeric and handle any nested structures
        for col in required_cols:
            if col in data.columns:
                # Handle nested data structures
                if hasattr(data[col].iloc[0], '__len__') and not isinstance(data[col].iloc[0], str):
                    data[col] = data[col].apply(lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x)
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Remove rows with NaN in essential columns
        data = data.dropna(subset=required_cols)
        
        if len(data) < CONFIG['MIN_DATA_POINTS']:
            st.warning(f"Insufficient data points ({len(data)}). Need at least {CONFIG['MIN_DATA_POINTS']}.")
            return pd.DataFrame()
        
        return data.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators with comprehensive error handling"""
    if df.empty:
        return df
    
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Validate required columns exist
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Ensure data types are correct
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any remaining NaN values
        df = df.dropna(subset=required_cols)
        
        if df.empty:
            return df
        
        # Extract series for calculations
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)

        # Calculate indicators with minimum data requirements
        try:
            # EMA indicators
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
                
            # RSI
            if len(close) >= 14:
                rsi = RSIIndicator(close=close, window=14)
                df['RSI'] = rsi.rsi()
            else:
                df['RSI'] = np.nan

            # VWAP
            typical_price = (high + low + close) / 3
            vwap_cumsum = (volume * typical_price).cumsum()
            volume_cumsum = volume.cumsum()
            
            # Avoid division by zero
            df['VWAP'] = np.where(volume_cumsum != 0, vwap_cumsum / volume_cumsum, np.nan)
            
            # Average Volume
            window_size = min(20, len(volume))
            if window_size > 1:
                df['avg_vol'] = volume.rolling(window=window_size, min_periods=1).mean()
            else:
                df['avg_vol'] = volume.mean()
                
        except Exception as e:
            st.error(f"Error computing indicators: {str(e)}")
            return pd.DataFrame()
        
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
        st.error(f"Error fetching expiries: {str(e)}")
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
            
            # Add expiry information
            calls['expiry'] = expiry
            puts['expiry'] = expiry
            
            # Validate required columns exist
            required_cols = ['strike', 'lastPrice', 'volume', 'openInterest']
            
            for df_name, df in [('calls', calls), ('puts', puts)]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing columns in {df_name} for {expiry}: {missing_cols}")
                    continue
            
            all_calls = pd.concat([all_calls, calls], ignore_index=True)
            all_puts = pd.concat([all_puts, puts], ignore_index=True)
            
        except Exception as e:
            st.warning(f"Failed to fetch options for {expiry}: {str(e)}")
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
    required_fields = ['delta', 'gamma', 'theta', 'strike', 'lastPrice']
    
    for field in required_fields:
        if field not in option or pd.isna(option[field]):
            return False
    
    # Check for reasonable values
    if option['lastPrice'] <= 0:
        return False
    
    return True

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame) -> Dict:
    """Generate trading signal with detailed analysis"""
    if stock_df.empty:
        return {'signal': False, 'reason': 'No stock data available'}
    
    if not validate_option_data(option):
        return {'signal': False, 'reason': 'Insufficient option data'}
    
    latest = stock_df.iloc[-1]
    
    try:
        # Extract option Greeks
        delta = float(option['delta'])
        gamma = float(option['gamma'])
        theta = float(option['theta'])
        
        # Extract stock data
        close = float(latest['Close'])
        ema_9 = float(latest['EMA_9']) if not pd.isna(latest['EMA_9']) else None
        ema_20 = float(latest['EMA_20']) if not pd.isna(latest['EMA_20']) else None
        rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
        vwap = float(latest['VWAP']) if not pd.isna(latest['VWAP']) else None
        volume = float(latest['Volume'])
        avg_vol = float(latest['avg_vol']) if not pd.isna(latest['avg_vol']) else volume
        
        # Get thresholds for the side
        thresholds = SIGNAL_THRESHOLDS[side]
        
        # Check conditions based on side
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
        else:  # put
            conditions = [
                (delta <= thresholds['delta_max'], f"Delta <= {thresholds['delta_max']}", delta),
                (gamma >= thresholds['gamma_min'], f"Gamma >= {thresholds['gamma_min']}", gamma),
                (theta <= thresholds['theta_max'], f"Theta <= {thresholds['theta_max']}", theta),
                (ema_9 is not None and ema_20 is not None and close < ema_9 < ema_20, "Price < EMA9 < EMA20", f"{close:.2f} < {ema_9:.2f} < {ema_20:.2f}" if ema_9 and ema_20 else "N/A"),
                (rsi is not None and rsi < thresholds['rsi_max'], f"RSI < {thresholds['rsi_max']}", rsi),
                (vwap is not None and close < vwap, "Price < VWAP", f"{close:.2f} < {vwap:.2f}" if vwap else "N/A"),
                (volume > thresholds['volume_multiplier'] * avg_vol, f"Volume > {thresholds['volume_multiplier']}x avg", f"{volume:.0f} > {avg_vol:.0f}")
            ]
        
        # Check all conditions
        passed_conditions = [desc for passed, desc, val in conditions if passed]
        failed_conditions = [f"{desc} (got {val})" for passed, desc, val in conditions if not passed]
        
        signal = all(passed for passed, desc, val in conditions)
        
        return {
            'signal': signal,
            'passed_conditions': passed_conditions,
            'failed_conditions': failed_conditions,
            'score': len(passed_conditions) / len(conditions)
        }
        
    except Exception as e:
        return {'signal': False, 'reason': f'Error in signal generation: {str(e)}'}

# =============================
# STREAMLIT INTERFACE
# =============================

st.title("📈 Options Greeks Buy Signal Analyzer")
st.markdown("**Enhanced robust version** with comprehensive error handling and detailed analysis.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Signal thresholds
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
    
    # Common thresholds
    st.write("**Common**")
    SIGNAL_THRESHOLDS['call']['theta_max'] = SIGNAL_THRESHOLDS['put']['theta_max'] = st.slider("Max Theta", 0.01, 0.1, 0.05, 0.01)
    SIGNAL_THRESHOLDS['call']['volume_multiplier'] = SIGNAL_THRESHOLDS['put']['volume_multiplier'] = st.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)

# Main interface
ticker = st.text_input("Enter Stock Ticker (e.g., IWM, SPY, AAPL):", value="IWM").upper()

if ticker:
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
                
                # Get options expiries
                expiries = get_options_expiries(ticker)
                
                if not expiries:
                    st.error("No options expiries available for this ticker.")
                    st.stop()
                
                # Expiry selection
                expiry_mode = st.radio("Select Expiration Filter:", ["0DTE Only", "All Near-Term Expiries"])
                
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
                
                # Strike range filter
                strike_range = st.slider("Strike Range Around Current Price ($):", -50, 50, (-10, 10), 1)
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
                m_filter = st.multiselect("Filter by Moneyness:", options=["ITM", "ATM", "OTM"], default=["ITM", "ATM", "OTM"])
                
                if not calls_filtered.empty:
                    calls_filtered = calls_filtered[calls_filtered['moneyness'].isin(m_filter)]
                if not puts_filtered.empty:
                    puts_filtered = puts_filtered[puts_filtered['moneyness'].isin(m_filter)]
                
                # Generate signals
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Call Option Signals")
                    if not calls_filtered.empty:
                        call_signals = []
                        for _, row in calls_filtered.iterrows():
                            signal_result = generate_signal(row, "call", df)
                            if signal_result['signal']:
                                row_dict = row.to_dict()
                                row_dict['signal_score'] = signal_result['score']
                                call_signals.append(row_dict)
                        
                        if call_signals:
                            signals_df = pd.DataFrame(call_signals)
                            # Sort by signal score
                            signals_df = signals_df.sort_values('signal_score', ascending=False)
                            
                            # Display key columns
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'moneyness', 'signal_score']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            
                            st.dataframe(
                                signals_df[available_cols].round(4),
                                use_container_width=True,
                                hide_index=True
                            )
                            
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
                                row_dict['signal_score'] = signal_result['score']
                                put_signals.append(row_dict)
                        
                        if put_signals:
                            signals_df = pd.DataFrame(put_signals)
                            # Sort by signal score
                            signals_df = signals_df.sort_values('signal_score', ascending=False)
                            
                            # Display key columns
                            display_cols = ['contractSymbol', 'strike', 'lastPrice', 'delta', 'gamma', 'theta', 'moneyness', 'signal_score']
                            available_cols = [col for col in display_cols if col in signals_df.columns]
                            
                            st.dataframe(
                                signals_df[available_cols].round(4),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            st.success(f"Found {len(put_signals)} put signals!")
                        else:
                            st.info("No put signals found matching criteria.")
                    else:
                        st.info("No put options available for selected filters.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please refresh the page and try again.")
    
    with tab2:
        # Stock data visualization
        if 'df' in locals() and not df.empty:
            st.subheader("📊 Stock Data & Indicators")
            
            # Display latest values
            latest = df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            # Display recent data
            st.subheader("Recent Data")
            display_df = df.tail(10)[['Close', 'EMA_9', 'EMA_20', 'RSI', 'VWAP', 'Volume']].round(2)
            st.dataframe(display_df, use_container_width=True)
        
    with tab3:
        st.subheader("🔍 Analysis Details")
        
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

else:
    st.info("Please enter a stock ticker to begin analysis.")
    
    # Display help information
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Steps to analyze options:**
        1. Enter a stock ticker (e.g., SPY, QQQ, AAPL)
        2. Select expiration filter (0DTE for same-day, or near-term)
        3. Adjust strike range around current price
        4. Filter by moneyness (ITM, ATM, OTM)
        5. Review generated signals
        
        **Signal Criteria:**
        - **Calls:** High delta, sufficient gamma, low theta, bullish technicals
        - **Puts:** Low delta, sufficient gamma, low theta, bearish technicals
        
        **Technical Indicators:**
        - EMA crossovers for trend direction
        - RSI for momentum
        - VWAP for intraday sentiment
        - Volume analysis for confirmation
        """)
