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

from plotly.subplots import make_subplots



# Suppress future warnings

warnings.filterwarnings('ignore', category=FutureWarning)

 st.set_page_config(
    page_title="Real-Time Options Greeks Analyzer",
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

    'CACHE_TTL': 15,  # Reduced to 15 seconds for more real-time feel

    'REAL_TIME_INTERVAL': 5,  # 5 seconds for real-time updates

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

# PROFIT ANALYSIS FUNCTIONS

# =============================



def calculate_profit_potential(option_data: pd.Series, current_price: float, side: str) -> Dict:

    """Calculate potential profit scenarios for options"""

    try:

        strike = float(option_data['strike'])

        premium = float(option_data['lastPrice'])

        delta = float(option_data['delta'])

        gamma = float(option_data['gamma'])

        theta = float(option_data['theta'])

        

        # Calculate price movement scenarios

        scenarios = []

        price_moves = [-5, -3, -1, 0, 1, 3, 5]  # Percentage moves

        

        for move in price_moves:

            new_price = current_price * (1 + move/100)

            

            if side == 'call':

                # Simplified profit calculation for calls

                intrinsic_value = max(0, new_price - strike)

                # Rough approximation of option value change using delta

                option_value_change = delta * (new_price - current_price)

                new_option_price = premium + option_value_change

                profit = (new_option_price - premium) * 100  # Per contract

                profit_pct = (profit / (premium * 100)) * 100 if premium > 0 else 0

            else:

                # Simplified profit calculation for puts

                intrinsic_value = max(0, strike - new_price)

                option_value_change = delta * (new_price - current_price)

                new_option_price = premium + option_value_change

                profit = (new_option_price - premium) * 100  # Per contract

                profit_pct = (profit / (premium * 100)) * 100 if premium > 0 else 0

            

            scenarios.append({

                'price_move': f"{move:+.1f}%",

                'new_price': new_price,

                'estimated_option_price': max(0.01, new_option_price),

                'profit_per_contract': profit,

                'profit_percentage': profit_pct

            })

        

        # Find breakeven

        if side == 'call':

            breakeven = strike + premium

            breakeven_move = ((breakeven - current_price) / current_price) * 100

        else:

            breakeven = strike - premium

            breakeven_move = ((breakeven - current_price) / current_price) * 100

        

        return {

            'scenarios': scenarios,

            'breakeven_price': breakeven,

            'breakeven_move': breakeven_move,

            'max_loss': premium * 100,  # Maximum loss per contract

            'cost_per_contract': premium * 100

        }

    except Exception as e:

        return {'error': str(e)}



def get_top_profit_opportunities(calls_df: pd.DataFrame, puts_df: pd.DataFrame, 

                               current_price: float, top_n: int = 5) -> Dict:

    """Identify top profit opportunities"""

    opportunities = []

    

    # Analyze calls

    for _, call in calls_df.iterrows():

        profit_analysis = calculate_profit_potential(call, current_price, 'call')

        if 'error' not in profit_analysis:

            # Get profit for +3% move scenario

            scenario_3pct = next((s for s in profit_analysis['scenarios'] if s['price_move'] == '+3.0%'), None)

            if scenario_3pct:

                opportunities.append({

                    'type': 'CALL',

                    'contract': call['contractSymbol'],

                    'strike': call['strike'],

                    'premium': call['lastPrice'],

                    'delta': call['delta'],

                    'gamma': call['gamma'],

                    'theta': call['theta'],

                    'profit_3pct': scenario_3pct['profit_per_contract'],

                    'profit_pct_3pct': scenario_3pct['profit_percentage'],

                    'breakeven_move': profit_analysis['breakeven_move'],

                    'max_loss': profit_analysis['max_loss'],

                    'moneyness': classify_moneyness(call['strike'], current_price)

                })

    

    # Analyze puts

    for _, put in puts_df.iterrows():

        profit_analysis = calculate_profit_potential(put, current_price, 'put')

        if 'error' not in profit_analysis:

            # Get profit for -3% move scenario

            scenario_neg3pct = next((s for s in profit_analysis['scenarios'] if s['price_move'] == '-3.0%'), None)

            if scenario_neg3pct:

                opportunities.append({

                    'type': 'PUT',

                    'contract': put['contractSymbol'],

                    'strike': put['strike'],

                    'premium': put['lastPrice'],

                    'delta': put['delta'],

                    'gamma': put['gamma'],

                    'theta': put['theta'],

                    'profit_3pct': scenario_neg3pct['profit_per_contract'],

                    'profit_pct_3pct': scenario_neg3pct['profit_percentage'],

                    'breakeven_move': profit_analysis['breakeven_move'],

                    'max_loss': profit_analysis['max_loss'],

                    'moneyness': classify_moneyness(put['strike'], current_price)

                })

    

    # Sort by profit percentage

    opportunities.sort(key=lambda x: x['profit_pct_3pct'], reverse=True)

    

    return {

        'top_calls': [opp for opp in opportunities[:top_n] if opp['type'] == 'CALL'],

        'top_puts': [opp for opp in opportunities[:top_n] if opp['type'] == 'PUT'],

        'all_opportunities': opportunities[:top_n]

    }



# =============================

# ORIGINAL UTILITY FUNCTIONS (Enhanced)

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

            interval="1m",  # Changed to 1-minute for more real-time data

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

        

        return data.reset_index()  # Keep timestamp index for real-time display

        

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

        else:  # put

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

# REAL-TIME FUNCTIONALITY

# =============================



def init_real_time_mode():

    """Initialize real-time mode"""

    if 'real_time_active' not in st.session_state:

        st.session_state.real_time_active = False

    if 'last_update' not in st.session_state:

        st.session_state.last_update = time.time()

    if 'update_counter' not in st.session_state:

        st.session_state.update_counter = 0



def create_real_time_chart(df: pd.DataFrame, ticker: str) -> go.Figure:

    """Create real-time price chart"""

    fig = make_subplots(

        rows=2, cols=1,

        subplot_titles=[f'{ticker} Real-Time Price', 'Volume'],

        vertical_spacing=0.1,

        row_heights=[0.7, 0.3]

    )

    

    # Price chart

    fig.add_trace(

        go.Scatter(

            x=df['Datetime'] if 'Datetime' in df.columns else df.index,

            y=df['Close'],

            mode='lines',

            name='Price',

            line=dict(color='#00ff00', width=2)

        ),

        row=1, col=1

    )

    

    # Add EMAs if available

    if 'EMA_9' in df.columns and not df['EMA_9'].isna().all():

        fig.add_trace(

            go.Scatter(

                x=df['Datetime'] if 'Datetime' in df.columns else df.index,

                y=df['EMA_9'],

                mode='lines',

                name='EMA 9',

                line=dict(color='#ff6b6b', width=1)

            ),

            row=1, col=1

        )

    

    if 'EMA_20' in df.columns and not df['EMA_20'].isna().all():

        fig.add_trace(

            go.Scatter(

                x=df['Datetime'] if 'Datetime' in df.columns else df.index,

                y=df['EMA_20'],

                mode='lines',

                name='EMA 20',

                line=dict(color='#4ecdc4', width=1)

            ),

            row=1, col=1

        )

    

    # Volume chart

    fig.add_trace(

        go.Bar(

            x=df['Datetime'] if 'Datetime' in df.columns else df.index,

            y=df['Volume'],

            name='Volume',

            marker_color='rgba(0,255,0,0.3)'

        ),

        row=2, col=1

    )

    

    fig.update_layout(

        title=f'{ticker} Real-Time Analysis',

        xaxis_title='Time',

        yaxis_title='Price ($)',

        height=600,

        showlegend=True,

        template='plotly_dark'

    )

    

    return fig



# =============================

# STREAMLIT INTERFACE

# =============================



st.title("🚀 Real-Time Options Greeks & Profit Analyzer")

st.markdown("**Enhanced with real-time updates and profit analysis**")



# Initialize real-time mode

init_real_time_mode()



# Sidebar for configuration

with st.sidebar:

    st.header("⚙️ Configuration")

    

    # Real-time settings

    st.subheader("🔄 Real-Time Settings")

    real_time_mode = st.checkbox("Enable Real-Time Mode", value=True)

    

    if real_time_mode:

        update_interval = st.selectbox(

            "Update Interval",

            options=[5, 10, 15, 30],

            index=0,

            format_func=lambda x: f"{x} seconds"

        )

        st.info(f"🔄 Updates every {update_interval} seconds")

        st.session_state.real_time_active = True

    else:

        st.session_state.real_time_active = False

    

    # Profit analysis settings

    st.subheader("💰 Profit Analysis")

    show_profit_analysis = st.checkbox("Show Profit Analysis", value=True)

    profit_scenarios = st.multiselect(

        "Price Movement Scenarios",

        options=[-5, -3, -1, 0, 1, 3, 5],

        default=[-3, -1, 1, 3],

        format_func=lambda x: f"{x:+.0f}%"

    )

    

    # Signal thresholds (existing code)

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

ticker = st.text_input("Enter Stock Ticker (e.g., SPY, QQQ, AAPL):", value="SPY").upper()



if ticker:

    # Real-time status indicator

    status_placeholder = st.empty()

    

    # Auto-refresh logic for real-time mode

    if real_time_mode:

        current_time = time.time()

        time_elapsed = current_time - st.session_state.last_update

        

        if time_elapsed >= update_interval:

            st.session_state.last_update = current_time

            st.session_state.update_counter += 1

            st.cache_data.clear()

            st.rerun()

    

    # Status display

    with status_placeholder.container():

        col1, col2, col3, col4 = st.columns(4)

        

        with col1:

            if real_time_mode:

                st.success("🟢 LIVE")

            else:

                st.info("🔴 MANUAL")

        

        with col2:

            st.caption(f"Updates: {st.session_state.update_counter}")

        

        with col3:

            st.caption(f"Last: {datetime.datetime.now().strftime('%H:%M:%S')}")

        

        with col4:

            if not real_time_mode:

                if st.button("🔄 Refresh"):

                    st.cache_data.clear()

                    st.rerun()

    

    # Create tabs for better organization

    tab1, tab2, tab3, tab4 = st.tabs(["💰 Profit Opportunities", "📊 Signals", "📈 Real-Time Chart", "⚙️ Details"])

    

    with tab1:

        if show_profit_analysis:

            try:

                with st.spinner("Analyzing profit opportunities..."):

                    # Get stock data

                    df = get_stock_data(ticker)

                    

                    if df.empty:

                        st.error("Unable to fetch stock data.")

                        st.stop()

                    

                    # Compute indicators

                    df = compute_indicators(df)

                    current_price = df.iloc[-1]['Close']

                    

                    # Get options data

                    expiries = get_options_expiries(ticker)

                    if not expiries:

                        st.error("No options expiries available.")

                        st.stop()

                    

                    # Use 0DTE and next expiry for profit analysis

                    today = datetime.date.today()

                    dte_expiries = [e for e in expiries if datetime.datetime.strptime(e, "%Y-%m-%d").date() == today]

                    if not dte_expiries:

                        dte_expiries = expiries[:2]

                    

                    calls, puts = fetch_options_data(ticker, dte_expiries)

                    

                    if calls.empty and puts.empty:

                        st.error("No options data available.")

                        st.stop()

                    

                    # Filter by strike range

                    strike_range = 20  # +/- $20 from current price

                    min_strike = current_price - strike_range

                    max_strike = current_price + strike_range

                    

                    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]

                    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

                    

                    # Get top profit opportunities

                    opportunities = get_top_profit_opportunities(calls_filtered, puts_filtered, current_price)

                    

                    # Display profit opportunities

                    st.subheader(f"💰 Top Profit Opportunities")
