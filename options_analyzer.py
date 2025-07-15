import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from polygon import RESTClient
from alpha_vantage.timeseries import TimeSeries
import iexfinance as iex
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
import time
from typing import Tuple, Dict  # Added missing imports

# Configuration
CONFIG = {
    'CACHE_TTL': 3600,  # 1 hour
    'MAX_OPTION_CONTRACTS': 50,
    'RSI_PERIOD': 14,
    'POLYGON_API_KEY': st.secrets.get("POLYGON_API_KEY", None),
    'ALPHA_VANTAGE_API_KEY': st.secrets.get("ALPHA_VANTAGE_API_KEY", None),
    'IEX_API_KEY': st.secrets.get("IEX_API_KEY", None)
}

# Initialize API clients
polygon_client = RESTClient(CONFIG['POLYGON_API_KEY']) if CONFIG['POLYGON_API_KEY'] else None

# Data fetching functions
@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_stock_data_av(ticker: str) -> pd.DataFrame:
    if not CONFIG['ALPHA_VANTAGE_API_KEY']:
        st.error("Alpha Vantage API key not configured")
        return pd.DataFrame()
    
    ts = TimeSeries(key=CONFIG['ALPHA_VANTAGE_API_KEY'], output_format='pandas')
    try:
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='compact')
        data = data.iloc[::-1]  # Reverse to chronological order
        data.columns = [col.split('. ')[1] for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Alpha Vantage error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CONFIG['CACHE_TTL'])
def get_options_data_polygon(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not polygon_client:
        st.error("Polygon client not initialized. Check API key and configuration.")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Get current stock price
        aggs = polygon_client.get_previous_close_agg(ticker)
        current_price = aggs.results[0].c if aggs.results else 0
        
        # Get options chain
        options_chain = polygon_client.list_options_contracts(
            underlying_ticker=ticker,
            expired=False,
            limit=CONFIG['MAX_OPTION_CONTRACTS']
        )
        
        # Process options data
        calls, puts = [], []
        for opt in options_chain:
            if opt.right == 'call':
                calls.append({
                    'strike': opt.strike_price,
                    'expiry': opt.expiration_date,
                    'last_price': opt.last_trade_price,
                    'volume': opt.day_trade_volume,
                    'open_interest': opt.open_interest,
                    'iv': opt.implied_volatility
                })
            elif opt.right == 'put':
                puts.append({
                    'strike': opt.strike_price,
                    'expiry': opt.expiration_date,
                    'last_price': opt.last_trade_price,
                    'volume': opt.day_trade_volume,
                    'open_interest': opt.open_interest,
                    'iv': opt.implied_volatility
                })
                
        calls_df = pd.DataFrame(calls)
        puts_df = pd.DataFrame(puts)
        
        # Calculate moneyness
        if not calls_df.empty:
            calls_df['moneyness'] = calls_df['strike'] / current_price - 1
        if not puts_df.empty:
            puts_df['moneyness'] = puts_df['strike'] / current_price - 1
            
        return calls_df, puts_df
        
    except Exception as e:
        st.error(f"Polygon options error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def generate_signal(option: pd.Series, side: str, stock_df: pd.DataFrame) -> Dict:
    if stock_df.empty or any(pd.isna(option.get(greek)) for greek in ['delta', 'theta', 'iv']):
        return {'signal': False, 'reason': 'Insufficient data'}
    
    latest = stock_df.iloc[-1]
    signal = False
    reason = []
    
    # Common criteria
    if option['volume'] < 100:
        reason.append('Low volume')
    if option['open_interest'] < 50:
        reason.append('Low OI')
    
    # Side-specific criteria
    if side == 'call':
        if option['delta'] > 0.6 and option['theta'] < -0.05 and option['iv'] > 0.3:
            signal = True
            reason.append('High delta, theta decay, IV')
    elif side == 'put':
        if option['delta'] < -0.6 and option['theta'] < -0.05 and option['iv'] > 0.3:
            signal = True
            reason.append('High delta, theta decay, IV')
    
    return {
        'signal': signal,
        'reason': ' | '.join(reason) if reason else 'No criteria met'
    }

# Streamlit App
def main():
    st.set_page_config(page_title="Options Analyzer", layout="wide")
    st.title("📊 Options Trading Signals Analyzer")
    
    # Ticker input
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    
    # Fixed refresh button syntax
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    if not ticker:
        st.warning("Please enter a stock ticker")
        return
    
    # Fetch data
    with st.spinner("Fetching data..."):
        stock_df = get_stock_data_av(ticker)
        calls_df, puts_df = get_options_data_polygon(ticker)
    
    # Show error if no data
    if stock_df.empty:
        st.error("Failed to fetch stock data")
        return
    if calls_df.empty and puts_df.empty:
        st.error("Failed to fetch options data")
        return
    
    # Display stock data
    st.subheader(f"{ticker} Price History")
    st.line_chart(stock_df['close'])
    
    # Calculate RSI
    rsi_indicator = RSIIndicator(close=stock_df['close'], window=CONFIG['RSI_PERIOD'])
    stock_df['RSI'] = rsi_indicator.rsi()
    
    # Display RSI
    st.subheader("RSI Indicator")
    st.line_chart(stock_df['RSI'])
    st.markdown(f"**Current RSI:** {stock_df['RSI'].iloc[-1]:.2f}")
    
    # Generate signals
    call_signals, put_signals = [], []
    
    # Process call options
    if not calls_df.empty:
        for _, row in calls_df.iterrows():
            signal = generate_signal(row, 'call', stock_df)
            if signal['signal']:
                call_signals.append({
                    'Strike': row['strike'],
                    'Expiry': row['expiry'],
                    'Signal Reason': signal['reason'],
                    'IV': row['iv'],
                    'Volume': row['volume']
                })
    
    # Process put options
    if not puts_df.empty:
        for _, row in puts_df.iterrows():
            signal = generate_signal(row, 'put', stock_df)
            if signal['signal']:
                put_signals.append({
                    'Strike': row['strike'],
                    'Expiry': row['expiry'],
                    'Signal Reason': signal['reason'],
                    'IV': row['iv'],
                    'Volume': row['volume']
                })
    
    # Display signals - fixed indentation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Call Option Signals")
        if call_signals:
            st.dataframe(pd.DataFrame(call_signals))
        else:
            st.info("No call signals found matching criteria")
    
    with col2:
        st.subheader("📉 Put Option Signals")
        if put_signals:
            st.dataframe(pd.DataFrame(put_signals))
        else:
            st.info("No put signals found matching criteria")
    
    # Additional debugging info
    st.subheader("Debugging Info")
    st.write(f"Stock data points: {len(stock_df)}")
    st.write(f"Call options: {len(calls_df)} | Put options: {len(puts_df)}")
    st.write(f"Call signals: {len(call_signals)} | Put signals: {len(put_signals)}")

if __name__ == "__main__":
    main()
