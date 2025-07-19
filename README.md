© 2025 [Your Name]. All rights reserved.
This project is licensed under the MIT License. Please provide attribution if you reuse any part of this code.
This repository is not for commercial use without permission. Contact me at [youssefsbai83@gmail.com].
# 📈 Options Greeks Buy Signal Analyzer

This Streamlit app analyzes options contracts using their Greeks (Delta, Gamma, Theta, Vega) to determine potential **Buy signals** for Calls and Puts.

## 🚀 Features
- Input a stock ticker symbol (e.g., IWM, AAPL, SPY)
- Select an expiration date and strike price
- Fetches live options data via Yahoo Finance (`yfinance`)
- Displays key Greeks: Delta, Gamma, Theta, Vega
- Gives a buy signal if:
  - **Call:** Delta ≥ 0.6, Gamma ≥ 0.1, Theta ≤ 0.03
  - **Put:** Delta ≤ -0.6, Gamma ≥ 0.1, Theta ≤ 0.03

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the App

```bash
streamlit run options_analyzer.py
```

## 📊 Example Use Cases

- Identify strong directional setups for 0DTE trades
- Combine with technical analysis to build automated entry signals

## 🔧 Future Improvements

- Integrate technical indicators (EMA, RSI, VWAP)
- Real-time charting with option volume overlays
- Multi-leg option analysis (spreads, straddles)

## 📚 Disclaimer

This tool is for educational purposes only and does not constitute financial advice.
