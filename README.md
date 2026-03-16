# 📈 NexusQuant — AI Stock Intelligence Platform

A professional AI-powered stock analysis dashboard built with **Streamlit**, featuring ML predictions, technical indicators, volatility analysis, and portfolio simulation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Category | Details |
|---|---|
| 🔍 **Company Search** | Search by name (Apple, Tesla, NVIDIA…) — auto-resolves to ticker |
| 📊 **3 ML Models** | Linear Regression · Random Forest · SVR |
| 🔮 **AI Predictions** | 30-day ensemble forecast with confidence bands |
| 📉 **Technical Indicators** | RSI · MACD · Bollinger Bands · MA50/200 |
| 🌊 **Volatility Analysis** | Daily returns distribution · Risk scoring |
| 🧪 **Backtesting** | Predicted vs actual with R² and RMSE |
| 💼 **Portfolio Simulator** | Single stock growth + multi-stock allocator |
| ⚡ **Buy/Hold/Sell Signal** | Ensemble signal based on trend analysis |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nexusquant.git
cd nexusquant
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
nexusquant/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark theme config
└── utils/
    ├── __init__.py
    ├── data_engine.py       # Data fetching, ML models, signals
    └── charts.py            # All Plotly chart builders
```

---

## 🛠 Tech Stack

- **Streamlit** — UI framework
- **yfinance** — Yahoo Finance data
- **scikit-learn** — ML models
- **Plotly** — Interactive charts
- **pandas / numpy** — Data processing
- **ta** — Technical analysis

---

## 📸 Screenshots

> Search "Apple" → get full analysis in seconds

**Overview** — Price chart, moving averages, KPIs, buy/sell signal  
**ML Models** — Train/test split, model comparison, R²/RMSE metrics  
**Technical** — Bollinger Bands, RSI, MACD  
**Predictions** — 30-day AI forecast, backtesting  
**Volatility** — Returns histogram, risk score  
**Portfolio** — Investment growth, multi-stock simulator  

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
