# 📈 StockPrediction — Enterprise Multi-Model Forecasting Terminal

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-TensorFlow%20%7C%20XGBoost%20%7C%20Statsmodels-orange.svg)](https://github.com/N-Sushank-S-N/Stock-Market-Price-Predictor)
[![Visualization](https://img.shields.io/badge/Frontend-ApexCharts.js-purple.svg)](https://apexcharts.com/)

**StockPrediction** is an enterprise-grade financial intelligence terminal designed to bridge the gap between raw, volatile time-series market data and actionable trading decisions. By cross-validating statistical, machine learning, and deep sequence models alongside simulated market sentiment, the terminal generates an algorithmic **Majority Vote Verdict** (Buy/Hold/Sell) to mitigate single-model predictive risk.

---

## 💡 The Core Idea & Engineering Motivation

Traditional automated stock forecasting systems often suffer from two major flaws:
1. **Single-Model Reliance:** Relying solely on one predictive paradigm (e.g., ARIMA or a single neural network) fails to account for regime shifts, non-linear volatility, and unexpected macro movements in financial markets.
2. **Disconnected Insights:** Outputs are often raw mathematical numbers (like standard RMSE or price targets) with no synthesis into actionable decision-making for end users.

### The Solution: Multi-Paradigm Ensemble & Consensus
Rather than forcing a single model to capture all market dynamics, **StockPrediction** runs three mathematically distinct architectures concurrently:
* **ARIMA (AutoRegressive Integrated Moving Average):** Establishes the linear baseline trend continuation.
* **XGBoost (Extreme Gradient Boosting):** Reacts quickly to short-term volatility, technical indicator shifts (25-day SMA), and price lag features.
* **LSTM (Long Short-Term Memory Sequence Network):** Discovers non-linear dependencies and long-term momentum across extended lookback sequences.

The individual predictions are then processed by a **Majority Vote Engine** and combined with **Simulated NLP Web Sentiment** to output a single, high-confidence decision gauge.

---

## 🔄 Evolutionary Leap: Legacy V1 vs. Enterprise V2

The codebase was completely refactored from a multi-script research setup into a decoupled, production-ready micro-architecture. *(The original files are archived in the `v1-legacy` branch).*

| Architectural Dimension | Legacy V1 Prototype (`v1-legacy`) | Enterprise V2 Terminal (`main`) |
| :--- | :--- | :--- |
| **System Architecture** | Tightly coupled scripts (`train_arima.py`, `train_lstm.py`, `train_xgboost.py`). | Modular 3-tier micro-architecture: `data_engine.py` (Ingestion), `inference.py` (Model Engine), and `app.py` (REST API). |
| **Decision Logic** | Ran individual validation and picked one single "best" model based on RMSE. | **Triple-Model Consensus Engine**: Evaluates all models simultaneously to deliver a collective **Majority Vote Verdict**. |
| **Market Sentiment** | Price-action only; no contextual qualitative data. | **Simulated NLP Sentiment Ingestion**: Quantifies public and forum sentiment to weigh against numerical forecasts. |
| **Data Pipeline Resilience** | Direct `yfinance` calls prone to crashes from schema updates. | **Defensive Data Engine**: Features `@lru_cache` optimization and an automated **MultiIndex flattener**. |
| **User Interface** | Basic charts; prone to freezing on weekends or market holidays. | **Interactive ApexCharts.js**: Real-time zoomable OHLC candlesticks, volume overlays, model-specific color theming, and weekend fallbacks. |

---

## 🛠️ Architecture & Pipeline Flow

```text
[ yfinance API ] ──► [ data_engine.py ] (MultiIndex Flattener + @lru_cache)
                             │
                             ▼
                     [ inference.py ] 
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
 [ ARIMA Baseline ]   [ XGBoost + SMA ]     [ LSTM Sequence ]
  (Linear Trend)       (Short-Term Vol)      (Momentum Shifts)
       └─────────────────────┬─────────────────────┘
                             ▼
              [ Majority Vote Consensus Engine ]
               + Simulated NLP Market Sentiment
                             │
                             ▼
            [ Flask REST API Endpoint (/predict) ]
                             │
                             ▼
             [ ApexCharts.js Dynamic Dashboard ]


🚀 Key Features
Multi-Model Inference: Generates 30-day projection trajectories across statistical, boosting, and deep learning algorithms.

Majority Vote Consensus: Aggregates directional signals (Buy, Hold, Sell) from all active models with confidence scoring.

Defensive Data Engine: Built with a custom column normalizer to handle Yahoo Finance schema changes and @lru_cache(maxsize=100) for sub-second retrieval.

Market-Aware Fallbacks: Handles weekend and market holiday queries gracefully by defaulting to the latest closing metrics with active UI status alerts.

Dynamic Visualization & Theming: Interactive candlestick, volume, and SMA panels with automatic color shifts (Neon Purple for LSTM, Orange for XGBoost, Blue for ARIMA) and full Dark/Light mode support.

📁 Repository Structure
Stock-Market-Price-Predictor/
├── app.py                  # Main Flask application router & /predict REST API
├── data_engine.py          # Data ingestion, @lru_cache, and MultiIndex flattener
├── inference.py            # Model inference, sequence preparation & forecast simulations
├── templates/
│   └── index.html          # Interactive frontend terminal with ApexCharts.js
├── static/
│   ├── css/                # Custom theming and UI glow effects
│   └── js/                 # Client-side state handling and async fetch requests
├── requirements.txt        # Python dependency manifest
├── .gitignore              # Environment and cache ignore rules
└── README.md               # Project documentation

⚡ Quickstart & Installation

1. Clone the Repository
git clone [https://github.com/N-Sushank-S-N/Stock-Market-Price-Predictor.git](https://github.com/N-Sushank-S-N/Stock-Market-Price-Predictor.git)
cd Stock-Market-Price-Predictor

2. Set Up Virtual Environment
# Windows
python -m venv venv
venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Terminal
Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000.

🖥️ Usage
Enter a valid equity ticker (e.g., AAPL, NVDA, TSLA, MSFT).

Choose your primary analytical view or inspect the Aggregate Model Consensus table.

Analyze the interactive 30-day forecast, moving average crossovers, and final Majority Vote Verdict.