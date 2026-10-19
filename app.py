import math
import os
import re

from flask import Flask, jsonify, render_template, request
from ta.trend import SMAIndicator

from data_engine import fetch_stock_data
from inference import ModelUnavailable, evaluate_model, generate_forecast

app = Flask(__name__)

ALLOWED_MODELS = ("LSTM", "XGBoost", "ARIMA")
TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")


def clean_float(val):
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return float(val)


def growth_pct(forecast, current_price):
    return ((forecast[-1] - current_price) / current_price) * 100


def decide_vote(votes):
    """Majority vote. A tie (e.g. 1 Buy vs 1 Sell) is not a majority, so it resolves to Hold."""
    top = max(votes.values())
    winners = [k for k, v in votes.items() if v == top]
    return winners[0] if len(winners) == 1 else "Hold"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Request body must be a JSON object'}), 400

        ticker = str(data.get('ticker', '')).strip().upper()
        selected_model = data.get('model', 'LSTM')

        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400
        if not TICKER_RE.match(ticker):
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        if selected_model not in ALLOWED_MODELS:
            return jsonify({'error': f"Unknown model. Choose from {', '.join(ALLOWED_MODELS)}"}), 400

        stock_df = fetch_stock_data(ticker)
        current_price = float(stock_df.iloc[-1]['Close'])
        prev_price = float(stock_df['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price) * 100

        theme_color = {'LSTM': '#a855f7', 'XGBoost': '#f97316'}.get(selected_model, '#3b82f6')

        # --- chart data ---
        stock_df['SMA_25'] = SMAIndicator(close=stock_df['Close'], window=25).sma_indicator()
        ohlc_data, ma_data = [], []
        for index, row in stock_df.tail(90).iterrows():
            ts = int(index.timestamp() * 1000)
            ohlc_data.append({'x': ts, 'y': [clean_float(row['Open']), clean_float(row['High']),
                                              clean_float(row['Low']), clean_float(row['Close'])]})
            if not math.isnan(row['SMA_25']):
                ma_data.append({'x': ts, 'y': clean_float(row['SMA_25'])})

        # --- selected model forecast ---
        try:
            _, _, _, forecast = generate_forecast(stock_df, selected_model)
        except ModelUnavailable as e:
            return jsonify({'error': str(e)}), 400

        prediction_line = [{'x': ohlc_data[-1]['x'], 'y': ohlc_data[-1]['y'][3]}]
        for i, pred_price in enumerate(forecast):
            prediction_line.append({'x': ohlc_data[-1]['x'] + (i + 1) * 86400000,
                                    'y': clean_float(pred_price)})

        projected_growth = growth_pct(forecast, current_price)
        if projected_growth > 2.0:
            signal, gauge_val, signal_color = "Strong Buy", min(85 + projected_growth, 100), "#22c55e"
        elif projected_growth > 0:
            signal, gauge_val, signal_color = "Hold", 60, "#eab308"
        else:
            signal, gauge_val, signal_color = "Sell", max(30 + projected_growth, 10), "#ef4444"

        # --- consensus: every model's signal plus its MEASURED hold-out accuracy ---
        consensus_data = []
        votes = {'Buy': 0, 'Hold': 0, 'Sell': 0}
        for m in ALLOWED_MODELS:
            try:
                _, _, _, m_forecast = generate_forecast(stock_df, m)
                metrics = evaluate_model(stock_df, m)
            except ModelUnavailable:
                consensus_data.append({'name': m, 'accuracy': 'n/a', 'growth': 'n/a',
                                       'sentiment': 'n/a', 'signal': 'Unavailable'})
                continue
            m_growth = growth_pct(m_forecast, current_price)
            m_sig = "Buy" if m_growth > 2.0 else "Hold" if m_growth > 0 else "Sell"
            votes[m_sig] += 1
            consensus_data.append({
                'name': m,
                'accuracy': f"RMSE ${metrics['rmse']:.2f} | Dir {metrics['directional_accuracy']:.0%}",
                'growth': f"{m_growth:+.2f}%",
                # No sentiment source is integrated yet; do not present made-up numbers.
                'sentiment': 'Not integrated',
                'signal': m_sig,
            })

        final_vote = decide_vote(votes)
        vote_color = "#22c55e" if final_vote == "Buy" else "#ef4444" if final_vote == "Sell" else "#eab308"

        # Static demo panel (not live data)
        trending = [
            {"sym": "AAPL", "name": "Apple Inc.", "price": 175.43, "change": "+1.2%", "color": "text-success"},
            {"sym": "MSFT", "name": "Microsoft", "price": 412.10, "change": "+2.4%", "color": "text-success"},
            {"sym": "TSLA", "name": "Tesla", "price": 168.20, "change": "-0.8%", "color": "text-danger"},
        ]

        return jsonify({
            'ticker': ticker,
            'current_price': f"${current_price:.2f}",
            'change_pct': f"{change_pct:+.2f}%",
            'historical': ohlc_data,
            'maData': ma_data,
            'forecast': prediction_line,
            'model': selected_model,
            'theme_color': theme_color,
            'projected_growth': f"{projected_growth:+.2f}%",
            'signal_text': signal,
            'signal_val': int(gauge_val),
            'signal_color': signal_color,
            'trending': trending,
            'consensus': consensus_data,
            'final_vote': final_vote,
            'vote_color': vote_color,
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:  # noqa: BLE001
        print(f"System Error: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
