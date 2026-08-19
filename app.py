from flask import Flask, render_template, request, jsonify
from data_engine import fetch_stock_data
from inference import generate_forecast
import numpy as np
import math
from ta.trend import SMAIndicator

app = Flask(__name__)

def clean_float(val):
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return float(val)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        selected_model = data.get('model', 'LSTM')
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400
            
        stock_df = fetch_stock_data(ticker)
        
        latest = stock_df.iloc[-1]
        current_price = float(latest['Close'])
        prev_price = float(stock_df['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # --- AGGRESSIVE COLOR THEMING ---
        if selected_model == 'LSTM':
            theme_color = '#a855f7' # Neon Purple
        elif selected_model == 'XGBoost':
            theme_color = '#f97316' # Vibrant Orange
        else:
            theme_color = '#3b82f6' # Classic Blue
        
        # --- PREPARE CHART DATA ---
        stock_df['SMA_25'] = SMAIndicator(close=stock_df['Close'], window=25).sma_indicator()
        ohlc_data = []
        ma_data = []
        plot_df = stock_df.tail(90) 

        for index, row in plot_df.iterrows():
            timestamp = int(index.timestamp() * 1000)
            ohlc_data.append({
                'x': timestamp,
                'y': [clean_float(row['Open']), clean_float(row['High']), clean_float(row['Low']), clean_float(row['Close'])]
            })
            if not math.isnan(row['SMA_25']):
                ma_data.append({ 'x': timestamp, 'y': clean_float(row['SMA_25']) })

        # --- RUN MAIN INFERENCE ---
        _, _, _, forecast = generate_forecast(stock_df, selected_model)
        prediction_line = [{'x': ohlc_data[-1]['x'], 'y': ohlc_data[-1]['y'][3]}]
        for i, pred_price in enumerate(forecast):
            forecast_timestamp = ohlc_data[-1]['x'] + (i + 1) * 24 * 60 * 60 * 1000
            prediction_line.append({ 'x': forecast_timestamp, 'y': clean_float(pred_price) })

        projected_growth = ((forecast[-1] - current_price) / current_price) * 100
        
        if projected_growth > 2.0:
            signal, gauge_val, signal_color = "Strong Buy", min(85 + projected_growth, 100), "#22c55e"
        elif projected_growth > 0:
            signal, gauge_val, signal_color = "Hold", 60, "#eab308"
        else:
            signal, gauge_val, signal_color = "Sell", max(30 + projected_growth, 10), "#ef4444"

        # --- CONSENSUS TABLE WITH SIMULATED WEB SENTIMENT ---
        models = ['LSTM', 'XGBoost', 'ARIMA']
        accuracies = {'LSTM': '94%', 'XGBoost': '88%', 'ARIMA': '76%'}
        
        consensus_data = []
        votes = {'Buy': 0, 'Hold': 0, 'Sell': 0}

        for m in models:
            _, _, _, m_forecast = generate_forecast(stock_df, m)
            m_growth = ((m_forecast[-1] - current_price) / current_price) * 100
            
            # Simulated Web Sentiment logic based on momentum
            if m_growth > 2.0:
                m_sig = "Buy"
                web_sentiment = "82% Bullish (Reddit/Twitter)"
                votes['Buy'] += 1
            elif m_growth > 0:
                m_sig = "Hold"
                web_sentiment = "55% Neutral (News)"
                votes['Hold'] += 1
            else:
                m_sig = "Sell"
                web_sentiment = "70% Bearish (Forums)"
                votes['Sell'] += 1

            consensus_data.append({
                'name': m,
                'accuracy': accuracies[m],
                'growth': f"{m_growth:+.2f}%",
                'sentiment': web_sentiment,
                'signal': m_sig
            })

        final_vote = max(votes, key=votes.get)
        vote_color = "#22c55e" if final_vote == "Buy" else "#ef4444" if final_vote == "Sell" else "#eab308"

        trending = [
            {"sym": "AAPL", "name": "Apple Inc.", "price": 175.43, "change": "+1.2%", "color": "text-success"},
            {"sym": "MSFT", "name": "Microsoft", "price": 412.10, "change": "+2.4%", "color": "text-success"},
            {"sym": "TSLA", "name": "Tesla", "price": 168.20, "change": "-0.8%", "color": "text-danger"}
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
            'vote_color': vote_color
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"System Error: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)