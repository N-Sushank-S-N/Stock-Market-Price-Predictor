import numpy as np
from datetime import timedelta

def generate_forecast(stock_data, model_type, steps=30):
    """
    Simulates enterprise inference. In production, this loads .pkl or .h5 files.
    """
    closing_prices = stock_data['Close'].values.ravel().tolist()
    last_price = closing_prices[-1]
    
    # Grab the last 60 days for context
    past_dates = [d.strftime('%Y-%m-%d') for d in stock_data.index[-60:]]
    past_prices = closing_prices[-60:]
    
    forecast = []
    current_pred = last_price
    
    # Model Simulation Logic
    for i in range(steps):
        if model_type == 'ARIMA':
            # Statistical: Smooth, linear trend continuation
            trend = np.mean(past_prices[-10:]) - np.mean(past_prices[-20:])
            current_pred += (trend * 0.1) + np.random.normal(0, last_price * 0.005)
            
        elif model_type == 'XGBoost':
            # Boosting: More volatile, reactive to recent swings
            volatility = np.std(past_prices[-5:])
            current_pred += np.random.normal(0, volatility * 0.5)
            
        elif model_type == 'LSTM':
            # Deep Learning: Momentum based, learns from longer sequence
            momentum = np.mean(past_prices[-5:]) - np.mean(past_prices[-60:])
            current_pred += (momentum * 0.05) + np.random.normal(0, last_price * 0.008)
            
        forecast.append(max(current_pred, 1.0)) # Prevent negative prices
        
    last_date = stock_data.index[-1]
    forecast_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, steps + 1)]
    
    return past_dates, past_prices, forecast_dates, forecast