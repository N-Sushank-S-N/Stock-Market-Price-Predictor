# train_arima.py - Contains the logic for running the ARIMA model.

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def run_arima(df, test_size=30):
    """
    Runs an ARIMA model on the stock data.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        test_size (int): Number of days to use as a test set.
    
    Returns:
        tuple: (actual, predicted, labels, rmse)
    """
    try:
        data = df['Close'].values
        
        train_data = data[:-test_size]
        test_data = data[-test_size:]
        
        # Train ARIMA model
        model = ARIMA(train_data, order=(5, 1, 0))
        model_fit = model.fit()
        
        # Get predictions for the test set
        test_predictions = model_fit.forecast(steps=test_size)
        
        actual_values = test_data
        rmse = calculate_rmse(actual_values, test_predictions)
        
        labels = [d.strftime('%Y-%m-%d') for d in df.index[-test_size:]]
        
        return actual_values, test_predictions, labels, rmse

    except Exception as e:
        print(f"Error in run_arima: {str(e)}")
        raise e
