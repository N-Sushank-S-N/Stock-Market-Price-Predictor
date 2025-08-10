# train_xgboost.py - Contains the logic for running the XGBoost model.

import yfinance as yf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def run_xgboost(df, test_size=30):
    """
    Runs an XGBoost model on the stock data using OHLCV features.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV prices.
        test_size (int): Number of days to use as a test set.
    
    Returns:
        tuple: (actual, predicted, labels, rmse)
    """
    try:
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
        
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = df['Target'].values
        
        train_size = len(X) - test_size
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        rmse = calculate_rmse(y_test, predictions)
        
        labels = [d.strftime('%Y-%m-%d') for d in df.index[train_size:]]
        
        return y_test, predictions, labels, rmse

    except Exception as e:
        print(f"Error in run_xgboost: {str(e)}")
        raise e
