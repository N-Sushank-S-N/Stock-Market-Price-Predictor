# train_lstm.py - Contains the logic for running the LSTM model.

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def run_lstm(df, test_size=30):
    """
    Runs an LSTM model on the stock data.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        test_size (int): Number of days to use as a test set.
    
    Returns:
        tuple: (actual, predicted, labels, rmse)
    """
    try:
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        train_data_len = len(scaled_data) - test_size
        train_data = scaled_data[:train_data_len]
        
        window_size = 60
        X_train, y_train = [], []
        for i in range(window_size, len(train_data)):
            X_train.append(train_data[i-window_size:i, 0])
            y_train.append(train_data[i, 0])
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(units=50),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        
        test_data = scaled_data[train_data_len - window_size:, :]
        X_test = []
        for i in range(window_size, len(test_data)):
            X_test.append(test_data[i-window_size:i, 0])
            
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predictions = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predictions).flatten()
        actual_prices = scaler.inverse_transform(scaled_data[train_data_len:]).flatten()
        
        rmse = calculate_rmse(actual_prices, predicted_prices)
        
        labels = [d.strftime('%Y-%m-%d') for d in df.index[train_data_len:]]
        
        return actual_prices, predicted_prices, labels, rmse
        
    except Exception as e:
        print(f"Error in run_lstm: {str(e)}")
        raise e
