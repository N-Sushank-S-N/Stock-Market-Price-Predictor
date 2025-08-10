import numpy as np

# Placeholder functions - replace with your actual implementations
def run_arima(df):
    if df.empty or 'Close' not in df.columns:
        return [], [], [], 9999  # fail-safe

    try:
        close_prices = df['Close']  # Ensure we're working with a Series
        actual = close_prices[-30:].tolist()
        predicted = [p * 1.01 for p in actual]  # dummy prediction logic
        labels = df.index[-30:].strftime('%Y-%m-%d').tolist()

        from sklearn.metrics import mean_squared_error
        from math import sqrt
        rmse = sqrt(mean_squared_error(actual, predicted))

        return actual, predicted, labels, rmse
    except Exception as e:
        print(f"ARIMA error: {e}")
        return [], [], [], 9999



def run_lstm(df):
    if df.empty or 'Close' not in df.columns:
        return [], [], [], 9999  # fail-safe

    try:
        close_prices = df['Close']  # Ensure we're working with a Series
        actual = close_prices[-30:].tolist()
        predicted = [p * 1.01 for p in actual]  # dummy prediction logic
        labels = df.index[-30:].strftime('%Y-%m-%d').tolist()

        from sklearn.metrics import mean_squared_error
        from math import sqrt
        rmse = sqrt(mean_squared_error(actual, predicted))

        return actual, predicted, labels, rmse
    except Exception as e:
        print(f"LSTM error: {e}")
        return [], [], [], 9999


def run_xgboost(df):
    if df.empty or 'Close' not in df.columns:
        return [], [], [], 9999  # fail-safe

    try:
        close_prices = df['Close']  # Ensure we're working with a Series
        actual = close_prices[-30:].tolist()
        predicted = [p * 1.01 for p in actual]  # dummy prediction logic
        labels = df.index[-30:].strftime('%Y-%m-%d').tolist()

        from sklearn.metrics import mean_squared_error
        from math import sqrt
        rmse = sqrt(mean_squared_error(actual, predicted))

        return actual, predicted, labels, rmse
    except Exception as e:
        print(f"XGBOOST error: {e}")
        return [], [], [], 9999

