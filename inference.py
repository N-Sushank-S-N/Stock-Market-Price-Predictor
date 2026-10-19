"""Real forecasting models (ARIMA, XGBoost, LSTM) behind one interface.

Replaces the earlier simulated/random-noise inference. Public API is unchanged:
    generate_forecast(stock_data, model_type, steps=30)
        -> (past_dates, past_prices, forecast_dates, forecast)
plus a new evaluate_model() that measures real hold-out accuracy.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

LAGS = 10            # XGBoost lag window (days of past returns)
LSTM_WINDOW = 60     # LSTM look-back window
MIN_HISTORY = 120    # minimum rows needed to train anything sensible
SUPPORTED = ("ARIMA", "XGBoost", "LSTM")

_CACHE = {}


class ModelUnavailable(RuntimeError):
    """Raised when a model's optional dependency (e.g. TensorFlow) is not installed."""


def _close_array(stock_data) -> np.ndarray:
    y = np.asarray(stock_data["Close"].values, dtype=float).ravel()
    if np.isnan(y).any():
        y = pd.Series(y).ffill().bfill().values
    return y


# ---------------------------------------------------------------- forecasters
def _forecast_arima(y: np.ndarray, steps: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    fit = ARIMA(y, order=(5, 1, 0)).fit()
    return np.asarray(fit.forecast(steps=steps), dtype=float)


def _returns(y):
    return np.diff(y) / y[:-1]


def _forecast_xgboost(y: np.ndarray, steps: int) -> np.ndarray:
    """Predict the next daily *return* from the previous LAGS returns, then roll forward.

    Trees cannot extrapolate price levels, so modelling returns is the standard fix.
    """
    import xgboost as xgb
    r = _returns(y)
    X = np.array([r[i - LAGS:i] for i in range(LAGS, len(r))])
    t = r[LAGS:]
    model = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                             subsample=0.8, random_state=42, objective="reg:squarederror")
    model.fit(X, t)
    hist, price, out = list(r), float(y[-1]), []
    for _ in range(steps):
        nxt = float(model.predict(np.array(hist[-LAGS:]).reshape(1, -1))[0])
        price *= (1.0 + nxt)
        out.append(price)
        hist.append(nxt)
    return np.array(out)


def _forecast_lstm(y: np.ndarray, steps: int) -> np.ndarray:
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense, Input
        from tensorflow.keras.models import Sequential
    except ImportError as e:
        raise ModelUnavailable("TensorFlow is not installed; LSTM is unavailable") from e
    from sklearn.preprocessing import MinMaxScaler

    tf.keras.utils.set_random_seed(42)
    scaler = MinMaxScaler()
    s = scaler.fit_transform(y.reshape(-1, 1))
    W = LSTM_WINDOW
    X = np.array([s[i - W:i, 0] for i in range(W, len(s))]).reshape(-1, W, 1)
    t = s[W:, 0]
    model = Sequential([Input(shape=(W, 1)), LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, t, batch_size=32, epochs=10, verbose=0)
    window, out = list(s[-W:, 0]), []
    for _ in range(steps):
        p = float(model.predict(np.array(window[-W:]).reshape(1, W, 1), verbose=0)[0, 0])
        out.append(p)
        window.append(p)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()


FORECASTERS = {"ARIMA": _forecast_arima, "XGBoost": _forecast_xgboost, "LSTM": _forecast_lstm}


def _validate(model_type: str, y: np.ndarray):
    if model_type not in FORECASTERS:
        raise ValueError(f"Unknown model '{model_type}'. Choose from {', '.join(SUPPORTED)}.")
    if len(y) < MIN_HISTORY:
        raise ValueError(f"Not enough price history ({len(y)} rows); need at least {MIN_HISTORY}.")


def _cached_forecast(model_type: str, y: np.ndarray, steps: int) -> np.ndarray:
    key = (model_type, steps, len(y), round(float(y[-1]), 4), round(float(y[0]), 4))
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = np.maximum(FORECASTERS[model_type](y, steps), 0.01)  # no negative prices
    return _CACHE[key]


# ------------------------------------------------------------------ public API
def generate_forecast(stock_data, model_type, steps=30):
    y = _close_array(stock_data)
    _validate(model_type, y)
    forecast = _cached_forecast(model_type, y, steps).tolist()
    past_dates = [d.strftime("%Y-%m-%d") for d in stock_data.index[-60:]]
    past_prices = y[-60:].tolist()
    last_date = stock_data.index[-1]
    forecast_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, steps + 1)]
    return past_dates, past_prices, forecast_dates, forecast


def evaluate_model(stock_data, model_type, test_size=30) -> dict:
    """Honest hold-out evaluation: train on everything except the last `test_size` days,
    forecast those days, compare with what actually happened.

    Returns rmse, mape (%), and directional accuracy (share of days where the forecast
    moved the same way as the actual price relative to the last training price).
    """
    y = _close_array(stock_data)
    _validate(model_type, y)
    train, actual = y[:-test_size], y[-test_size:]
    pred = _cached_forecast(model_type, train, test_size)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    direction = float(np.mean(np.sign(pred - train[-1]) == np.sign(actual - train[-1])))
    return {"rmse": rmse, "mape": mape, "directional_accuracy": direction, "test_size": test_size}
