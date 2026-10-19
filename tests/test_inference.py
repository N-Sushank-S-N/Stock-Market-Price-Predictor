import numpy as np
import pytest

import inference
from conftest import make_prices


@pytest.mark.parametrize("model", ["ARIMA", "XGBoost"])
def test_forecast_shape_and_positive_prices(prices, model):
    pd_, pp, fd, f = inference.generate_forecast(prices, model)
    assert len(f) == len(fd) == 30 and len(pd_) == len(pp) == 60
    assert min(f) > 0 and np.isfinite(f).all()


@pytest.mark.parametrize("model", ["ARIMA", "XGBoost"])
def test_forecast_is_deterministic(prices, model):
    a = inference.generate_forecast(prices, model)[3]
    inference._CACHE.clear()
    b = inference.generate_forecast(prices, model)[3]
    assert np.allclose(a, b)


def test_forecast_is_not_random_noise_around_last_price(prices):
    # ARIMA on a drifting series should not be a flat constant either
    f = inference.generate_forecast(prices, "ARIMA")[3]
    assert len(set(np.round(f, 6))) > 1


def test_custom_steps(prices):
    assert len(inference.generate_forecast(prices, "ARIMA", steps=10)[3]) == 10


def test_unknown_model_raises(prices):
    with pytest.raises(ValueError):
        inference.generate_forecast(prices, "Prophet")


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        inference.generate_forecast(make_prices(n=50), "ARIMA")


@pytest.mark.parametrize("model", ["ARIMA", "XGBoost"])
def test_evaluate_model_metrics_are_sane(prices, model):
    m = inference.evaluate_model(prices, model)
    assert m["rmse"] > 0 and 0 < m["mape"] < 50 and 0 <= m["directional_accuracy"] <= 1


def test_evaluation_never_sees_the_test_window(prices, monkeypatch):
    seen = {}

    def spy(y, steps):
        seen["len"] = len(y)
        return np.full(steps, y[-1])

    monkeypatch.setitem(inference.FORECASTERS, "ARIMA", spy)
    inference.evaluate_model(prices, "ARIMA", test_size=30)
    assert seen["len"] == len(prices) - 30


def test_lstm_unavailable_raises_modelunavailable(prices):
    with pytest.raises(inference.ModelUnavailable):
        inference.generate_forecast(prices, "LSTM")
