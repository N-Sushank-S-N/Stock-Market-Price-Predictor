import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_prices(n=400, seed=7, drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture(autouse=True)
def no_tensorflow(monkeypatch):
    """Keep tests fast/offline: LSTM is treated as unavailable unless a test opts in."""
    import inference

    def _unavailable(y, steps):
        raise inference.ModelUnavailable("TensorFlow disabled in tests")

    monkeypatch.setitem(inference.FORECASTERS, "LSTM", _unavailable)
    inference._CACHE.clear()


@pytest.fixture
def client(monkeypatch, prices):
    import app as app_module
    monkeypatch.setattr(app_module, "fetch_stock_data", lambda ticker: prices.copy())
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
