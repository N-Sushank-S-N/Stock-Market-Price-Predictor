"""API tests for POST /predict (offline: price data is synthetic, LSTM disabled by conftest)."""
import numpy as np
import pytest

import app as app_module
import inference


def post(client, payload=None, **kw):
    return client.post("/predict", json=payload, **kw)


# ---- happy path -----------------------------------------------------------
def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200 and b"<html" in r.data.lower()


def test_valid_request_returns_expected_fields(client):
    r = post(client, {"ticker": "AAPL", "model": "ARIMA"})
    assert r.status_code == 200
    body = r.get_json()
    for key in ("ticker", "current_price", "change_pct", "historical", "maData", "forecast",
                "signal_text", "consensus", "final_vote"):
        assert key in body


def test_forecast_has_anchor_plus_30_points(client):
    body = post(client, {"ticker": "AAPL", "model": "XGBoost"}).get_json()
    assert len(body["forecast"]) == 31


def test_historical_capped_at_90_candles(client):
    assert len(post(client, {"ticker": "AAPL", "model": "ARIMA"}).get_json()["historical"]) == 90


def test_price_is_formatted_in_dollars(client):
    assert post(client, {"ticker": "AAPL", "model": "ARIMA"}).get_json()["current_price"].startswith("$")


def test_ticker_is_normalised_to_uppercase(client):
    assert post(client, {"ticker": "  aapl ", "model": "ARIMA"}).get_json()["ticker"] == "AAPL"


def test_consensus_lists_three_models_with_measured_accuracy(client):
    rows = post(client, {"ticker": "AAPL", "model": "ARIMA"}).get_json()["consensus"]
    assert [r["name"] for r in rows] == ["LSTM", "XGBoost", "ARIMA"]
    for r in rows:
        assert r["accuracy"] not in ("94%", "88%", "76%")      # old hard-coded values
    assert rows[1]["accuracy"].startswith("RMSE $")


def test_unavailable_model_is_reported_not_voted(client):
    lstm_row = post(client, {"ticker": "AAPL", "model": "ARIMA"}).get_json()["consensus"][0]
    assert lstm_row["signal"] == "Unavailable"


def test_sentiment_is_not_faked(client):
    rows = post(client, {"ticker": "AAPL", "model": "ARIMA"}).get_json()["consensus"]
    assert all("Reddit" not in r["sentiment"] and "%" not in r["sentiment"] for r in rows)


# ---- input validation -----------------------------------------------------
@pytest.mark.parametrize("payload", [{}, {"ticker": ""}, {"ticker": "   "}, {"model": "ARIMA"}])
def test_missing_or_blank_ticker_is_400(client, payload):
    r = post(client, payload)
    assert r.status_code == 400 and "Ticker" in r.get_json()["error"]


@pytest.mark.parametrize("ticker", ["AAPL;DROP", "<script>", "A" * 40, "AA PL"])
def test_malformed_ticker_is_400(client, ticker):
    assert post(client, {"ticker": ticker, "model": "ARIMA"}).status_code == 400


def test_unknown_model_is_400(client):
    assert post(client, {"ticker": "AAPL", "model": "Prophet"}).status_code == 400


def test_non_json_body_is_400_not_500(client):
    r = client.post("/predict", data="ticker=AAPL", content_type="text/plain")
    assert r.status_code == 400


def test_json_array_body_is_400(client):
    assert post(client, ["AAPL"]).status_code == 400


def test_get_on_predict_is_405(client):
    assert client.get("/predict").status_code == 405


def test_selecting_unavailable_model_gives_clear_400(client):
    r = post(client, {"ticker": "AAPL", "model": "LSTM"})
    assert r.status_code == 400 and "TensorFlow" in r.get_json()["error"]


# ---- error handling ---------------------------------------------------------
def test_unknown_symbol_surfaces_data_error_as_400(client, monkeypatch):
    def boom(ticker):
        raise ValueError("Could not retrieve data for ZZZZ. Please check the symbol.")
    monkeypatch.setattr(app_module, "fetch_stock_data", boom)
    r = post(client, {"ticker": "ZZZZ", "model": "ARIMA"})
    assert r.status_code == 400 and "ZZZZ" in r.get_json()["error"]


def test_unexpected_failure_is_500_without_internal_details(client, monkeypatch):
    def boom(ticker):
        raise RuntimeError("secret db password")
    monkeypatch.setattr(app_module, "fetch_stock_data", boom)
    r = post(client, {"ticker": "AAPL", "model": "ARIMA"})
    assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)


# ---- decision logic -------------------------------------------------------
@pytest.mark.parametrize("move,expected", [(1.05, "Strong Buy"), (1.01, "Hold"), (0.95, "Sell")])
def test_signal_thresholds(client, monkeypatch, prices, move, expected):
    last = float(prices["Close"].iloc[-1])
    fake = lambda df, m, steps=30: ([], [], [], [last * move] * steps)  # noqa: E731
    monkeypatch.setattr(app_module, "generate_forecast", fake)
    assert post(client, {"ticker": "AAPL", "model": "ARIMA"}).get_json()["signal_text"] == expected


def test_decide_vote_majority():
    assert app_module.decide_vote({"Buy": 2, "Hold": 0, "Sell": 1}) == "Buy"


@pytest.mark.parametrize("votes", [{"Buy": 1, "Hold": 0, "Sell": 1}, {"Buy": 1, "Hold": 1, "Sell": 1},
                                   {"Buy": 0, "Hold": 0, "Sell": 0}])
def test_decide_vote_tie_resolves_to_hold(votes):
    assert app_module.decide_vote(votes) == "Hold"


def test_clean_float_handles_nan_inf_none():
    assert app_module.clean_float(float("nan")) is None
    assert app_module.clean_float(float("inf")) is None
    assert app_module.clean_float(None) is None
    assert app_module.clean_float(3) == 3.0
