# Bug Report: Stock-Market-Price-Predictor (`main` before this change)

Environment: Python 3.12, Flask 3.1, price data replaced by a fixed synthetic series so results are reproducible.
Method: manual requests against `POST /predict`, then automated as pytest cases in `tests/test_api.py`.

| ID | Severity | Summary | Status |
|----|----------|---------|--------|
| BUG-001 | Medium | Non-JSON request body returns 500 instead of 4xx | Fixed |
| BUG-002 | Medium | Unknown model name and malformed ticker are accepted (HTTP 200) | Fixed |
| BUG-003 | High | "Accuracy" column shows hard-coded values (94% / 88% / 76%), not measured | Fixed |
| BUG-004 | High | Forecasts are random: identical requests return different predictions | Fixed |
| BUG-005 | Medium | Vote ties resolve to "Buy" because of dict ordering | Fixed |
| BUG-006 | Medium | Sentiment column displays fabricated percentages with source names | Fixed |

## BUG-001: Non-JSON body returns 500
**Steps:** `curl -i -X POST http://127.0.0.1:5000/predict -d "ticker=AAPL"`
**Expected:** 400 (or 415) with a clear message.
**Actual:** `500 {"error": "An internal server error occurred."}`; server log shows `415 Unsupported Media Type`.
**Cause:** `request.get_json()` raises an HTTP error that the blanket `except Exception` converts to 500.
**Fix:** `get_json(silent=True)` plus an explicit "must be a JSON object" check. Test: `test_non_json_body_is_400_not_500`.

## BUG-002: Invalid input accepted
**Steps:** `curl -X POST .../predict -H "Content-Type: application/json" -d '{"ticker":"AAPL;DROP","model":"Prophet"}'`
**Expected:** 400 for both fields.
**Actual:** 200. An unknown model silently produced a flat forecast; the ticker string was passed unvalidated to the data layer.
**Fix:** ticker regex and model allow-list. Tests: `test_malformed_ticker_is_400`, `test_unknown_model_is_400`.

## BUG-003: Hard-coded accuracy
**Steps:** send any valid request and read `consensus[*].accuracy`.
**Actual:** always `94%`, `88%`, `76%` for LSTM / XGBoost / ARIMA, regardless of ticker or data.
**Expected:** a value computed from model output versus actual prices.
**Fix:** `evaluate_model()` runs a 30-day hold-out and the UI shows RMSE and directional accuracy. Test: `test_consensus_lists_three_models_with_measured_accuracy`.

## BUG-004: Non-deterministic simulated forecasts
**Steps:** send the same valid request three times and compare `forecast[-1].y`.
**Actual:** three different values (69.51, 72.64, 68.59 in my run). `inference.py` generated random-walk noise instead of running trained models.
**Fix:** real ARIMA, XGBoost and LSTM implementations (fixed seeds, cached). Tests: `test_forecast_is_deterministic`, `test_forecast_is_not_random_noise_around_last_price`.

## BUG-005: Tie-break favours "Buy"
**Steps:** `votes = {"Buy": 1, "Hold": 1, "Sell": 1}; max(votes, key=votes.get)` returns `"Buy"`.
**Expected:** no majority means Hold.
**Fix:** `decide_vote()`. Test: `test_decide_vote_tie_resolves_to_hold`.

## BUG-006: Fabricated sentiment data
**Actual:** the sentiment column shows strings like "82% Bullish (Reddit/Twitter)" generated purely from the forecast growth, with no data source.
**Fix:** column now reads "Not integrated" until a real sentiment source exists. Test: `test_sentiment_is_not_faked`.

> Add your own screenshots of BUG-003/BUG-006 from the original UI (checkout the commit before this change) before publishing.
