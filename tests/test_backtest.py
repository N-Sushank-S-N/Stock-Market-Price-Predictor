import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backtest import metrics, run_backtest, signals_to_positions, walk_forward_signals  # noqa: E402

IDX = pd.bdate_range("2024-01-01", periods=6)
CLOSE = pd.Series([100, 110, 99, 108.9, 108.9, 119.79], index=IDX)


def test_hold_keeps_position_and_sell_goes_flat():
    s = pd.Series(["BUY", "HOLD", "SELL", "HOLD"], index=IDX[:4])
    assert signals_to_positions(s).tolist() == [1, 1, 0, 0]


def test_no_lookahead_signal_is_traded_next_day():
    # BUY on day 0 should earn day-1 return (+10%), not day-0 return
    sig = pd.Series(["BUY"] + ["HOLD"] * 5, index=IDX)
    res = run_backtest(CLOSE, sig, cost_bps=0)
    assert abs(res["equity"]["strategy"].iloc[1] - 1.10) < 1e-9


def test_costs_reduce_returns():
    sig = pd.Series(["BUY", "SELL", "BUY", "SELL", "BUY", "SELL"], index=IDX)
    free = run_backtest(CLOSE, sig, cost_bps=0)["strategy"]["total_return"]
    paid = run_backtest(CLOSE, sig, cost_bps=50)["strategy"]["total_return"]
    assert paid < free


def test_max_drawdown_known_value():
    r = pd.Series([0.10, -0.10, 0.05])
    # equity 1.1, 0.99, 1.0395 -> drawdown 0.99/1.1 - 1 = -10%
    assert abs(metrics(r)["max_drawdown"] - (-0.10)) < 1e-9


def test_buy_and_hold_matches_price_ratio():
    sig = pd.Series("HOLD", index=IDX)
    bh = run_backtest(CLOSE, sig, cost_bps=0)["buy_and_hold"]["total_return"]
    assert abs(bh - (CLOSE.iloc[-1] / CLOSE.iloc[0] - 1)) < 1e-9


def test_walk_forward_uses_only_past_data_for_training():
    df = pd.DataFrame({"close": np.arange(100.0)}, index=pd.bdate_range("2024-01-01", periods=100))
    seen = []

    def fp(train, test):
        seen.append((train.index.max(), test.index.min()))
        return ["HOLD"] * len(test)

    walk_forward_signals(df, fp, train_window=30, test_window=10)
    assert all(tr_end < te_start for tr_end, te_start in seen)
