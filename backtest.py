"""Walk-forward backtesting for Buy/Hold/Sell signals.

Plug in your own models via `fit_predict(train_df, test_df) -> signals for test_df`.

Rules that keep the backtest honest:
  * A signal generated at the close of day t is traded at day t+1 (no lookahead).
  * BUY -> long, SELL -> flat, HOLD -> keep the current position (long-only).
  * Transaction costs are charged on every position change.
  * Report out-of-sample results only, always next to buy-and-hold.
"""
import numpy as np
import pandas as pd


def signals_to_positions(signals: pd.Series) -> pd.Series:
    pos, cur = [], 0
    for s in signals.fillna("HOLD").astype(str).str.upper():
        if s == "BUY":
            cur = 1
        elif s == "SELL":
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=signals.index, dtype=float)


def _simulate(ret: pd.Series, pos: pd.Series, cost_bps: float) -> pd.Series:
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])            # entering on day 0 costs too
    return pos * ret - turnover * cost_bps / 1e4


def _trade_win_rate(pos: pd.Series, strat_ret: pd.Series):
    entries = (pos.diff().fillna(pos.iloc[0]) == 1)
    trade_id = entries.cumsum().where(pos == 1)
    per_trade = (1 + strat_ret).groupby(trade_id).prod() - 1
    return int(len(per_trade)), (float((per_trade > 0).mean()) if len(per_trade) else float("nan"))


def metrics(ret: pd.Series, periods: int = 252) -> dict:
    equity = (1 + ret).cumprod()
    years = len(ret) / periods
    sd = ret.std(ddof=1)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan"),
        "ann_volatility": float(sd * np.sqrt(periods)),
        "sharpe": float(ret.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def run_backtest(close: pd.Series, signals: pd.Series, cost_bps: float = 10.0, periods: int = 252) -> dict:
    """close and signals share a DatetimeIndex. Returns strategy and buy-and-hold results."""
    ret = close.pct_change().fillna(0.0)
    pos = signals_to_positions(signals.reindex(close.index)).shift(1).fillna(0.0)
    strat = _simulate(ret, pos, cost_bps)
    bh = _simulate(ret, pd.Series(1.0, index=ret.index), cost_bps)
    n_trades, win_rate = _trade_win_rate(pos, strat)
    s = metrics(strat, periods)
    s.update(trades=n_trades, trade_win_rate=win_rate, time_in_market=float(pos.mean()))
    return {"strategy": s, "buy_and_hold": metrics(bh, periods),
            "equity": pd.DataFrame({"strategy": (1 + strat).cumprod(), "buy_and_hold": (1 + bh).cumprod()})}


def walk_forward_signals(df: pd.DataFrame, fit_predict, train_window: int = 252, test_window: int = 21) -> pd.Series:
    """Rolling retrain: fit on the past `train_window` rows, predict the next `test_window` rows.

    fit_predict(train_df, test_df) must return one of BUY/HOLD/SELL per test row, using only
    information available at that row. Rows before the first test window stay HOLD, so slice
    the backtest to start at df.index[train_window].
    """
    out = pd.Series("HOLD", index=df.index, dtype=object)
    start = train_window
    while start < len(df):
        end = min(start + test_window, len(df))
        out.iloc[start:end] = list(fit_predict(df.iloc[start - train_window:start], df.iloc[start:end]))
        start = end
    return out


def print_report(res: dict) -> None:
    t = pd.DataFrame({"strategy": res["strategy"], "buy_and_hold": res["buy_and_hold"]})
    print(t.to_string(float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    # DEMO ONLY: synthetic random-walk prices and a naive momentum rule, to show the output format.
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=900)
    df = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx))))}, index=idx)

    def momentum(train, test):
        full = pd.concat([train["close"], test["close"]])
        r5 = full.pct_change(5).iloc[len(train):]
        return np.where(r5 > 0, "BUY", "SELL")

    sig = walk_forward_signals(df, momentum)
    oos = df.index[252]
    print_report(run_backtest(df["close"].loc[oos:], sig.loc[oos:]))
