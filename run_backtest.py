"""Walk-forward backtest of the real models on real (or synthetic) prices.

Usage (needs internet for yfinance):
    python run_backtest.py --ticker AAPL
Smoke test without internet:
    python run_backtest.py --synthetic

Every REBALANCE days the models are retrained on the previous TRAIN days only, forecast the
next REBALANCE days, and each model emits BUY / SELL / HOLD from the projected move. The
majority vote (ties -> HOLD) is then traded the following day with transaction costs, and
compared with buy-and-hold on the out-of-sample period only.
"""
import argparse
import os

import numpy as np
import pandas as pd

import inference
from backtest import print_report, run_backtest, walk_forward_signals

TRAIN, REBALANCE = 252, 21   # 1y training window, retrain monthly


def model_signal(model, train_df, horizon, thresh):
    y = inference._close_array(train_df)
    fc = inference._cached_forecast(model, y, horizon)
    move = fc[-1] / y[-1] - 1
    return "BUY" if move > thresh else "SELL" if move < -thresh else "HOLD"


def make_fit_predict(models, thresh):
    def fit_predict(train_df, test_df):
        sigs = []
        for m in models:
            try:
                sigs.append(model_signal(m, train_df, len(test_df), thresh))
            except inference.ModelUnavailable:
                pass
        votes = {s: sigs.count(s) for s in ("BUY", "HOLD", "SELL")}
        top = max(votes.values())
        winners = [k for k, v in votes.items() if v == top]
        return [winners[0] if len(winners) == 1 else "HOLD"] * len(test_df)
    return fit_predict


def load_prices(args):
    if args.synthetic:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        from conftest import make_prices
        return make_prices(n=args.days, seed=3)
    from data_engine import fetch_stock_data
    return fetch_stock_data(args.ticker, years=args.years)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--models", default="ARIMA,XGBoost")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.002,
                    help="projected move over the window needed for BUY/SELL (0.002 = 0.2%%)")
    args = ap.parse_args()

    df = load_prices(args)
    models = [m.strip() for m in args.models.split(",")]
    sig = walk_forward_signals(df, make_fit_predict(models, args.threshold), TRAIN, REBALANCE)
    oos = df.index[TRAIN]
    res = run_backtest(df["Close"].loc[oos:], sig.loc[oos:], cost_bps=args.cost_bps)

    label = "SYNTHETIC DATA (smoke test only)" if args.synthetic else args.ticker
    print(f"\n{label}: out-of-sample {oos.date()} -> {df.index[-1].date()}, "
          f"models={models}, retrain every {REBALANCE}d, cost {args.cost_bps} bps\n")
    print_report(res)
    mix = sig.loc[oos:].value_counts().to_dict()
    print(f"\nSignal mix over out-of-sample days: {mix}  (threshold {args.threshold:.2%})")

    os.makedirs("results", exist_ok=True)
    tag = "synthetic" if args.synthetic else args.ticker
    pd.DataFrame({"strategy": res["strategy"], "buy_and_hold": res["buy_and_hold"]}).to_csv(f"results/backtest_{tag}.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = res["equity"].plot(figsize=(8, 4), title=f"{label}: walk-forward equity curve")
        ax.set_ylabel("growth of 1.0")
        plt.tight_layout()
        plt.savefig(f"results/equity_{tag}.png", dpi=130)
        print(f"\nSaved results/backtest_{tag}.csv and results/equity_{tag}.png")
    except ImportError:
        print("\n(matplotlib not installed: skipped chart)")


if __name__ == "__main__":
    main()
