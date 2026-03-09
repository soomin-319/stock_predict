from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_backtest_figures(backtest_series: pd.DataFrame, out_dir: str) -> dict:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    out = {}
    if backtest_series.empty:
        return out

    bt = backtest_series.copy()
    bt["Date"] = pd.to_datetime(bt["Date"])

    fig1 = p / "equity_curve.png"
    plt.figure(figsize=(10, 4))
    plt.plot(bt["Date"], bt["equity"])
    plt.title("Backtest Equity Curve")
    plt.tight_layout()
    plt.savefig(fig1)
    plt.close()

    fig2 = p / "drawdown_curve.png"
    plt.figure(figsize=(10, 4))
    plt.plot(bt["Date"], bt["drawdown"])
    plt.title("Backtest Drawdown")
    plt.tight_layout()
    plt.savefig(fig2)
    plt.close()

    out["equity_curve"] = str(fig1)
    out["drawdown_curve"] = str(fig2)
    return out


def save_signal_histogram(pred_df: pd.DataFrame, out_dir: str) -> str | None:
    if pred_df.empty or "signal_score" not in pred_df.columns:
        return None
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig = p / "signal_score_hist.png"

    plt.figure(figsize=(8, 4))
    pred_df["signal_score"].hist(bins=30)
    plt.title("Signal Score Distribution")
    plt.tight_layout()
    plt.savefig(fig)
    plt.close()
    return str(fig)
