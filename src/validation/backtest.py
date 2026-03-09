from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import BacktestConfig


def run_long_only_topk_backtest(pred_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    df = pred_df.copy()
    df = df.sort_values(["Date", "signal_score"], ascending=[True, False])

    daily_returns = []
    for dt, grp in df.groupby("Date"):
        top = grp.head(cfg.top_k)
        if top.empty:
            continue
        gross = top["target_log_return"].mean()
        cost = (cfg.fee_bps + cfg.slippage_bps) / 10000.0
        net = gross - cost
        daily_returns.append((dt, net))

    if not daily_returns:
        return {"days": 0, "cum_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    series = pd.Series({d: r for d, r in daily_returns}).sort_index()
    equity = (1 + series).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1

    sharpe = 0.0
    if series.std() > 0:
        sharpe = np.sqrt(252) * (series.mean() / series.std())

    return {
        "days": int(series.shape[0]),
        "cum_return": float(equity.iloc[-1] - 1),
        "avg_daily_return": float(series.mean()),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
    }
