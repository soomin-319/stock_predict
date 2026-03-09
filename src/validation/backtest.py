from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import BacktestConfig


def run_long_only_topk_backtest(pred_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    df = pred_df.copy()
    df = df.sort_values(["Date", "signal_score"], ascending=[True, False])

    daily_returns = []
    holdings_history: list[set[str]] = []
    selected_count = []

    for dt, grp in df.groupby("Date"):
        eligible = grp[(grp["up_probability"] >= cfg.min_up_probability) & (grp["signal_score"] >= cfg.min_signal_score)]
        top = eligible.head(cfg.top_k)
        if top.empty:
            daily_returns.append((pd.to_datetime(dt), 0.0))
            holdings_history.append(set())
            selected_count.append(0)
            continue

        gross = top["target_log_return"].mean()
        cost = (cfg.fee_bps + cfg.slippage_bps) / 10000.0
        net = gross - cost
        daily_returns.append((pd.to_datetime(dt), net))
        holdings_history.append(set(top["Symbol"].astype(str).tolist()))
        selected_count.append(int(len(top)))

    if not daily_returns:
        return {
            "days": 0,
            "cum_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "avg_selected_count": 0.0,
            "series": [],
        }

    series = pd.Series({d: r for d, r in daily_returns}).sort_index()
    equity = (1 + series).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1

    sharpe = 0.0
    if series.std() > 0:
        sharpe = np.sqrt(252) * (series.mean() / series.std())

    turnovers = []
    prev = None
    for current in holdings_history:
        if prev is None:
            turnovers.append(0.0)
        else:
            denom = max(1, len(prev | current))
            turnovers.append(len(prev.symmetric_difference(current)) / denom)
        prev = current

    series_payload = pd.DataFrame(
        {
            "Date": series.index,
            "daily_return": series.values,
            "equity": equity.values,
            "drawdown": dd.values,
            "selected_count": selected_count,
            "turnover": turnovers,
        }
    )

    return {
        "days": int(series.shape[0]),
        "cum_return": float(equity.iloc[-1] - 1),
        "avg_daily_return": float(series.mean()),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "avg_turnover": float(np.mean(turnovers)),
        "avg_selected_count": float(np.mean(selected_count)),
        "series": series_payload.to_dict(orient="records"),
    }
