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

    prev_symbols: set[str] = set()
    for dt, grp in df.groupby("Date"):
        eligible = grp[(grp["up_probability"] >= cfg.min_up_probability) & (grp["signal_score"] >= cfg.min_signal_score)]
        ranked = eligible.head(max(cfg.top_k * 3, cfg.top_k))

        # Turnover cap: keep as many previous holdings as possible, then add newcomers.
        if prev_symbols and cfg.turnover_limit < 1.0 and not ranked.empty:
            old = ranked[ranked["Symbol"].astype(str).isin(prev_symbols)]
            new = ranked[~ranked["Symbol"].astype(str).isin(prev_symbols)]
            max_new = int(round(cfg.top_k * max(0.0, cfg.turnover_limit)))
            top = pd.concat([old.head(cfg.top_k - max_new), new.head(max_new)], ignore_index=True)
            top = top.sort_values("signal_score", ascending=False).head(cfg.top_k)
        else:
            top = ranked.head(cfg.top_k)

        if top.empty:
            daily_returns.append((pd.to_datetime(dt), 0.0))
            holdings_history.append(set())
            selected_count.append(0)
            prev_symbols = set()
            continue

        gross = top["target_log_return"].mean()
        dyn_penalty = 0.0
        if "uncertainty_score" in top.columns:
            dyn_penalty += float(top["uncertainty_score"].fillna(0).mean())
        if "vol_ratio_20" in top.columns:
            vol_pen = top["vol_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            dyn_penalty += float((vol_pen - 1.0).clip(lower=0).mean())

        cost = (cfg.fee_bps + cfg.slippage_bps + cfg.dynamic_slippage_bps * dyn_penalty) / 10000.0
        net = gross - cost
        daily_returns.append((pd.to_datetime(dt), net))
        prev_symbols = set(top["Symbol"].astype(str).tolist())
        holdings_history.append(prev_symbols)
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
