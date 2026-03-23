from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import BacktestConfig


def _coverage_halt(grp: pd.DataFrame, cfg: BacktestConfig) -> bool:
    external_source = grp["external_coverage_ratio"] if "external_coverage_ratio" in grp.columns else pd.Series(1.0, index=grp.index)
    investor_source = grp["investor_coverage_ratio"] if "investor_coverage_ratio" in grp.columns else pd.Series(1.0, index=grp.index)
    external_ratio = pd.to_numeric(external_source, errors="coerce").fillna(1.0).mean()
    investor_ratio = pd.to_numeric(investor_source, errors="coerce").fillna(1.0).mean()
    if cfg.min_external_coverage_ratio > 0 and external_ratio < cfg.min_external_coverage_ratio:
        return True
    if cfg.min_investor_coverage_ratio > 0 and investor_ratio < cfg.min_investor_coverage_ratio:
        return True
    status = grp.get("coverage_gate_status")
    if status is not None and status.astype(str).str.contains("halt", case=False).any():
        return True
    return False


def _apply_liquidity_and_capacity_filters(grp: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    eligible = grp[(grp["up_probability"] >= cfg.min_up_probability) & (grp["signal_score"] >= cfg.min_signal_score)].copy()
    if "value_traded" in eligible.columns:
        eligible["value_traded"] = pd.to_numeric(eligible["value_traded"], errors="coerce").fillna(0.0)
        eligible = eligible[eligible["value_traded"] >= cfg.min_value_traded]
        per_position_notional = float(cfg.portfolio_value) / float(max(1, cfg.top_k))
        eligible["max_capacity_notional"] = eligible["value_traded"] * float(cfg.max_daily_participation)
        eligible = eligible[eligible["max_capacity_notional"] >= per_position_notional]
    return eligible


def _enforce_market_type_caps(ranked: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if ranked.empty or "market_type" not in ranked.columns or cfg.max_positions_per_market_type <= 0:
        return ranked

    selected = []
    counts: dict[str, int] = {}
    for _, row in ranked.iterrows():
        market_type = str(row.get("market_type", "unknown") or "unknown")
        if counts.get(market_type, 0) >= cfg.max_positions_per_market_type:
            continue
        selected.append(row)
        counts[market_type] = counts.get(market_type, 0) + 1
        if len(selected) >= cfg.top_k:
            break
    if not selected:
        return ranked.head(cfg.top_k)
    return pd.DataFrame(selected).reset_index(drop=True)


def _select_top_portfolio(grp: pd.DataFrame, prev_symbols: set[str], cfg: BacktestConfig) -> pd.DataFrame:
    eligible = _apply_liquidity_and_capacity_filters(grp, cfg)
    ranked = eligible.sort_values("signal_score", ascending=False).head(max(cfg.top_k * 3, cfg.top_k))
    ranked = _enforce_market_type_caps(ranked, cfg)
    if prev_symbols and cfg.turnover_limit < 1.0 and not ranked.empty:
        old = ranked[ranked["Symbol"].astype(str).isin(prev_symbols)]
        new = ranked[~ranked["Symbol"].astype(str).isin(prev_symbols)]
        max_new = int(round(cfg.top_k * max(0.0, cfg.turnover_limit)))
        top = pd.concat([old.head(cfg.top_k - max_new), new.head(max_new)], ignore_index=True)
        ranked = top.sort_values("signal_score", ascending=False).head(cfg.top_k)
    return ranked.head(cfg.top_k)


def _benchmark_return(grp: pd.DataFrame) -> float:
    if "ks11_ret_1d" not in grp.columns:
        return 0.0
    series = pd.to_numeric(grp["ks11_ret_1d"], errors="coerce").dropna()
    return float(series.mean()) if not series.empty else 0.0


def _cost_breakdown(top: pd.DataFrame, cfg: BacktestConfig, turnover: float) -> tuple[float, float, float]:
    dyn_penalty = 0.0
    if "uncertainty_score" in top.columns:
        dyn_penalty += float(pd.to_numeric(top["uncertainty_score"], errors="coerce").fillna(0.0).mean())
    if "vol_ratio_20" in top.columns:
        vol_pen = pd.to_numeric(top["vol_ratio_20"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
        dyn_penalty += float((vol_pen - 1.0).clip(lower=0).mean())
    turnover_cost = turnover * float(cfg.dynamic_slippage_bps) / 10000.0
    static_cost = (float(cfg.fee_bps) + float(cfg.slippage_bps)) / 10000.0
    dynamic_cost = turnover_cost + float(cfg.dynamic_slippage_bps) * dyn_penalty / 10000.0
    return dyn_penalty, static_cost, dynamic_cost


def run_long_only_topk_backtest(pred_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    df = pred_df.copy().sort_values(["Date", "signal_score"], ascending=[True, False])

    daily_returns = []
    benchmark_returns = []
    holdings_history: list[set[str]] = []
    selected_count = []
    daily_market_type_mix = []
    halted_days = 0
    liquidity_blocked_days = 0

    prev_symbols: set[str] = set()
    for dt, grp in df.groupby("Date"):
        if _coverage_halt(grp, cfg):
            halted_days += 1
            daily_returns.append((pd.to_datetime(dt), 0.0))
            benchmark_returns.append((pd.to_datetime(dt), _benchmark_return(grp)))
            holdings_history.append(set())
            selected_count.append(0)
            daily_market_type_mix.append({})
            prev_symbols = set()
            continue

        top = _select_top_portfolio(grp, prev_symbols, cfg)
        if top.empty:
            liquidity_blocked_days += 1
            daily_returns.append((pd.to_datetime(dt), 0.0))
            benchmark_returns.append((pd.to_datetime(dt), _benchmark_return(grp)))
            holdings_history.append(set())
            selected_count.append(0)
            daily_market_type_mix.append({})
            prev_symbols = set()
            continue

        current_symbols = set(top["Symbol"].astype(str).tolist())
        denom = max(1, len(prev_symbols | current_symbols))
        turnover = 0.0 if not prev_symbols else len(prev_symbols.symmetric_difference(current_symbols)) / denom

        gross = float(pd.to_numeric(top["target_log_return"], errors="coerce").fillna(0.0).mean())
        dyn_penalty, static_cost, dynamic_cost = _cost_breakdown(top, cfg, turnover)
        net = gross - static_cost - dynamic_cost

        daily_returns.append((pd.to_datetime(dt), net))
        benchmark_returns.append((pd.to_datetime(dt), _benchmark_return(grp)))
        prev_symbols = current_symbols
        holdings_history.append(prev_symbols)
        selected_count.append(int(len(top)))
        if "market_type" in top.columns:
            daily_market_type_mix.append(top["market_type"].astype(str).value_counts().to_dict())
        else:
            daily_market_type_mix.append({})

    if not daily_returns:
        return {
            "days": 0,
            "cum_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "avg_selected_count": 0.0,
            "benchmark_cum_return": 0.0,
            "excess_cum_return": 0.0,
            "cost_scenarios": {},
            "halted_days": 0,
            "liquidity_blocked_days": 0,
            "avg_market_type_count": 0.0,
            "series": [],
        }

    series = pd.Series({d: r for d, r in daily_returns}).sort_index()
    benchmark = pd.Series({d: r for d, r in benchmark_returns}).sort_index() if benchmark_returns else pd.Series(dtype=float)
    equity = (1 + series).cumprod()
    benchmark_equity = (1 + benchmark).cumprod() if not benchmark.empty else pd.Series(dtype=float)
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
            "benchmark_return": benchmark.reindex(series.index).fillna(0.0).values if not benchmark.empty else 0.0,
            "market_type_count": [len(mix) for mix in daily_market_type_mix],
        }
    )

    avg_dyn_penalty = 0.0
    if "uncertainty_score" in df.columns or "vol_ratio_20" in df.columns:
        dyn_components = []
        if "uncertainty_score" in df.columns:
            dyn_components.append(pd.to_numeric(df["uncertainty_score"], errors="coerce").fillna(0.0).mean())
        if "vol_ratio_20" in df.columns:
            dyn_components.append(
                (pd.to_numeric(df["vol_ratio_20"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0) - 1.0)
                .clip(lower=0)
                .mean()
            )
        avg_dyn_penalty = float(np.mean(dyn_components)) if dyn_components else 0.0

    avg_turnover = float(np.mean(turnovers))
    gross_daily_mean = float(series.mean()) + (cfg.fee_bps + cfg.slippage_bps) / 10000.0
    scenario_costs = {
        "conservative": (
            cfg.fee_bps
            + cfg.slippage_bps * cfg.conservative_slippage_multiplier
            + cfg.dynamic_slippage_bps * (cfg.conservative_slippage_multiplier * avg_dyn_penalty + avg_turnover)
        )
        / 10000.0,
        "neutral": (
            cfg.fee_bps
            + cfg.slippage_bps
            + cfg.dynamic_slippage_bps * (avg_dyn_penalty + avg_turnover)
        )
        / 10000.0,
        "aggressive": (
            cfg.fee_bps
            + cfg.slippage_bps * cfg.aggressive_slippage_multiplier
            + cfg.dynamic_slippage_bps * (cfg.aggressive_slippage_multiplier * avg_dyn_penalty + avg_turnover)
        )
        / 10000.0,
    }
    scenario_results = {
        name: float((1 + pd.Series([gross_daily_mean - scenario_cost] * max(1, len(series)))).cumprod().iloc[-1] - 1)
        for name, scenario_cost in scenario_costs.items()
    }
    benchmark_cum_return = float(benchmark_equity.iloc[-1] - 1) if not benchmark_equity.empty else 0.0

    return {
        "days": int(series.shape[0]),
        "cum_return": float(equity.iloc[-1] - 1),
        "avg_daily_return": float(series.mean()),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "avg_turnover": avg_turnover,
        "avg_selected_count": float(np.mean(selected_count)),
        "benchmark_cum_return": benchmark_cum_return,
        "excess_cum_return": float(equity.iloc[-1] - 1 - benchmark_cum_return),
        "cost_scenarios": scenario_results,
        "halted_days": int(halted_days),
        "liquidity_blocked_days": int(liquidity_blocked_days),
        "avg_market_type_count": float(np.mean([len(mix) for mix in daily_market_type_mix])) if daily_market_type_mix else 0.0,
        "series": series_payload.to_dict(orient="records"),
    }
