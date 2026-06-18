from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import BacktestConfig


BACKTEST_SUMMARY_KEYS = (
    "days",
    "cum_return",
    "avg_daily_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "avg_selected_count",
    "benchmark_cum_return",
    "excess_cum_return",
    "halted_days",
    "liquidity_blocked_days",
    "avg_market_type_count",
)


def coverage_gate_status(
    cfg: BacktestConfig,
    external_coverage_ratio: float,
    investor_coverage_ratio: float,
) -> str:
    cfg = getattr(cfg, "backtest", cfg)
    if cfg.min_external_coverage_ratio > 0 and external_coverage_ratio < cfg.min_external_coverage_ratio:
        return "halt"
    if cfg.min_investor_coverage_ratio > 0 and investor_coverage_ratio < cfg.min_investor_coverage_ratio:
        return "halt"
    if external_coverage_ratio < max(0.7, cfg.min_external_coverage_ratio) or investor_coverage_ratio < max(
        0.7, cfg.min_investor_coverage_ratio
    ):
        return "caution"
    return "normal"


def backtest_summary_fields(backtest: dict) -> dict[str, float]:
    out = {}
    for key in BACKTEST_SUMMARY_KEYS:
        value = backtest.get(key, 0.0)
        out[f"backtest_{key}"] = float(value) if isinstance(value, (int, float)) else 0.0
    return out


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
    eligible = grp[pd.to_numeric(grp["up_probability"], errors="coerce").fillna(0.0) >= cfg.min_up_probability].copy()
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

    selected_indices = []
    counts: dict[str, int] = {}
    market_type_pos = ranked.columns.get_loc("market_type")
    for row in ranked.itertuples(index=True, name=None):
        row_index = row[0]
        market_type = str(row[market_type_pos + 1] or "unknown")
        if counts.get(market_type, 0) >= cfg.max_positions_per_market_type:
            continue
        selected_indices.append(row_index)
        counts[market_type] = counts.get(market_type, 0) + 1
        if len(selected_indices) >= cfg.top_k:
            break
    if not selected_indices:
        return ranked.head(cfg.top_k)
    return ranked.loc[selected_indices].reset_index(drop=True)


def _select_top_portfolio(grp: pd.DataFrame, prev_symbols: set[str], cfg: BacktestConfig) -> pd.DataFrame:
    eligible = _apply_liquidity_and_capacity_filters(grp, cfg)
    ranked = eligible.sort_values(["predicted_return", "signal_score"], ascending=[False, False]).head(max(cfg.top_k * 3, cfg.top_k))
    ranked = _enforce_market_type_caps(ranked, cfg)
    if prev_symbols and cfg.turnover_limit < 1.0 and not ranked.empty:
        old = ranked[ranked["Symbol"].astype(str).isin(prev_symbols)]
        new = ranked[~ranked["Symbol"].astype(str).isin(prev_symbols)]
        max_new = int(round(cfg.top_k * max(0.0, cfg.turnover_limit)))
        top = pd.concat([old.head(cfg.top_k - max_new), new.head(max_new)], ignore_index=True)
        ranked = top.sort_values(["predicted_return", "signal_score"], ascending=[False, False]).head(cfg.top_k)
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


def _capacity_weights(top: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    if top.empty:
        return pd.Series(dtype=float)

    index = top.index
    target = pd.Series(1.0 / len(top), index=index, dtype=float)
    if "max_capacity_notional" not in top.columns or float(cfg.portfolio_value) <= 0:
        return target

    caps = pd.to_numeric(top["max_capacity_notional"], errors="coerce").fillna(0.0).astype(float)
    caps = (caps / float(cfg.portfolio_value)).clip(lower=0.0, upper=1.0)
    weights = pd.Series(0.0, index=index, dtype=float)
    remaining = 1.0
    available = set(index)
    while available and remaining > 1e-12:
        per_name = remaining / len(available)
        progressed = False
        for row_index in list(available):
            room = float(caps.loc[row_index] - weights.loc[row_index])
            add = min(per_name, max(0.0, room))
            if add <= 1e-12:
                available.remove(row_index)
                continue
            weights.loc[row_index] += add
            remaining -= add
            progressed = True
            if weights.loc[row_index] >= caps.loc[row_index] - 1e-12:
                available.remove(row_index)
        if not progressed:
            break
    return weights


def _weighted_simple_return(top: pd.DataFrame, weights: pd.Series) -> float:
    simple_returns = np.expm1(pd.to_numeric(top["target_log_return"], errors="coerce").fillna(0.0))
    return float((simple_returns * weights.reindex(top.index).fillna(0.0)).sum())


def _scenario_daily_cost(
    cfg: BacktestConfig,
    dyn_penalty: float,
    turnover: float,
    invested_weight: float,
    slippage_multiplier: float,
) -> float:
    return (
        float(cfg.fee_bps)
        + float(cfg.slippage_bps) * slippage_multiplier
        + float(cfg.dynamic_slippage_bps) * (slippage_multiplier * dyn_penalty + turnover)
    ) / 10000.0 * invested_weight


def run_long_only_topk_backtest(pred_df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    required = {"Date", "Symbol", "predicted_return", "up_probability", "target_log_return"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"backtest input missing required columns: {sorted(missing)}")
    df = pred_df.copy().sort_values(["Date", "predicted_return", "signal_score"], ascending=[True, False, False])

    daily_returns = []
    daily_gross_returns = []
    daily_cost_inputs = []
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
            daily_gross_returns.append((pd.to_datetime(dt), 0.0))
            daily_cost_inputs.append((pd.to_datetime(dt), 0.0, 0.0, 0.0))
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
            daily_gross_returns.append((pd.to_datetime(dt), 0.0))
            daily_cost_inputs.append((pd.to_datetime(dt), 0.0, 0.0, 0.0))
            benchmark_returns.append((pd.to_datetime(dt), _benchmark_return(grp)))
            holdings_history.append(set())
            selected_count.append(0)
            daily_market_type_mix.append({})
            prev_symbols = set()
            continue

        current_symbols = set(top["Symbol"].astype(str).tolist())
        denom = max(1, len(prev_symbols | current_symbols))
        turnover = 0.0 if not prev_symbols else len(prev_symbols.symmetric_difference(current_symbols)) / denom

        weights = _capacity_weights(top, cfg)
        invested_weight = float(weights.sum())
        gross = _weighted_simple_return(top, weights)
        dyn_penalty, static_cost, dynamic_cost = _cost_breakdown(top, cfg, turnover)
        net = gross - (static_cost + dynamic_cost) * invested_weight

        daily_returns.append((pd.to_datetime(dt), net))
        daily_gross_returns.append((pd.to_datetime(dt), gross))
        daily_cost_inputs.append((pd.to_datetime(dt), dyn_penalty, turnover, invested_weight))
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
    gross_series = pd.Series({d: r for d, r in daily_gross_returns}).reindex(series.index).fillna(0.0)
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

    avg_turnover = float(np.mean(turnovers))
    cost_frame = pd.DataFrame(
        daily_cost_inputs,
        columns=["Date", "dyn_penalty", "turnover", "invested_weight"],
    ).set_index("Date")
    cost_frame = cost_frame.reindex(series.index).fillna(0.0)
    scenario_multipliers = {
        "conservative": float(cfg.conservative_slippage_multiplier),
        "neutral": 1.0,
        "aggressive": float(cfg.aggressive_slippage_multiplier),
    }
    scenario_results = {}
    for name, multiplier in scenario_multipliers.items():
        costs = [
            _scenario_daily_cost(cfg, row.dyn_penalty, row.turnover, row.invested_weight, multiplier)
            for row in cost_frame.itertuples()
        ]
        scenario_series = gross_series - pd.Series(costs, index=series.index)
        scenario_results[name] = float((1 + scenario_series).cumprod().iloc[-1] - 1)
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
