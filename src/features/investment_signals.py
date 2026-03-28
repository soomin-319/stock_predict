from __future__ import annotations

import pandas as pd

from src.config.settings import InvestmentCriteriaConfig


def _to_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _compute_turnover_rank(df: pd.DataFrame) -> pd.Series:
    if "turnover_rank_daily" in df.columns:
        return _to_numeric(df, "turnover_rank_daily", default=999.0)
    if "Date" in df.columns and "value_traded" in df.columns:
        value_traded = _to_numeric(df, "value_traded", default=0.0)
        return value_traded.groupby(df["Date"]).rank(ascending=False, method="dense")
    return pd.Series(999.0, index=df.index, dtype=float)


def _leader_confirmation(df: pd.DataFrame, cfg: InvestmentCriteriaConfig) -> pd.DataFrame:
    out = df.copy()
    leader_1 = pd.Series(0.0, index=out.index, dtype=float)
    leader_2 = pd.Series(0.0, index=out.index, dtype=float)
    leader_3 = pd.Series(0.0, index=out.index, dtype=float)
    leader_confirm = pd.Series(0, index=out.index, dtype=int)
    if "Date" not in out.columns:
        out["leader_1_return"] = leader_1
        out["leader_2_return"] = leader_2
        out["leader_3_return"] = leader_3
        out["leader_confirmation_flag"] = leader_confirm
        return out

    ret = _to_numeric(out, "daily_return", default=0.0)
    rank = _to_numeric(out, "turnover_rank_daily", default=999.0)
    leader_top_n = max(2, int(cfg.leader_top_n))
    min_co_movers = max(1, int(cfg.leader_min_co_movers))
    min_ret = float(cfg.leader_min_return)

    for _, idx in out.groupby("Date").groups.items():
        date_idx = list(idx)
        ordered = sorted(date_idx, key=lambda i: (rank.loc[i], -ret.loc[i]))
        top = ordered[:leader_top_n]
        if not top:
            continue
        top_returns = [float(ret.loc[i]) for i in top]
        leader_1_ret = top_returns[0]
        leader_2_ret = top_returns[1] if len(top_returns) > 1 else 0.0
        leader_3_ret = top_returns[2] if len(top_returns) > 2 else 0.0
        co_mover_count = sum(1 for r in top_returns if r > min_ret)
        confirmed = int(leader_1_ret > min_ret and co_mover_count >= min_co_movers)
        for i in date_idx:
            leader_1.loc[i] = leader_1_ret
            leader_2.loc[i] = leader_2_ret
            leader_3.loc[i] = leader_3_ret
            leader_confirm.loc[i] = confirmed

    out["leader_1_return"] = leader_1
    out["leader_2_return"] = leader_2
    out["leader_3_return"] = leader_3
    out["leader_confirmation_flag"] = leader_confirm
    return out


def add_investment_signal_features(df: pd.DataFrame, cfg: InvestmentCriteriaConfig) -> pd.DataFrame:
    """Add investment-support features used by short-term close betting and medium-term judgment."""
    if df.empty:
        return df.copy()

    out = df.copy()
    rank = _compute_turnover_rank(out)
    out["turnover_rank_daily"] = rank
    out["is_top_turnover_15"] = (rank <= float(cfg.top_turnover_rank)).astype(int)

    foreign_net = _to_numeric(out, "foreign_net_buy", default=0.0)
    institution_net = _to_numeric(out, "institution_net_buy", default=0.0)
    high_conviction = float(cfg.high_conviction_net_buy_krw)
    out["foreign_high_conviction_buy_flag"] = (foreign_net >= high_conviction).astype(int)
    out["institution_high_conviction_buy_flag"] = (institution_net >= high_conviction).astype(int)
    out["dual_high_conviction_buy_flag"] = (
        (out["foreign_high_conviction_buy_flag"] > 0) & (out["institution_high_conviction_buy_flag"] > 0)
    ).astype(int)

    close_to_52w = _to_numeric(out, "close_to_52w_high", default=0.0)
    out["distance_to_52w_high"] = (1.0 - close_to_52w).clip(lower=0.0)
    out["near_52w_high_flag"] = (out["distance_to_52w_high"] <= float(cfg.near_52w_distance_threshold)).astype(int)
    out["breakout_52w_flag"] = (close_to_52w >= 1.0).astype(int)

    nq_ret = _to_numeric(out, "nq_f_ret_1d", default=0.0)
    out["nasdaq_tailwind_flag"] = (nq_ret >= float(cfg.nasdaq_tailwind_threshold)).astype(int)
    out["nasdaq_headwind_flag"] = (nq_ret <= float(cfg.nasdaq_headwind_threshold)).astype(int)

    rsi = _to_numeric(out, "rsi_14", default=50.0)
    out["rsi_buy_watch_flag"] = rsi.between(float(cfg.rsi_buy_watch_low), float(cfg.rsi_buy_watch_high), inclusive="both").astype(int)
    out["rsi_overbought_sell_flag"] = (rsi >= float(cfg.rsi_overbought)).astype(int)

    out["news_same_day_signal"] = (_to_numeric(out, "news_article_count", default=0.0) > 0).astype(int)
    out["disclosure_same_day_signal"] = (_to_numeric(out, "disclosure_score", default=0.0) > 0).astype(int)

    out = _leader_confirmation(out, cfg)
    return out


__all__ = ["add_investment_signal_features"]
