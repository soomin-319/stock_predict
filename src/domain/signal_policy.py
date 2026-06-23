from __future__ import annotations

import pandas as pd

from src.config.settings import BacktestConfig, InvestmentCriteriaConfig, SignalConfig


HIGH_CONVICTION_NET_BUY = 100_000_000_000
TOP3_TURNOVER_EVENT_BOOST = 0.05
DUAL_BUY_EVENT_BOOST = 0.04
LEADER_CONFIRMATION_EVENT_BOOST = 0.05
FIFTY_TWO_WEEK_HIGH_EVENT_BOOST = 0.03
RSI_PULLBACK_EVENT_BOOST = 0.02
NASDAQ_STRONG_TAILWIND_EVENT_BOOST = 0.06
NASDAQ_STRONG_HEADWIND_EVENT_PENALTY = 0.12
RSI_OVERBOUGHT_EVENT_PENALTY = 0.08
TOP_TURNOVER_EVENT_BOOST = 0.04
STRONG_DUAL_BUY_EVENT_BOOST = 0.06
HIGH_CONVICTION_COMBINED_EVENT_BOOST = 0.08
NASDAQ_FUTURES_TAILWIND_EVENT_BOOST = 0.03
DEFAULT_CRITERIA = InvestmentCriteriaConfig()
DEFAULT_SIGNAL_CONFIG = SignalConfig()
DEFAULT_MIN_LIQUIDITY_THRESHOLD = BacktestConfig().min_value_traded


def _to_numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _to_numeric_series_preserve_na(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _criteria(cfg: InvestmentCriteriaConfig | None) -> InvestmentCriteriaConfig:
    return cfg or DEFAULT_CRITERIA


def _signal_cfg(cfg: SignalConfig | None) -> SignalConfig:
    return cfg or DEFAULT_SIGNAL_CONFIG


def _row_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([row.to_dict()], index=[row.name])


def _first_scalar(series: pd.Series):
    return series.iloc[0]


def _first_record(frame: pd.DataFrame) -> dict[str, str]:
    return frame.iloc[0].to_dict()


def recommendation_from_signal(
    signal_score: float | int | None,
    predicted_return: float | int | None,
    up_probability: float | int | None = None,
    uncertainty_score: float | int | None = None,
    *,
    signal_cfg: SignalConfig | None = None,
) -> str:
    """Return buy/sell/hold policy from next-day expected return only.

    Other inputs are accepted for backward compatibility, but they must not
    change the user-facing recommendation.
    """
    if pd.isna(predicted_return):
        return "관망"

    cfg = _signal_cfg(signal_cfg)
    ret = float(predicted_return)
    if ret > float(cfg.recommendation_buy_threshold_pct):
        return "매수"
    if ret <= float(cfg.recommendation_sell_threshold_pct):
        return "매도"
    return "관망"


def confidence_label(confidence_score: float | int | None) -> str:
    if pd.isna(confidence_score):
        return "신뢰도 보통"
    c = float(confidence_score)
    if c >= 0.80:
        return "신뢰도 매우 높음"
    if c >= 0.67:
        return "신뢰도 높음"
    if c >= 0.34:
        return "신뢰도 보통"
    return "신뢰도 낮음"


def risk_flag(row: pd.Series) -> str:
    return str(_first_scalar(_risk_flag_series(_row_frame(row))))


def _position_size_hint(confidence_score: float | int | None, risk_flag_value: str) -> str:
    c = float(confidence_score) if not pd.isna(confidence_score) else 0.5
    if "DATA_COVERAGE_LOW" in risk_flag_value or "LOW_LIQUIDITY" in risk_flag_value:
        return "관망"
    if "HIGH_UNCERTAINTY" in risk_flag_value or "MARKET_HEADWIND" in risk_flag_value:
        return "소액"
    if c >= 0.80:
        return "중간"
    if c >= 0.6:
        return "소액"
    return "관망"


def _policy_recommendation(
    row: pd.Series,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> str:
    predicted_return = row.get("predicted_return")
    return recommendation_from_signal(None, predicted_return, signal_cfg=signal_cfg)


def _format_percentage_text(value, digits: int = 1, unit_interval: bool = False) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    percent_value = float(numeric) * 100.0 if unit_interval else float(numeric)
    return f"{percent_value:.{digits}f}%"


def _format_korean_amount(value: float | int | None) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric) / 100_000_000:,.0f}억"


def vectorized_event_signal_boost(
    pred_df: pd.DataFrame,
    cfg: InvestmentCriteriaConfig | None = None,
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df

    criteria = _criteria(cfg)
    out = pred_df.copy()
    turnover_rank = _to_numeric_series(out, "turnover_rank_daily", default=999.0)
    foreign_net_buy = _to_numeric_series(out, "foreign_net_buy")
    institution_net_buy = _to_numeric_series(out, "institution_net_buy")
    nq_ret = _to_numeric_series(out, "nq_f_ret_1d")
    rsi = _to_numeric_series(out, "rsi_14", default=50.0)
    near_52w = _to_numeric_series(out, "near_52w_high_flag")
    breakout_52w = _to_numeric_series(out, "breakout_52w_flag")
    leader_confirmed = _to_numeric_series(out, "leader_confirmation_flag")

    top3_turnover_mask = turnover_rank <= 3.0
    top_turnover_mask = turnover_rank <= float(criteria.top_turnover_rank)
    dual_buy_mask = (foreign_net_buy > 0) & (institution_net_buy > 0)
    strong_dual_buy_mask = (
        (foreign_net_buy >= float(criteria.high_conviction_net_buy_krw))
        & (institution_net_buy >= float(criteria.high_conviction_net_buy_krw))
    )
    combined_mask = top_turnover_mask & strong_dual_buy_mask
    fifty_two_week_mask = (near_52w > 0) | (breakout_52w > 0)
    rsi_pullback_mask = rsi.between(float(criteria.rsi_buy_watch_low), float(criteria.rsi_buy_watch_high), inclusive="both")
    rsi_overbought_mask = rsi >= float(criteria.rsi_overbought)
    nasdaq_tailwind_mask = nq_ret > 0
    strong_nasdaq_tailwind_mask = nq_ret >= float(criteria.nasdaq_tailwind_threshold)
    strong_nasdaq_headwind_mask = nq_ret <= float(criteria.nasdaq_headwind_threshold)

    event_boost = pd.Series(0.0, index=out.index, dtype=float)
    event_boost = event_boost + top3_turnover_mask.astype(float) * TOP3_TURNOVER_EVENT_BOOST
    event_boost = event_boost + top_turnover_mask.astype(float) * TOP_TURNOVER_EVENT_BOOST
    event_boost = event_boost + dual_buy_mask.astype(float) * DUAL_BUY_EVENT_BOOST
    event_boost = event_boost + strong_dual_buy_mask.astype(float) * STRONG_DUAL_BUY_EVENT_BOOST
    event_boost = event_boost + combined_mask.astype(float) * HIGH_CONVICTION_COMBINED_EVENT_BOOST
    event_boost = event_boost + (leader_confirmed > 0).astype(float) * LEADER_CONFIRMATION_EVENT_BOOST
    event_boost = event_boost + fifty_two_week_mask.astype(float) * FIFTY_TWO_WEEK_HIGH_EVENT_BOOST
    event_boost = event_boost + rsi_pullback_mask.astype(float) * RSI_PULLBACK_EVENT_BOOST
    event_boost = event_boost - rsi_overbought_mask.astype(float) * RSI_OVERBOUGHT_EVENT_PENALTY
    event_boost = event_boost + nasdaq_tailwind_mask.astype(float) * NASDAQ_FUTURES_TAILWIND_EVENT_BOOST
    event_boost = event_boost + strong_nasdaq_tailwind_mask.astype(float) * NASDAQ_STRONG_TAILWIND_EVENT_BOOST
    event_boost = event_boost - strong_nasdaq_headwind_mask.astype(float) * NASDAQ_STRONG_HEADWIND_EVENT_PENALTY

    existing_boost = _to_numeric_series(out, "event_boost_score") if "event_boost_score" in out.columns else None
    out["event_boost_score"] = event_boost.round(6)
    if "signal_score" in out.columns:
        base_score = pd.to_numeric(out["signal_score"], errors="coerce").fillna(0.0)
        if existing_boost is not None:
            base_score = base_score - existing_boost
        out["signal_score"] = base_score + out["event_boost_score"]
    return out


def _append_flag(flags: pd.Series, mask: pd.Series, label: str) -> None:
    active = mask.fillna(False)
    if not active.any():
        return
    current = flags.loc[active]
    prefix = current.where(current.eq(""), current + "|")
    flags.loc[active] = prefix + label


def _recommendation_series(df: pd.DataFrame, signal_cfg: SignalConfig | None = None) -> pd.Series:
    cfg = _signal_cfg(signal_cfg)
    predicted_return = _to_numeric_series_preserve_na(df, "predicted_return", default=float("nan"))
    recommendation = pd.Series("관망", index=df.index, dtype=object)
    recommendation.loc[predicted_return > float(cfg.recommendation_buy_threshold_pct)] = "매수"
    recommendation.loc[predicted_return <= float(cfg.recommendation_sell_threshold_pct)] = "매도"
    return recommendation


def _confidence_label_series(df: pd.DataFrame) -> pd.Series:
    confidence = _to_numeric_series_preserve_na(df, "confidence_score", default=float("nan"))
    label = pd.Series("신뢰도 보통", index=df.index, dtype=object)
    label.loc[confidence < 0.34] = "신뢰도 낮음"
    label.loc[confidence >= 0.34] = "신뢰도 보통"
    label.loc[confidence >= 0.67] = "신뢰도 높음"
    label.loc[confidence >= 0.80] = "신뢰도 매우 높음"
    return label


def _risk_flag_series(df: pd.DataFrame) -> pd.Series:
    flags = pd.Series("", index=df.index, dtype=object)
    if "coverage_gate_status" in df.columns:
        coverage_status = df["coverage_gate_status"].fillna("").astype(str).str.lower()
    else:
        coverage_status = pd.Series("", index=df.index, dtype=object)

    uncertainty = _to_numeric_series_preserve_na(df, "uncertainty_score", default=0.0)
    up_probability = _to_numeric_series_preserve_na(df, "up_probability", default=0.0)
    history_accuracy = _to_numeric_series_preserve_na(df, "history_direction_accuracy", default=0.5)
    value_traded = _to_numeric_series_preserve_na(df, "value_traded", default=0.0)
    min_liquidity = _to_numeric_series(df, "min_liquidity_threshold", default=0.0)
    external_coverage = _to_numeric_series_preserve_na(df, "external_coverage_ratio", default=1.0)
    investor_coverage = _to_numeric_series_preserve_na(df, "investor_coverage_ratio", default=1.0)
    market_headwind = _to_numeric_series_preserve_na(df, "market_headwind_score", default=0.0)
    effective_min_liquidity = min_liquidity.mask(min_liquidity <= 0, DEFAULT_MIN_LIQUIDITY_THRESHOLD)

    _append_flag(flags, coverage_status == "halt", "COVERAGE_HALT")
    _append_flag(flags, uncertainty >= 0.75, "HIGH_UNCERTAINTY")
    _append_flag(flags, up_probability < 0.5, "LOW_UP_PROB")
    _append_flag(flags, history_accuracy < 0.45, "LOW_HISTORY_ACC")
    _append_flag(flags, value_traded < effective_min_liquidity, "LOW_LIQUIDITY")
    _append_flag(flags, external_coverage < 0.6, "EXTERNAL_FEATURE_MISSING")
    _append_flag(flags, investor_coverage < 0.34, "DATA_COVERAGE_LOW")
    _append_flag(flags, market_headwind <= -1, "MARKET_HEADWIND")
    return flags.mask(flags.eq(""), "NORMAL")


def _position_size_hint_series(df: pd.DataFrame, risk_flags: pd.Series) -> pd.Series:
    confidence = _to_numeric_series_preserve_na(df, "confidence_score", default=float("nan")).fillna(0.5)
    position = pd.Series("관망", index=df.index, dtype=object)
    position.loc[confidence >= 0.6] = "소액"
    position.loc[confidence >= 0.80] = "중간"
    position.loc[
        risk_flags.str.contains("HIGH_UNCERTAINTY", regex=False)
        | risk_flags.str.contains("MARKET_HEADWIND", regex=False)
    ] = "소액"
    position.loc[
        risk_flags.str.contains("DATA_COVERAGE_LOW", regex=False)
        | risk_flags.str.contains("LOW_LIQUIDITY", regex=False)
    ] = "관망"
    return position


def _pm_summary_frame(
    df: pd.DataFrame,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> pd.DataFrame:
    action = _recommendation_series(df, signal_cfg=signal_cfg)
    risk = _risk_flag_series(df)
    position_size = _position_size_hint_series(df, risk)
    confidence = _confidence_label_series(df)
    if "coverage_gate_status" in df.columns:
        coverage_status = df["coverage_gate_status"].fillna("").astype(str).str.lower()
    else:
        coverage_status = pd.Series("", index=df.index, dtype=object)

    portfolio_action = pd.Series("관망", index=df.index, dtype=object)
    portfolio_action.loc[action == "매도"] = "비중축소"
    portfolio_action.loc[action == "매수"] = "관심관찰"
    portfolio_action.loc[(action == "매수") & (position_size == "중간")] = "신규매수"
    portfolio_action.loc[coverage_status == "halt"] = "거래보류"

    trading_gate = pd.Series("정상", index=df.index, dtype=object)
    trading_gate.loc[risk.str.contains("LOW_LIQUIDITY", regex=False)] = "체결주의"
    trading_gate.loc[
        risk.str.contains("MARKET_HEADWIND", regex=False)
        | risk.str.contains("DATA_COVERAGE_LOW", regex=False)
    ] = "보수모드"
    trading_gate.loc[(coverage_status == "halt") | risk.str.contains("COVERAGE_HALT", regex=False)] = "거래중단"

    return pd.DataFrame(
        {
            "recommendation": action,
            "risk_flag": risk,
            "position_size_hint": position_size,
            "portfolio_action": portfolio_action,
            "trading_gate": trading_gate,
            "confidence_label": confidence,
        },
        index=df.index,
    )


def prediction_reason(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> str:
    return str(_first_scalar(_prediction_reason_series(_row_frame(row), cfg=cfg)))


def _format_korean_amount_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda value: "-" if pd.isna(value) else f"{float(value) / 100_000_000:,.0f}억")


def _append_reason(reasons: pd.Series, mask: pd.Series, text: str | pd.Series) -> None:
    active = mask.fillna(False)
    if not active.any():
        return
    current = reasons.loc[active]
    prefix = current.where(current.eq(""), current + " / ")
    if isinstance(text, pd.Series):
        addition = text.loc[active].astype(str)
    else:
        addition = pd.Series(text, index=current.index, dtype=object)
    reasons.loc[active] = prefix + addition


def _prediction_reason_series(df: pd.DataFrame, cfg: InvestmentCriteriaConfig | None = None) -> pd.Series:
    criteria = _criteria(cfg)
    reasons = pd.Series("", index=df.index, dtype=object)
    turnover_rank = _to_numeric_series_preserve_na(df, "turnover_rank_daily", default=999.0)
    foreign_net_buy = _to_numeric_series_preserve_na(df, "foreign_net_buy", default=0.0)
    institution_net_buy = _to_numeric_series_preserve_na(df, "institution_net_buy", default=0.0)
    breakout_52w = _to_numeric_series_preserve_na(df, "breakout_52w_flag", default=0.0)
    nq_ret = _to_numeric_series_preserve_na(df, "nq_f_ret_1d", default=0.0)
    rsi_14 = _to_numeric_series_preserve_na(df, "rsi_14", default=float("nan"))
    leader_1 = _to_numeric_series_preserve_na(df, "leader_1_return", default=float("nan"))
    leader_2 = _to_numeric_series_preserve_na(df, "leader_2_return", default=float("nan"))
    leader_3 = _to_numeric_series_preserve_na(df, "leader_3_return", default=float("nan"))
    conviction_threshold = float(criteria.high_conviction_net_buy_krw)

    _append_reason(
        reasons,
        turnover_rank <= float(criteria.top_turnover_rank),
        f"종배수급: 거래대금 상위권, 거래대금 상위 {int(criteria.top_turnover_rank)}위 종목입니다",
    )

    strong_dual_buy = (foreign_net_buy >= conviction_threshold) & (institution_net_buy >= conviction_threshold)
    strong_dual_buy_text = (
        "수급조건: 외국인 "
        + _format_korean_amount_series(foreign_net_buy)
        + ", 기관 "
        + _format_korean_amount_series(institution_net_buy)
        + "로 각각 1,000억 이상 순매수입니다"
    )
    _append_reason(reasons, strong_dual_buy, strong_dual_buy_text)

    leader_confirmed = leader_1.notna() & leader_2.notna() & leader_3.notna() & (leader_1 > 0) & (leader_2 > 0) & (leader_3 > 0)
    _append_reason(reasons, leader_confirmed, "주도주확인: 1등주 상승과 함께 2·3등주 동반 상승이 확인됩니다")
    _append_reason(reasons, breakout_52w > 0, "추세조건: 52주 신고가 종목입니다")
    _append_reason(reasons, nq_ret >= float(criteria.nasdaq_tailwind_threshold), "해외조건: 나스닥 선물 +1% 이상으로 종배 우호 환경입니다")
    _append_reason(reasons, nq_ret <= float(criteria.nasdaq_headwind_threshold), "해외조건: 나스닥 선물 -1% 이하로 리스크 회피(매도) 구간입니다")
    _append_reason(
        reasons,
        rsi_14.between(float(criteria.rsi_buy_watch_low), float(criteria.rsi_buy_watch_high), inclusive="both"),
        "중장기조건: RSI 30~35 구간으로 분할매수 관찰 구간입니다",
    )
    _append_reason(reasons, rsi_14 >= float(criteria.rsi_overbought), "중장기조건: RSI 70 이상으로 이익실현/매도 우선 구간입니다")
    return reasons


def _jongbae_score(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> float:
    return float(_first_scalar(_jongbae_score_series(_row_frame(row), cfg=cfg)))


def _jongbae_score_series(df: pd.DataFrame, cfg: InvestmentCriteriaConfig | None = None) -> pd.Series:
    criteria = _criteria(cfg)
    score = pd.Series(0.0, index=df.index, dtype=float)
    turnover_rank = _to_numeric_series_preserve_na(df, "turnover_rank_daily", default=float("nan"))
    foreign_net_buy = _to_numeric_series_preserve_na(df, "foreign_net_buy", default=float("nan"))
    institution_net_buy = _to_numeric_series_preserve_na(df, "institution_net_buy", default=float("nan"))
    breakout_52w = _to_numeric_series_preserve_na(df, "breakout_52w_flag", default=0.0)
    near_52w = _to_numeric_series_preserve_na(df, "near_52w_high_flag", default=0.0)
    nq_ret = _to_numeric_series_preserve_na(df, "nq_f_ret_1d", default=float("nan"))
    leader_confirmed = _to_numeric_series_preserve_na(df, "leader_confirmation_flag", default=0.0)
    conviction_threshold = float(criteria.high_conviction_net_buy_krw)
    investor_values_present = foreign_net_buy.notna() & institution_net_buy.notna()

    score = score + (turnover_rank <= float(criteria.top_turnover_rank)).astype(float) * 0.30
    score = score + (investor_values_present & (foreign_net_buy >= conviction_threshold)).astype(float) * 0.15
    score = score + (investor_values_present & (institution_net_buy >= conviction_threshold)).astype(float) * 0.15
    score = score + ((breakout_52w > 0) | (near_52w > 0)).astype(float) * 0.15
    score = score + (nq_ret >= float(criteria.nasdaq_tailwind_threshold)).astype(float) * 0.20
    score = score - (nq_ret <= float(criteria.nasdaq_headwind_threshold)).astype(float) * 0.35
    score = score + (leader_confirmed > 0).astype(float) * 0.15
    return score.round(4)


def build_pm_summary_fields(
    row: pd.Series,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> dict[str, str]:
    return _first_record(_pm_summary_frame(_row_frame(row), cfg=cfg, signal_cfg=signal_cfg))


def build_prediction_policy_frame(
    pred_df: pd.DataFrame,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    out = vectorized_event_signal_boost(pred_df, cfg=cfg)
    pm = _pm_summary_frame(out, cfg=cfg, signal_cfg=signal_cfg)
    out = pd.concat([out, pm], axis=1)
    out["jongbae_score"] = _jongbae_score_series(out, cfg=cfg)
    out["jongbae_signal"] = out["jongbae_score"].map(lambda v: "관심" if v >= 0.45 else ("경계" if v < 0 else "중립"))
    out["prediction_reason"] = _prediction_reason_series(out, cfg=cfg)
    out["recommendation"] = out["recommendation"].fillna("관망")
    out["confidence_label"] = out["confidence_label"].fillna("신뢰도 보통")
    out["signal_label"] = out.get("signal_label")
    out["risk_flag"] = out["risk_flag"].fillna("NORMAL")
    return out


__all__ = [
    "build_pm_summary_fields",
    "build_prediction_policy_frame",
    "confidence_label",
    "prediction_reason",
    "recommendation_from_signal",
    "risk_flag",
    "vectorized_event_signal_boost",
]
