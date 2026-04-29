from __future__ import annotations

import pandas as pd

from src.config.settings import InvestmentCriteriaConfig


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


def _to_numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _criteria(cfg: InvestmentCriteriaConfig | None) -> InvestmentCriteriaConfig:
    return cfg or DEFAULT_CRITERIA


def recommendation_from_signal(
    signal_score: float | int | None,
    predicted_return: float | int | None,
    up_probability: float | int | None = None,
    uncertainty_score: float | int | None = None,
) -> str:
    if pd.isna(predicted_return):
        return "관망"

    ret = float(predicted_return)
    if ret > 1.0:
        return "매수"
    if ret < -1.0:
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
    flags = []
    coverage_status = str(row.get("coverage_gate_status", "") or "").lower()
    if coverage_status == "halt":
        flags.append("COVERAGE_HALT")
    if float(row.get("uncertainty_score", 0) or 0) >= 0.75:
        flags.append("HIGH_UNCERTAINTY")
    if float(row.get("up_probability", 0) or 0) < 0.5:
        flags.append("LOW_UP_PROB")
    if float(row.get("history_direction_accuracy", 0.5) or 0.5) < 0.45:
        flags.append("LOW_HISTORY_ACC")
    if float(row.get("value_traded", 0) or 0) < float(row.get("min_liquidity_threshold", 0) or 0):
        flags.append("LOW_LIQUIDITY")
    if float(row.get("external_coverage_ratio", 1.0) or 1.0) < 0.6:
        flags.append("EXTERNAL_FEATURE_MISSING")
    if float(row.get("investor_coverage_ratio", 1.0) or 1.0) < 0.34:
        flags.append("DATA_COVERAGE_LOW")
    if float(row.get("market_headwind_score", 0) or 0) <= -1:
        flags.append("MARKET_HEADWIND")
    pred_5d = pd.to_numeric(pd.Series([row.get("predicted_return_5d")]), errors="coerce").iloc[0]
    pred_20d = pd.to_numeric(pd.Series([row.get("predicted_return_20d")]), errors="coerce").iloc[0]
    if not pd.isna(pred_5d) and not pd.isna(pred_20d) and pred_5d < 0 < pred_20d:
        flags.append("SHORT_TERM_PULLBACK")
    if not pd.isna(pred_5d) and not pd.isna(pred_20d) and pred_5d > 0 > pred_20d:
        flags.append("LONG_TERM_DOWNSIDE")
    return "|".join(flags) if flags else "NORMAL"


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


def _policy_recommendation(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> str:
    predicted_return = row.get("predicted_return")
    return recommendation_from_signal(None, predicted_return)


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

    out["event_boost_score"] = event_boost.round(6)
    if "signal_score" in out.columns:
        out["signal_score"] = pd.to_numeric(out["signal_score"], errors="coerce").fillna(0.0) + out["event_boost_score"]
    return out


def prediction_reason(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> str:
    criteria = _criteria(cfg)
    reasons: list[str] = []
    foreign_net_buy = float(row.get("foreign_net_buy", 0) or 0)
    institution_net_buy = float(row.get("institution_net_buy", 0) or 0)
    turnover_rank = float(row.get("turnover_rank_daily", 999) or 999)
    breakout_52w_flag = float(row.get("breakout_52w_flag", 0) or 0)
    nq_ret = float(row.get("nq_f_ret_1d", 0) or 0)
    rsi_14 = pd.to_numeric(pd.Series([row.get("rsi_14")]), errors="coerce").iloc[0]
    conviction_threshold = float(criteria.high_conviction_net_buy_krw)

    if turnover_rank <= float(criteria.top_turnover_rank):
        reasons.append(f"종배수급: 거래대금 상위권, 거래대금 상위 {int(criteria.top_turnover_rank)}위 종목입니다")
    if foreign_net_buy >= conviction_threshold and institution_net_buy >= conviction_threshold:
        reasons.append(
            f"수급조건: 외국인 {_format_korean_amount(foreign_net_buy)}, 기관 {_format_korean_amount(institution_net_buy)}로 각각 1,000억 이상 순매수입니다"
        )

    leader_1 = pd.to_numeric(pd.Series([row.get("leader_1_return")]), errors="coerce").iloc[0]
    leader_2 = pd.to_numeric(pd.Series([row.get("leader_2_return")]), errors="coerce").iloc[0]
    leader_3 = pd.to_numeric(pd.Series([row.get("leader_3_return")]), errors="coerce").iloc[0]
    if not pd.isna(leader_1) and not pd.isna(leader_2) and not pd.isna(leader_3):
        if leader_1 > 0 and leader_2 > 0 and leader_3 > 0:
            reasons.append("주도주확인: 1등주 상승과 함께 2·3등주 동반 상승이 확인됩니다")

    if breakout_52w_flag > 0:
        reasons.append("추세조건: 52주 신고가 종목입니다")
    if nq_ret >= float(criteria.nasdaq_tailwind_threshold):
        reasons.append("해외조건: 나스닥 선물 +1% 이상으로 종배 우호 환경입니다")
    elif nq_ret <= float(criteria.nasdaq_headwind_threshold):
        reasons.append("해외조건: 나스닥 선물 -1% 이하로 리스크 회피(매도) 구간입니다")
    if not pd.isna(rsi_14):
        if float(criteria.rsi_buy_watch_low) <= rsi_14 <= float(criteria.rsi_buy_watch_high):
            reasons.append("중장기조건: RSI 30~35 구간으로 분할매수 관찰 구간입니다")
        elif rsi_14 >= float(criteria.rsi_overbought):
            reasons.append("중장기조건: RSI 70 이상으로 이익실현/매도 우선 구간입니다")

    return " / ".join(reasons)


def _jongbae_score(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> float:
    criteria = _criteria(cfg)
    score = 0.0
    turnover_rank = pd.to_numeric(pd.Series([row.get("turnover_rank_daily")]), errors="coerce").iloc[0]
    foreign_net_buy = pd.to_numeric(pd.Series([row.get("foreign_net_buy")]), errors="coerce").iloc[0]
    institution_net_buy = pd.to_numeric(pd.Series([row.get("institution_net_buy")]), errors="coerce").iloc[0]
    if not pd.isna(turnover_rank) and turnover_rank <= float(criteria.top_turnover_rank):
        score += 0.30
    if not pd.isna(foreign_net_buy) and not pd.isna(institution_net_buy):
        if foreign_net_buy >= float(criteria.high_conviction_net_buy_krw):
            score += 0.15
        if institution_net_buy >= float(criteria.high_conviction_net_buy_krw):
            score += 0.15
    if float(row.get("breakout_52w_flag", 0) or 0) > 0 or float(row.get("near_52w_high_flag", 0) or 0) > 0:
        score += 0.15
    nq_ret = pd.to_numeric(pd.Series([row.get("nq_f_ret_1d")]), errors="coerce").iloc[0]
    if not pd.isna(nq_ret):
        if nq_ret >= float(criteria.nasdaq_tailwind_threshold):
            score += 0.20
        elif nq_ret <= float(criteria.nasdaq_headwind_threshold):
            score -= 0.35
    if float(row.get("leader_confirmation_flag", 0) or 0) > 0:
        score += 0.15
    return round(score, 4)


def build_pm_summary_fields(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> dict[str, str]:
    action = _policy_recommendation(row, cfg=cfg)
    risk = risk_flag(row)
    position_size = _position_size_hint(row.get("confidence_score"), risk)
    coverage_status = str(row.get("coverage_gate_status", "") or "").lower()
    if coverage_status == "halt":
        portfolio_action = "거래보류"
    elif action == "매수" and position_size == "중간":
        portfolio_action = "신규매수"
    elif action == "매수":
        portfolio_action = "관심관찰"
    elif action == "매도":
        portfolio_action = "비중축소"
    else:
        portfolio_action = "관망"

    if coverage_status == "halt" or "COVERAGE_HALT" in risk:
        trading_gate = "거래중단"
    elif "MARKET_HEADWIND" in risk or "DATA_COVERAGE_LOW" in risk or "LONG_TERM_DOWNSIDE" in risk:
        trading_gate = "보수모드"
    elif "LOW_LIQUIDITY" in risk:
        trading_gate = "체결주의"
    else:
        trading_gate = "정상"

    return {
        "recommendation": action,
        "risk_flag": risk,
        "position_size_hint": position_size,
        "portfolio_action": portfolio_action,
        "trading_gate": trading_gate,
        "confidence_label": confidence_label(row.get("confidence_score")),
    }


def build_prediction_policy_frame(
    pred_df: pd.DataFrame,
    cfg: InvestmentCriteriaConfig | None = None,
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    out = vectorized_event_signal_boost(pred_df, cfg=cfg)
    pm = out.apply(lambda row: build_pm_summary_fields(row, cfg=cfg), axis=1, result_type="expand")
    out = pd.concat([out, pm], axis=1)
    out["jongbae_score"] = out.apply(lambda row: _jongbae_score(row, cfg=cfg), axis=1)
    out["jongbae_signal"] = out["jongbae_score"].map(lambda v: "관심" if v >= 0.45 else ("경계" if v < 0 else "중립"))
    out["prediction_reason"] = out.apply(lambda row: prediction_reason(row, cfg=cfg), axis=1)
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
