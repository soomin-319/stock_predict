from __future__ import annotations

import pandas as pd


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


def _to_numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def recommendation_from_signal(
    signal_score: float | int | None,
    predicted_return: float | int | None,
    up_probability: float | int | None = None,
    uncertainty_score: float | int | None = None,
) -> str:
    if pd.isna(predicted_return):
        return "관망"

    ret = float(predicted_return)
    signal = pd.to_numeric(pd.Series([signal_score]), errors="coerce").iloc[0]
    up_prob = pd.to_numeric(pd.Series([up_probability]), errors="coerce").iloc[0]
    uncertainty = pd.to_numeric(pd.Series([uncertainty_score]), errors="coerce").iloc[0]

    if pd.isna(signal) or pd.isna(up_prob) or pd.isna(uncertainty):
        if ret > 1.0:
            return "매수"
        if ret <= -1.0:
            return "매도"
        return "관망"

    if signal >= 0.55 and up_prob >= 0.55 and uncertainty <= 0.60 and ret > 0:
        return "매수"
    if signal <= 0.25 or up_prob < 0.45 or (ret <= -1.0 and uncertainty >= 0.5):
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


def _policy_recommendation(row: pd.Series) -> str:
    signal = row.get("signal_score")
    predicted_return = row.get("predicted_return")
    up_probability = row.get("up_probability")
    uncertainty_score = row.get("uncertainty_score")
    nq_ret = pd.to_numeric(pd.Series([row.get("nq_f_ret_1d")]), errors="coerce").iloc[0]
    rsi = pd.to_numeric(pd.Series([row.get("rsi_14")]), errors="coerce").iloc[0]

    if not pd.isna(nq_ret) and nq_ret <= -0.01:
        return "매도"
    if not pd.isna(rsi) and rsi >= 70.0:
        return "매도"
    return recommendation_from_signal(signal, predicted_return, up_probability, uncertainty_score)


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


def vectorized_event_signal_boost(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df

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
    top_turnover_mask = turnover_rank <= 15.0
    dual_buy_mask = (foreign_net_buy > 0) & (institution_net_buy > 0)
    strong_dual_buy_mask = (foreign_net_buy >= HIGH_CONVICTION_NET_BUY) & (institution_net_buy >= HIGH_CONVICTION_NET_BUY)
    combined_mask = top_turnover_mask & strong_dual_buy_mask
    fifty_two_week_mask = (near_52w > 0) | (breakout_52w > 0)
    rsi_pullback_mask = rsi.between(30.0, 35.0, inclusive="both")
    rsi_overbought_mask = rsi >= 70.0
    nasdaq_tailwind_mask = nq_ret > 0
    strong_nasdaq_tailwind_mask = nq_ret >= 0.01
    strong_nasdaq_headwind_mask = nq_ret <= -0.01

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


def prediction_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    up_probability = float(row.get("up_probability", 0.5) or 0.5)
    foreign_net_buy = float(row.get("foreign_net_buy", 0) or 0)
    institution_net_buy = float(row.get("institution_net_buy", 0) or 0)
    history_acc = float(row.get("history_direction_accuracy", 0.5) or 0.5)
    uncertainty_score = float(row.get("uncertainty_score", 0.5) or 0.5)
    turnover_rank = float(row.get("turnover_rank_daily", 999) or 999)
    breakout_52w_flag = float(row.get("breakout_52w_flag", 0) or 0)
    near_52w_high_flag = float(row.get("near_52w_high_flag", 0) or 0)
    nq_ret = float(row.get("nq_f_ret_1d", 0) or 0)
    liquidity = float(row.get("value_traded", 0) or 0)
    pred_5d = pd.to_numeric(pd.Series([row.get("predicted_return_5d")]), errors="coerce").iloc[0]
    pred_20d = pd.to_numeric(pd.Series([row.get("predicted_return_20d")]), errors="coerce").iloc[0]
    up_prob_5d = pd.to_numeric(pd.Series([row.get("up_probability_5d")]), errors="coerce").iloc[0]
    up_prob_20d = pd.to_numeric(pd.Series([row.get("up_probability_20d")]), errors="coerce").iloc[0]

    if turnover_rank <= 15:
        reasons.append("수급: 거래대금 상위권이며 거래대금 상위 15위 종목입니다")
    if foreign_net_buy > 0 and institution_net_buy > 0:
        reasons.append(
            f"수급: 외국인 {_format_korean_amount(foreign_net_buy)}, 기관 {_format_korean_amount(institution_net_buy)} 동반 순매수입니다"
        )
    if breakout_52w_flag > 0:
        reasons.append("추세: 52주 고점을 돌파한 흐름입니다")
    elif near_52w_high_flag > 0:
        reasons.append("추세: 52주 고점 부근에서 버티는 흐름입니다")
    if nq_ret >= 0.01:
        reasons.append("해외 흐름: 나스닥 선물 강세가 우호적입니다")
    elif nq_ret <= -0.01:
        reasons.append("해외 경고: 나스닥 선물 약세가 부담입니다")
    if up_probability >= 0.7:
        reasons.append(f"확률: 상승 가능성이 {up_probability * 100:.1f}%로 높습니다")
    elif up_probability >= 0.55:
        reasons.append(f"확률: 상승 가능성이 {up_probability * 100:.1f}%로 우세합니다")
    if history_acc >= 0.6:
        reasons.append(f"신뢰도: 과거 방향 적중률이 {history_acc * 100:.1f}%였습니다")
    elif uncertainty_score >= 0.7:
        reasons.append("주의: 불확실성이 높아 비중을 줄이는 편이 좋습니다")
    if liquidity > 0 and row.get("min_liquidity_threshold") is not None and liquidity < float(row.get("min_liquidity_threshold") or 0):
        reasons.append("유동성: 거래대금 기준이 낮아 체결 리스크가 있습니다")
    if not pd.isna(pred_5d) and not pd.isna(pred_20d):
        reasons.append(f"호라이즌: 5일 {pred_5d:.2f}%, 20일 {pred_20d:.2f}% 기대수익률입니다")
    if not pd.isna(up_prob_5d) and not pd.isna(up_prob_20d):
        reasons.append(f"중기확률: 5일 {up_prob_5d * 100:.1f}%, 20일 {up_prob_20d * 100:.1f}%입니다")
    if str(row.get("coverage_gate_status", "") or "").lower() == "halt":
        reasons.append("운용게이트: 데이터 커버리지가 낮아 오늘은 거래를 중단합니다")

    if not reasons:
        reasons.append("종합: 신호·수급·추세가 중립권이라 모델 점수를 중심으로 판단했습니다")
    return " / ".join(reasons[:4])


def build_pm_summary_fields(row: pd.Series) -> dict[str, str]:
    action = _policy_recommendation(row)
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


def build_prediction_policy_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    out = vectorized_event_signal_boost(pred_df)
    pm = out.apply(build_pm_summary_fields, axis=1, result_type="expand")
    out = pd.concat([out, pm], axis=1)
    out["prediction_reason"] = out.apply(prediction_reason, axis=1)
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
