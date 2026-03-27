import pandas as pd
import pytest

from src.pipeline import (
    _apply_event_signal_boost,
    _build_result_simple,
    _policy_recommendation,
    _print_prediction_console_summary,
    _recommendation_from_signal,
)


def test_console_summary_uses_direction_accuracy_top10(capsys):
    rows = []
    for i in range(12):
        rows.append(
            {
                "Date": pd.Timestamp("2024-01-01"),
                "Symbol": f"S{i:02d}",
                "symbol_name": f"N{i:02d}",
                "predicted_return": 0.1,
                "signal_score": 0.1,
                "predicted_close": 100 + i,
                "uncertainty_score": 0.2,
                "history_direction_accuracy": i / 20.0,
            }
        )
    df = pd.DataFrame(rows)
    _print_prediction_console_summary(df)
    out = capsys.readouterr().out

    assert "=== Prediction ===" in out
    assert "S11" in out and "S10" in out
    assert "S00" not in out and "S01" not in out
    assert "예측 신뢰도" in out
    assert "포트폴리오 액션" not in out
    assert "예측 이유" not in out


def test_recommendation_uses_hold_between_minus_two_and_plus_two_percent():
    assert _recommendation_from_signal(0.3, 2.2) == "매수"
    assert _recommendation_from_signal(0.3, -2.2) == "매도"
    assert _recommendation_from_signal(0.3, 2.0) == "관망"
    assert _recommendation_from_signal(0.3, -2.0) == "매도"
    assert _recommendation_from_signal(0.3, 1.8) == "관망"
    assert _recommendation_from_signal(0.3, -1.8) == "관망"
    assert _recommendation_from_signal(float("nan"), -2.1) == "매도"


def test_recommendation_aligns_with_signal_probability_and_uncertainty_when_available():
    assert _recommendation_from_signal(0.7, 0.8, 0.7, 0.2) == "매수"
    assert _recommendation_from_signal(0.2, 2.0, 0.6, 0.2) == "매도"
    assert _recommendation_from_signal(0.5, 1.2, 0.52, 0.7) == "관망"


def test_policy_recommendation_forces_sell_on_strong_nasdaq_headwind_and_overbought_rsi():
    assert _policy_recommendation(pd.Series({"signal_score": 0.9, "predicted_return": 3.0, "up_probability": 0.8, "uncertainty_score": 0.1, "nq_f_ret_1d": -0.011, "rsi_14": 55})) == "매도"
    assert _policy_recommendation(pd.Series({"signal_score": 0.9, "predicted_return": 3.0, "up_probability": 0.8, "uncertainty_score": 0.1, "nq_f_ret_1d": 0.012, "rsi_14": 72})) == "매도"


def test_build_result_simple_includes_up_probability_and_intuitive_flow_reason():
    df = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "signal_score": 0.8,
                "predicted_close": 70000,
                "predicted_return": 2.5,
                "up_probability": 0.8,
                "confidence_score": 0.8,
                "history_direction_accuracy": 0.7,
                "foreign_net_buy": 120_000_000_000,
                "institution_net_buy": 110_000_000_000,
                "turnover_rank_daily": 7,
                "breakout_52w_flag": 1.0,
                "near_52w_high_flag": 1.0,
                "uncertainty_score": 0.2,
            }
        ]
    )

    simple = _build_result_simple(df)

    assert "상승확률(%)" in simple.columns
    assert simple.loc[0, "상승확률(%)"] == "80.0%"
    assert simple.loc[0, "예측 신뢰도"] == "75.0%"
    assert "거래대금 상위권" in simple.loc[0, "예측 이유"]


def test_build_result_simple_mentions_top_turnover_only_as_probability_tailwind():
    df = pd.DataFrame(
        [
            {
                "Symbol": "000660.KS",
                "symbol_name": "SK하이닉스",
                "signal_score": 0.5,
                "predicted_close": 200000,
                "predicted_return": 1.2,
                "up_probability": 0.66,
                "confidence_score": 0.7,
                "history_direction_accuracy": 0.6,
                "foreign_net_buy": 10_000_000_000,
                "institution_net_buy": -5_000_000_000,
                "turnover_rank_daily": 12,
                "uncertainty_score": 0.3,
            }
        ]
    )

    simple = _build_result_simple(df)

    assert "거래대금 상위 15위" in simple.loc[0, "예측 이유"]


def test_apply_event_signal_boost_preserves_probability_and_adds_event_score():
    df = pd.DataFrame(
        [
            {
                "Symbol": "A",
                "signal_score": 0.30,
                "up_probability": 0.52,
                "turnover_rank_daily": 14,
                "foreign_net_buy": 0,
                "institution_net_buy": 0,
            },
            {
                "Symbol": "B",
                "signal_score": 0.30,
                "up_probability": 0.51,
                "turnover_rank_daily": 30,
                "foreign_net_buy": 120_000_000_000,
                "institution_net_buy": 110_000_000_000,
            },
            {
                "Symbol": "C",
                "signal_score": 0.30,
                "up_probability": 0.5,
                "turnover_rank_daily": 8,
                "foreign_net_buy": 120_000_000_000,
                "institution_net_buy": 100_000_000_000,
            },
        ]
    )

    out = _apply_event_signal_boost(df)

    assert out.loc[0, "up_probability"] == 0.52
    assert out.loc[1, "up_probability"] == 0.51
    assert out.loc[2, "up_probability"] == 0.5
    assert out.loc[0, "event_boost_score"] == 0.04
    assert out.loc[1, "event_boost_score"] == 0.10
    assert out.loc[2, "event_boost_score"] == pytest.approx(0.22)
    assert out.loc[2, "signal_score"] > out.loc[1, "signal_score"] > out.loc[0, "signal_score"]


def test_apply_event_signal_boost_reflects_positive_nasdaq_futures():
    df = pd.DataFrame(
        [
            {
                "Symbol": "NQ",
                "signal_score": 0.40,
                "up_probability": 0.51,
                "turnover_rank_daily": 50,
                "foreign_net_buy": 0,
                "institution_net_buy": 0,
                "nq_f_ret_1d": 0.012,
            }
        ]
    )

    out = _apply_event_signal_boost(df)

    assert out.loc[0, "up_probability"] == 0.51
    assert out.loc[0, "event_boost_score"] == 0.09
    assert out.loc[0, "signal_score"] == pytest.approx(0.49)
