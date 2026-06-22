import inspect

import numpy as np
import pandas as pd
import pytest

from src.config.settings import BacktestConfig, SignalConfig
from src.domain import signal_policy
from src.domain.signal_policy import (
    build_pm_summary_fields,
    build_prediction_policy_frame,
    prediction_reason,
    risk_flag,
)


def _policy_input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": "BUY",
                "predicted_return": 3.1,
                "signal_score": 0.40,
                "confidence_score": 0.82,
                "coverage_gate_status": "normal",
                "uncertainty_score": 0.20,
                "up_probability": 0.70,
                "history_direction_accuracy": 0.60,
                "value_traded": 5_000_000_000,
                "min_liquidity_threshold": 0,
                "external_coverage_ratio": 1.0,
                "investor_coverage_ratio": 1.0,
                "market_headwind_score": 0,
                "turnover_rank_daily": 2,
                "foreign_net_buy": 120_000_000_000,
                "institution_net_buy": 130_000_000_000,
                "breakout_52w_flag": 1,
                "near_52w_high_flag": 0,
                "leader_confirmation_flag": 1,
                "leader_1_return": 0.02,
                "leader_2_return": 0.01,
                "leader_3_return": 0.03,
                "nq_f_ret_1d": 0.012,
                "rsi_14": 32,
            },
            {
                "Symbol": "SELL",
                "predicted_return": -2.4,
                "signal_score": 0.55,
                "confidence_score": 0.30,
                "coverage_gate_status": "halt",
                "uncertainty_score": 0.80,
                "up_probability": 0.40,
                "history_direction_accuracy": 0.40,
                "value_traded": 10_000_000,
                "min_liquidity_threshold": 0,
                "external_coverage_ratio": 0.50,
                "investor_coverage_ratio": 0.20,
                "market_headwind_score": -1,
                "turnover_rank_daily": 20,
                "foreign_net_buy": 0,
                "institution_net_buy": 0,
                "breakout_52w_flag": 0,
                "near_52w_high_flag": 0,
                "leader_confirmation_flag": 0,
                "leader_1_return": -0.01,
                "leader_2_return": 0.01,
                "leader_3_return": 0.02,
                "nq_f_ret_1d": -0.012,
                "rsi_14": 72,
            },
            {
                "Symbol": "HOLD",
                "predicted_return": np.nan,
                "signal_score": 0.10,
                "confidence_score": np.nan,
                "coverage_gate_status": "caution",
                "uncertainty_score": np.nan,
                "up_probability": np.nan,
                "history_direction_accuracy": np.nan,
                "value_traded": 4_000_000_000,
                "min_liquidity_threshold": np.nan,
                "external_coverage_ratio": np.nan,
                "investor_coverage_ratio": np.nan,
                "market_headwind_score": np.nan,
                "turnover_rank_daily": np.nan,
                "foreign_net_buy": np.nan,
                "institution_net_buy": np.nan,
                "breakout_52w_flag": np.nan,
                "near_52w_high_flag": np.nan,
                "leader_confirmation_flag": np.nan,
                "leader_1_return": np.nan,
                "leader_2_return": np.nan,
                "leader_3_return": np.nan,
                "nq_f_ret_1d": np.nan,
                "rsi_14": np.nan,
            },
        ]
    )


def test_build_prediction_policy_frame_matches_scalar_policy_helpers():
    frame = _policy_input_frame()

    out = build_prediction_policy_frame(frame)

    for idx, original_row in frame.iterrows():
        boosted_row = signal_policy.vectorized_event_signal_boost(frame).loc[idx]
        expected_pm = build_pm_summary_fields(boosted_row)
        for column, expected_value in expected_pm.items():
            assert out.loc[idx, column] == expected_value
        expected_jongbae_score = signal_policy._jongbae_score(boosted_row)
        assert out.loc[idx, "jongbae_score"] == expected_jongbae_score
        expected_jongbae_signal = "관심" if expected_jongbae_score >= 0.45 else ("경계" if expected_jongbae_score < 0 else "중립")
        assert out.loc[idx, "jongbae_signal"] == expected_jongbae_signal
        assert out.loc[idx, "prediction_reason"] == prediction_reason(boosted_row)

    assert out["recommendation"].tolist() == ["\uB9E4\uC218", "\uB9E4\uB3C4", "\uAD00\uB9DD"]


def test_nan_liquidity_threshold_uses_default_minimum():
    row = pd.Series(
        {
            "value_traded": BacktestConfig().min_value_traded - 1,
            "min_liquidity_threshold": np.nan,
        }
    )

    assert "LOW_LIQUIDITY" in risk_flag(row)


def test_build_prediction_policy_frame_has_no_rowwise_apply_calls():
    source = inspect.getsource(build_prediction_policy_frame)

    assert ".apply(" not in source


def test_vectorized_recommendation_matches_scalar_with_custom_thresholds():
    cfg = SignalConfig(
        recommendation_buy_threshold_pct=3.0,
        recommendation_sell_threshold_pct=-1.0,
    )
    frame = pd.DataFrame(
        {
            "predicted_return": [3.1, 3.0, 2.5, -0.9, -1.0, -1.1, np.nan],
        }
    )

    vectorized = signal_policy._recommendation_series(frame, signal_cfg=cfg).tolist()
    scalar = [
        signal_policy.recommendation_from_signal(None, value, signal_cfg=cfg)
        for value in frame["predicted_return"].tolist()
    ]

    assert vectorized == scalar
    assert vectorized == ["매수", "관망", "관망", "관망", "매도", "매도", "관망"]
