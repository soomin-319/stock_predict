from __future__ import annotations

import pandas as pd

from src.domain.signal_policy import prediction_reason


def test_prediction_reason_excludes_probability_and_horizon_phrases() -> None:
    row = pd.Series(
        {
            "up_probability": 0.809,
            "predicted_return_5d": 6.16,
            "predicted_return_20d": 13.22,
            "up_probability_5d": 0.798,
            "up_probability_20d": 0.734,
            "turnover_rank_daily": 10,
            "history_direction_accuracy": 0.64,
        }
    )

    reason = prediction_reason(row)

    assert "확률:" not in reason
    assert "호라이즌:" not in reason
    assert "중기확률:" not in reason


def test_prediction_reason_can_include_feature_based_explanations() -> None:
    row = pd.Series(
        {
            "turnover_rank_daily": 8,
            "foreign_net_buy": 130_000_000_000,
            "institution_net_buy": 120_000_000_000,
            "breakout_52w_flag": 1.0,
            "nq_f_ret_1d": 0.013,
            "rsi_14": 33.0,
        }
    )

    reason = prediction_reason(row)

    assert "종배수급:" in reason
    assert "수급조건:" in reason
    assert "추세조건:" in reason
    assert "해외조건:" in reason
    assert "중장기조건:" in reason
