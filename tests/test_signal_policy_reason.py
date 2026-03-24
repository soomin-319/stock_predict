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
            "close_to_ma_20": 0.04,
            "rsi_14": 34.0,
            "macd_hist": 0.2,
            "obv_change_5d": 0.08,
        }
    )

    reason = prediction_reason(row)

    assert "추세강도:" in reason or "모멘텀:" in reason or "수급강도:" in reason
