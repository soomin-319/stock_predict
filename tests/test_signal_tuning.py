import pandas as pd
import pytest

from src.validation.signal_tuning import tune_signal_weights


def test_tune_signal_weights_uses_time_split_and_reports_in_out_performance():
    rows = []
    for idx, dt in enumerate(pd.date_range("2024-01-01", periods=12)):
        rows.append(
            {
                "Date": dt,
                "norm_return": 1.0 if idx < 8 else 0.0,
                "up_probability": 0.0 if idx < 8 else 1.0,
                "uncertainty_score": 0.0,
                "target_log_return": 0.10 if idx < 8 else 0.20,
            }
        )
        rows.append(
            {
                "Date": dt,
                "norm_return": 0.0 if idx < 8 else 1.0,
                "up_probability": 1.0 if idx < 8 else 0.0,
                "uncertainty_score": 0.0,
                "target_log_return": 0.01 if idx < 8 else -0.20,
            }
        )

    tuned = tune_signal_weights(pd.DataFrame(rows))

    assert tuned["up_prob_weight"] == pytest.approx(0.35)
    assert tuned["validation_top_decile_return"] > 0.0
    assert tuned["train_top_decile_return"] < tuned["validation_top_decile_return"]
    assert tuned["top_decile_generalization_gap"] < 0.0
