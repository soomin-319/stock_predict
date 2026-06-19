import pandas as pd
import pytest

from src.validation.signal_tuning import DEFAULT_WEIGHTS, tune_signal_weights


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


def test_tune_signal_weights_penalizes_negative_rank_ic():
    rows = []
    for dt in pd.date_range("2024-01-01", periods=12):
        for norm_return, up_probability, target_log_return in [
            (1.00, 0.00, 0.18),
            (0.90, 0.10, -0.20),
            (0.80, 0.20, -0.20),
            (0.70, 0.30, -0.20),
            (0.60, 0.40, -0.20),
            (0.40, 0.60, 0.10),
            (0.30, 0.70, 0.11),
            (0.20, 0.80, 0.12),
            (0.10, 0.90, 0.13),
            (0.00, 1.00, 0.14),
        ]:
            rows.append(
                {
                    "Date": dt,
                    "norm_return": norm_return,
                    "up_probability": up_probability,
                    "uncertainty_score": 0.0,
                    "target_log_return": target_log_return,
                }
            )

    tuned = tune_signal_weights(pd.DataFrame(rows))

    assert tuned["return_weight"] == pytest.approx(0.30)
    assert tuned["up_prob_weight"] >= 0.35
    assert tuned["validation_rank_ic"] > 0.0
    assert tuned["validation_objective_score"] > tuned["validation_top_decile_return"]


def test_tune_signal_weights_falls_back_to_default_on_large_generalization_gap():
    rows = []
    for idx, dt in enumerate(pd.date_range("2024-01-01", periods=20)):
        if idx < 14:
            rows.append(
                {
                    "Date": dt,
                    "norm_return": 0.4,
                    "up_probability": 0.0,
                    "uncertainty_score": 0.0,
                    "target_log_return": -0.10,
                }
            )
            rows.append(
                {
                    "Date": dt,
                    "norm_return": 0.0,
                    "up_probability": 1.0,
                    "uncertainty_score": 0.0,
                    "target_log_return": 0.50,
                }
            )
        else:
            rows.append(
                {
                    "Date": dt,
                    "norm_return": 0.4,
                    "up_probability": 0.0,
                    "uncertainty_score": 0.0,
                    "target_log_return": 0.01,
                }
            )
            rows.append(
                {
                    "Date": dt,
                    "norm_return": 0.0,
                    "up_probability": 1.0,
                    "uncertainty_score": 0.0,
                    "target_log_return": 0.02,
                }
            )

    tuned = tune_signal_weights(pd.DataFrame(rows))

    assert tuned["overfit_fallback_applied"] is True
    assert tuned["return_weight"] == pytest.approx(DEFAULT_WEIGHTS["return_weight"])
    assert tuned["up_prob_weight"] == pytest.approx(DEFAULT_WEIGHTS["up_prob_weight"])
    assert tuned["uncertainty_penalty"] == pytest.approx(DEFAULT_WEIGHTS["uncertainty_penalty"])
