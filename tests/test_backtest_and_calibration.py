import numpy as np
import pandas as pd

from src.config.settings import BacktestConfig
from src.validation.backtest import run_long_only_topk_backtest
from src.validation.metrics import probability_calibration_metrics


def test_probability_calibration_metrics_range():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
    m = probability_calibration_metrics(y_true, y_prob, n_bins=5)
    assert 0 <= m["brier"] <= 1
    assert 0 <= m["ece"] <= 1


def test_backtest_turnover_limit_caps_new_entries():
    # Day1 picks A,B ; Day2 ranking would be C,D first but turnover_limit=0 keeps prior holdings.
    pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "Symbol": ["A", "B", "X", "C", "D", "A"],
            "signal_score": [0.9, 0.8, 0.1, 0.95, 0.85, 0.7],
            "up_probability": [0.8, 0.7, 0.4, 0.9, 0.8, 0.7],
            "target_log_return": [0.01, 0.01, 0.0, 0.01, 0.01, 0.01],
            "uncertainty_score": [0.2, 0.2, 0.5, 0.2, 0.2, 0.2],
        }
    )

    cfg = BacktestConfig(top_k=2, turnover_limit=0.0)
    out = run_long_only_topk_backtest(pred, cfg)
    series = pd.DataFrame(out["series"])

    assert out["days"] == 2
    # second day turnover should stay low because we preserve prior holdings as much as possible.
    assert float(series.iloc[1]["turnover"]) <= 0.5
