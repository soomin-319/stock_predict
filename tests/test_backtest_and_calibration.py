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


def test_backtest_halts_when_coverage_gate_is_below_threshold():
    pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Symbol": ["A", "B"],
            "signal_score": [0.9, 0.8],
            "up_probability": [0.8, 0.7],
            "target_log_return": [0.01, 0.01],
            "external_coverage_ratio": [0.4, 0.4],
            "investor_coverage_ratio": [0.3, 0.3],
            "coverage_gate_status": ["halt", "halt"],
            "value_traded": [10_000_000_000.0, 10_000_000_000.0],
        }
    )

    cfg = BacktestConfig(top_k=2, min_external_coverage_ratio=0.5, min_investor_coverage_ratio=0.5)
    out = run_long_only_topk_backtest(pred, cfg)

    assert out["halted_days"] == 1
    assert out["avg_selected_count"] == 0.0


def test_backtest_respects_market_type_cap_and_liquidity_capacity():
    pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"] * 4),
            "Symbol": ["A", "B", "C", "D"],
            "signal_score": [0.95, 0.9, 0.85, 0.8],
            "up_probability": [0.9, 0.9, 0.9, 0.9],
            "target_log_return": [0.01, 0.01, 0.01, 0.01],
            "market_type": ["KOSPI", "KOSPI", "KOSDAQ", "KOSDAQ"],
            "value_traded": [20_000_000_000.0, 20_000_000_000.0, 20_000_000_000.0, 1_000_000_000.0],
        }
    )

    cfg = BacktestConfig(
        top_k=3,
        portfolio_value=1_500_000_000.0,
        max_daily_participation=0.05,
        max_positions_per_market_type=1,
    )
    out = run_long_only_topk_backtest(pred, cfg)
    series = pd.DataFrame(out["series"])

    assert out["avg_selected_count"] <= 2.0
    assert float(series.iloc[0]["market_type_count"]) == 2.0
