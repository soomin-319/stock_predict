import numpy as np
import pandas as pd
import pytest

from src.config.settings import BacktestConfig
from src.validation.backtest import run_long_only_topk_backtest
from src.validation.metrics import probability_calibration_metrics


def test_probability_calibration_metrics_range():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
    m = probability_calibration_metrics(y_true, y_prob, n_bins=5, min_samples=1)
    assert 0 <= m["brier"] <= 1
    assert 0 <= m["ece"] <= 1


def test_calibration_insufficient_sample_returns_null_ece():
    result = probability_calibration_metrics([1], [0.8], min_samples=20)

    assert result["ece"] is None
    assert result["valid"] is False
    assert result["reason"] == "insufficient_samples"


def test_calibration_reports_non_empty_bins():
    result = probability_calibration_metrics([0, 1] * 20, [0.1, 0.9] * 20)

    assert result["valid"] is True
    assert result["bins"]


def test_backtest_turnover_limit_caps_new_entries():
    # Day1 picks A,B ; Day2 ranking would be C,D first but turnover_limit=0 keeps prior holdings.
    pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "Symbol": ["A", "B", "X", "C", "D", "A"],
            "signal_score": [0.9, 0.8, 0.1, 0.95, 0.85, 0.7],
            "predicted_return": [2.0, 1.5, -1.0, 3.0, 2.5, 1.0],
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
            "predicted_return": [2.0, 1.0],
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
            "predicted_return": [4.0, 3.0, 2.0, 1.0],
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


def test_backtest_ranks_by_predicted_return_not_signal_score_or_news_columns():
    pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"] * 3),
            "Symbol": ["LOW_RET_HIGH_SIGNAL", "HIGH_RET_LOW_SIGNAL", "MID"],
            "signal_score": [0.99, 0.10, 0.50],
            "predicted_return": [0.1, 5.0, 2.0],
            "up_probability": [0.9, 0.9, 0.9],
            "target_log_return": [-0.20, 0.30, 0.10],
            "news_impact_final_score": [100.0, -100.0, 0.0],
            "value_traded": [10_000_000_000.0, 10_000_000_000.0, 10_000_000_000.0],
        }
    )

    cfg = BacktestConfig(top_k=1)
    out = run_long_only_topk_backtest(pred, cfg)

    assert out["avg_daily_return"] > 0.0
    assert out["series"][0]["daily_return"] == pytest.approx(np.expm1(0.30) - 0.0015)


def test_backtest_converts_log_returns_and_caps_capacity_weight():
    pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Symbol": ["A", "B"],
            "signal_score": [0.9, 0.8],
            "predicted_return": [2.0, 1.0],
            "up_probability": [0.9, 0.9],
            "target_log_return": [np.log1p(0.10), np.log1p(0.30)],
            "value_traded": [5_000_000_000.0, 20_000_000_000.0],
        }
    )

    cfg = BacktestConfig(
        top_k=2,
        portfolio_value=1_000_000_000.0,
        max_daily_participation=0.10,
        fee_bps=0.0,
        slippage_bps=0.0,
        dynamic_slippage_bps=0.0,
    )
    out = run_long_only_topk_backtest(pred, cfg)

    # A can carry at most 50% of the portfolio; B carries the other 50%.
    assert out["series"][0]["daily_return"] == pytest.approx(0.5 * 0.10 + 0.5 * 0.30)


def test_cost_scenarios_replay_daily_path_instead_of_flat_mean():
    pred = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Symbol": ["A", "A"],
            "signal_score": [0.9, 0.9],
            "predicted_return": [2.0, 2.0],
            "up_probability": [0.9, 0.9],
            "target_log_return": [np.log1p(0.50), np.log1p(-0.40)],
        }
    )

    cfg = BacktestConfig(
        top_k=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        dynamic_slippage_bps=0.0,
    )
    out = run_long_only_topk_backtest(pred, cfg)

    path_cum = (1 + 0.50) * (1 - 0.40) - 1
    flat_mean_cum = (1 + ((0.50 - 0.40) / 2)) ** 2 - 1
    assert out["cost_scenarios"]["neutral"] == pytest.approx(path_cum)
    assert out["cost_scenarios"]["neutral"] != pytest.approx(flat_mean_cum)
