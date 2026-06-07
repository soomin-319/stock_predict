from src.validation.result_validity import evaluate_backtest_validity


def test_invalid_backtest_reports_blocking_reason():
    validity = evaluate_backtest_validity(
        backtest={"days": 20, "halted_days": 20, "avg_selected_count": 0.0},
        tradable_prediction_count=0,
    )

    assert validity["backtest_valid"] is False
    assert "tradable_prediction_count_zero" in validity["blocking_reasons"]
    assert "all_days_halted" in validity["blocking_reasons"]
    assert "avg_selected_count_zero" in validity["blocking_reasons"]


def test_backtest_without_evaluation_days_is_invalid():
    validity = evaluate_backtest_validity(
        backtest={"days": 0, "halted_days": 0, "avg_selected_count": 0.0},
        tradable_prediction_count=3,
    )

    assert "no_evaluation_days" in validity["blocking_reasons"]
