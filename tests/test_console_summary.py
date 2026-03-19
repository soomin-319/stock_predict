import pandas as pd

from src.pipeline import _print_prediction_console_summary, _recommendation_from_signal


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
    assert "예측 이유" not in out


def test_recommendation_requires_more_than_one_percent_return():
    assert _recommendation_from_signal(0.3, 1.2) == "매수"
    assert _recommendation_from_signal(-0.3, -1.2) == "매도"
    assert _recommendation_from_signal(-1.0, 1.0) == "관망"
