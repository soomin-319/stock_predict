import pandas as pd

from src.pipeline import _print_prediction_console_summary


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
