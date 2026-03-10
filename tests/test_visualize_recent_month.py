import pandas as pd

from src.reports.visualize import _prepare_recent_month_frame


def test_prepare_recent_month_frame_sorts_and_deduplicates_dates():
    sdf = pd.DataFrame(
        {
            "Date": pd.to_datetime([
                "2024-01-31",
                "2024-01-30",
                "2024-01-31",  # duplicate date
                "2023-12-15",  # outside 31-day window from max(2024-01-31)
            ]),
            "actual_next_close": [100, 90, 110, 70],
            "predicted_next_close": [101, 91, 111, 71],
            "actual_return_pct": [1.0, 2.0, 3.0, 4.0],
            "predicted_return_pct": [1.5, 2.5, 3.5, 4.5],
        }
    )

    out = _prepare_recent_month_frame(sdf)

    assert out["Date"].is_monotonic_increasing
    assert out["Date"].nunique() == len(out)
    # 2024-01-15 is older than 31 days from max date in this sample and must be dropped.
    assert (out["Date"] >= pd.Timestamp("2024-01-31") - pd.Timedelta(days=31)).all()
    # duplicate date is averaged
    row = out[out["Date"] == pd.Timestamp("2024-01-31")].iloc[0]
    assert row["actual_next_close"] == 105
    assert row["predicted_return_pct"] == 2.5
