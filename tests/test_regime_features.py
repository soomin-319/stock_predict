import pandas as pd

from src.features.regime_features import annotate_market_regime


def test_market_regime_high_vol_uses_past_and_current_values_only_per_symbol():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "Symbol": ["A"] * 5,
            "close_to_ma_20": [0.02] * 5,
            "vol_20": [0.10, 0.11, 0.12, 0.13, 9.00],
        }
    )

    out = annotate_market_regime(df)

    assert out["market_regime"].iloc[3] == "uptrend_high_vol"
    assert out["market_regime"].iloc[4] == "uptrend_high_vol"


def test_market_regime_expanding_vol_threshold_is_symbol_scoped():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "Symbol": ["A", "A", "B", "B"],
            "close_to_ma_20": [0.02, 0.02, -0.02, -0.02],
            "vol_20": [0.10, 0.11, 10.0, 10.1],
        }
    )

    out = annotate_market_regime(df)

    assert out.loc[1, "market_regime"] == "uptrend_high_vol"
    assert out.loc[3, "market_regime"] == "downtrend_high_vol"
