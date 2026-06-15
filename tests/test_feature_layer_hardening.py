from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import FeatureConfig
from src.config.settings import InvestmentCriteriaConfig
from src.features.investment_signals import add_investment_signal_features
from src.features.price_features import build_features
from src.features.technical_indicators import compute_technical_indicator_block


def test_obv_change_is_finite_when_obv_crosses_zero():
    frame = pd.DataFrame(
        {
            "High": [11.0, 10.0, 11.0, 10.0, 11.0, 12.0, 11.0],
            "Low": [9.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0],
            "Close": [10.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0],
            "Volume": [100.0] * 7,
        }
    )

    block = compute_technical_indicator_block(
        frame,
        rsi_period=3,
        stochastic_period=3,
        cci_period=3,
    )

    assert np.isfinite(block["obv_change_5d"].dropna()).all()


def _price_frame(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Date": dates,
            "Symbol": ["AAA"] * periods,
            "Open": np.arange(100.0, 100.0 + periods),
            "High": np.arange(101.0, 101.0 + periods),
            "Low": np.arange(99.0, 99.0 + periods),
            "Close": np.arange(100.0, 100.0 + periods),
            "Volume": np.arange(1_000.0, 1_000.0 + periods),
        }
    )


def test_build_features_sorts_for_calculation_and_restores_input_row_order():
    frame = _price_frame(30)
    frame["row_id"] = np.arange(len(frame))
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    out = build_features(shuffled, FeatureConfig())

    assert out["row_id"].tolist() == shuffled["row_id"].tolist()
    row = out.loc[out["Date"].eq(frame["Date"].iloc[1])].iloc[0]
    assert row["daily_return"] == pytest.approx(frame["Close"].iloc[1] / frame["Close"].iloc[0] - 1)


def test_build_features_technical_columns_match_indicator_block():
    frame = _price_frame(30)

    out = build_features(frame, FeatureConfig())
    expected = compute_technical_indicator_block(
        frame,
        rsi_period=14,
        stochastic_period=14,
        cci_period=20,
    )

    pd.testing.assert_series_equal(out["atr_14"], expected["atr_14"], check_names=False)
    pd.testing.assert_series_equal(out["obv"], expected["obv"], check_names=False)


def test_vol_ratio_20_is_current_volume_over_twenty_day_average():
    frame = _price_frame(30)

    out = build_features(frame, FeatureConfig())

    assert out["vol_ratio_20"].iloc[-1] == pytest.approx(
        frame["Volume"].iloc[-1] / frame["Volume"].rolling(20).mean().iloc[-1]
    )


def test_build_features_leaves_near_52w_threshold_to_investment_signals():
    out = build_features(_price_frame(30), FeatureConfig())

    assert "close_to_52w_high" in out.columns
    assert "near_52w_high_flag" not in out.columns


def test_near_52w_high_flag_obeys_investment_criteria_config():
    frame = pd.DataFrame({"close_to_52w_high": [0.96]})

    strict = add_investment_signal_features(
        frame,
        InvestmentCriteriaConfig(near_52w_distance_threshold=0.03),
    )
    loose = add_investment_signal_features(
        frame,
        InvestmentCriteriaConfig(near_52w_distance_threshold=0.05),
    )

    assert strict["near_52w_high_flag"].iloc[0] == 0
    assert loose["near_52w_high_flag"].iloc[0] == 1


def test_price_limit_flags_use_historical_krx_thresholds():
    frame = pd.DataFrame(
        [
            {"Date": "2015-06-11", "Symbol": "OLD", "Open": 100, "High": 100, "Low": 100, "Close": 100, "Volume": 1_000},
            {"Date": "2015-06-12", "Symbol": "OLD", "Open": 120, "High": 120, "Low": 120, "Close": 120, "Volume": 1_000},
            {"Date": "2015-06-15", "Symbol": "NEW", "Open": 100, "High": 100, "Low": 100, "Close": 100, "Volume": 1_000},
            {"Date": "2015-06-16", "Symbol": "NEW", "Open": 120, "High": 120, "Low": 120, "Close": 120, "Volume": 1_000},
        ]
    )
    frame["Date"] = pd.to_datetime(frame["Date"])

    out = build_features(frame, FeatureConfig())

    assert out.loc[(out["Symbol"] == "OLD") & out["Date"].eq("2015-06-12"), "limit_hit_up_flag"].iloc[0] == 1
    assert out.loc[(out["Symbol"] == "NEW") & out["Date"].eq("2015-06-16"), "limit_hit_up_flag"].iloc[0] == 0


def test_price_limit_flags_use_explicit_row_override():
    frame = _price_frame(2)
    frame.loc[1, ["Open", "High", "Low", "Close"]] = 110.0
    frame["price_limit_pct"] = 0.10

    out = build_features(frame, FeatureConfig())

    assert out["limit_hit_up_flag"].iloc[1] == 1
