import pandas as pd
import pytest

from src.domain.signal_policy import build_prediction_policy_frame, vectorized_event_signal_boost


def _boostable_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": "AAA",
                "predicted_return": 1.0,
                "up_probability": 0.55,
                "uncertainty_score": 0.2,
                "confidence_score": 0.8,
                "signal_score": 0.40,
                "value_traded": 5_000_000_000.0,
                "min_liquidity_threshold": 3_000_000_000.0,
                "turnover_rank_daily": 1,
                "foreign_net_buy": 120_000_000_000.0,
                "institution_net_buy": 120_000_000_000.0,
                "nq_f_ret_1d": 0.012,
                "rsi_14": 45.0,
                "near_52w_high_flag": 1,
                "breakout_52w_flag": 0,
                "leader_confirmation_flag": 1,
            }
        ]
    )


def test_build_prediction_policy_frame_does_not_add_existing_event_boost_twice():
    boosted = vectorized_event_signal_boost(_boostable_frame())
    once_score = boosted.loc[0, "signal_score"]
    boost = boosted.loc[0, "event_boost_score"]

    finalized = build_prediction_policy_frame(boosted)

    assert boost > 0
    assert finalized.loc[0, "event_boost_score"] == pytest.approx(boost)
    assert finalized.loc[0, "signal_score"] == pytest.approx(once_score)


def test_build_prediction_policy_frame_uses_default_liquidity_threshold_when_missing():
    frame = pd.DataFrame(
        [
            {
                "Symbol": "LOW",
                "predicted_return": 0.5,
                "up_probability": 0.55,
                "uncertainty_score": 0.2,
                "confidence_score": 0.8,
                "signal_score": 0.40,
                "value_traded": 1_000_000_000.0,
            }
        ]
    )

    out = build_prediction_policy_frame(frame)

    assert "LOW_LIQUIDITY" in out.loc[0, "risk_flag"]
