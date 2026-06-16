from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import SignalConfig
from src.inference.predict import build_prediction_frame
from src.models.lgbm_heads import MultiHeadPrediction


def test_build_prediction_frame_clips_negative_uncertainty_width():
    latest = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-06-16")],
            "Symbol": ["005930.KS"],
            "Close": [100.0],
            "market_regime": ["normal"],
        }
    )
    pred = MultiHeadPrediction(
        predicted_return=np.array([0.01]),
        up_probability=np.array([0.6]),
        quantile_low=np.array([0.05]),
        quantile_mid=np.array([0.01]),
        quantile_high=np.array([-0.05]),
    )

    out = build_prediction_frame(latest, pred, SignalConfig())

    assert out.loc[0, "uncertainty_width"] == 0.0
    assert out.loc[0, "uncertainty_score"] == 0.5
