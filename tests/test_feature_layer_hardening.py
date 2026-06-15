from __future__ import annotations

import numpy as np
import pandas as pd

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
