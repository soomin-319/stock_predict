from __future__ import annotations

import numpy as np
import pandas as pd


def test_feature_selection_module_keeps_display_only_context_out_of_model_features():
    from src.features.feature_selection import DISPLAY_ONLY_CONTEXT_COLUMNS, select_feature_columns

    df = pd.DataFrame(
        {
            "ret_1d": [0.1],
            "rsi_14": [50.0],
            "news_impact_score": [0.8],
            "disclosure_score": [0.7],
        }
    )

    selected = select_feature_columns(df)

    assert "ret_1d" in selected
    assert "rsi_14" in selected
    assert not (set(selected) & DISPLAY_ONLY_CONTEXT_COLUMNS)


def test_technical_indicators_module_exposes_core_price_helpers():
    from src.features.technical_indicators import compute_macd, compute_obv, compute_rsi, rolling_zscore

    close = pd.Series([10.0, 11.0, 10.0, 12.0, 13.0])
    volume = pd.Series([100.0, 120.0, 130.0, 140.0, 150.0])

    rsi = compute_rsi(close, period=3)
    macd, signal, hist = compute_macd(close)
    obv = compute_obv(close, volume)
    zscore = rolling_zscore(close, window=3)

    assert len(rsi) == len(close)
    assert len(macd) == len(signal) == len(hist) == len(close)
    assert obv.iloc[0] == 0.0
    assert obv.iloc[-1] == 280.0
    assert np.isfinite(zscore.fillna(0.0)).all()
