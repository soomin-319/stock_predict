from __future__ import annotations

import pandas as pd

from src.config.settings import FeatureConfig
from src.features.price_features import DISPLAY_ONLY_CONTEXT_COLUMNS, build_features, select_feature_columns


def _sample_ohlcv(news_value: float, disclosure_value: float) -> pd.DataFrame:
    rows = []
    for day in range(30):
        close = 100.0 + day
        rows.append(
            {
                "Date": f"2024-01-{day + 1:02d}",
                "Symbol": "005930.KS",
                "Open": close - 1.0,
                "High": close + 1.0,
                "Low": close - 2.0,
                "Close": close,
                "Volume": 1_000_000 + day,
                "news_sentiment": news_value,
                "news_relevance_score": news_value,
                "news_impact_score": news_value * 100.0,
                "news_article_count": int(news_value * 10),
                "disclosure_score": disclosure_value,
            }
        )
    return pd.DataFrame(rows)


def test_select_feature_columns_excludes_display_only_news_and_disclosure_context():
    df = pd.DataFrame({name: [1.0] for name in DISPLAY_ONLY_CONTEXT_COLUMNS})
    df["daily_return"] = [0.01]
    df["ret_1"] = [0.01]

    selected = select_feature_columns(df)

    assert "daily_return" in selected
    assert "ret_1" in selected
    assert sorted(set(selected) & DISPLAY_ONLY_CONTEXT_COLUMNS) == []


def test_model_feature_values_do_not_change_when_news_and_disclosure_context_changes():
    low_context = build_features(_sample_ohlcv(news_value=0.0, disclosure_value=0.0), FeatureConfig())
    high_context = build_features(_sample_ohlcv(news_value=1.0, disclosure_value=1.0), FeatureConfig())
    selected = select_feature_columns(low_context)

    assert selected == select_feature_columns(high_context)
    assert sorted(set(selected) & DISPLAY_ONLY_CONTEXT_COLUMNS) == []
    pd.testing.assert_frame_equal(low_context[selected], high_context[selected])
