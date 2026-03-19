import pandas as pd

from src.config.settings import AppConfig
from src.features.price_features import build_features
from src.pipeline import _feature_columns


EXPECTED_CORE_CONTEXT_COLUMNS = [
    "foreign_net_buy",
    "institution_net_buy",
    "disclosure_score",
    "news_sentiment",
    "news_relevance_score",
    "news_impact_score",
    "news_article_count",
    "value_traded",
    "turnover_rank_daily",
    "is_top_turnover_10",
    "foreign_buy_signal",
    "institution_buy_signal",
    "smart_money_buy_signal",
    "news_positive_signal",
    "news_negative_signal",
    "close_to_52w_high",
    "near_52w_high_flag",
    "breakout_52w_flag",
    "investor_event_score",
]


REMOVED_LOW_PRIORITY_COLUMNS = [
    "individual_net_buy",
    "program_trading_flow",
    "warning_level",
    "short_sell_ratio",
    "buyback_flag",
    "market_type_kosdaq",
]


def test_investor_feature_columns_are_created_from_optional_inputs():
    cfg = AppConfig()
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "Date": d,
                "Symbol": "AAA",
                "Open": 100 + i,
                "High": 101 + i,
                "Low": 99 + i,
                "Close": 100 + i,
                "Volume": 100000 + i * 100,
                "외국인순매수": 1000 if i % 2 == 0 else -1000,
                "기관순매수": 500 if i % 3 == 0 else -300,
                "공시점수": 0.7,
                "뉴스점수": 0.4,
                "뉴스관련도": 0.8,
                "뉴스영향점수": 0.3,
                "뉴스건수": 2,
            }
        )

    df = pd.DataFrame(rows)
    out = build_features(df, cfg.feature)
    feature_cols = _feature_columns(out)

    for c in EXPECTED_CORE_CONTEXT_COLUMNS:
        assert c in out.columns
        assert c in feature_cols

    assert out["foreign_buy_signal"].isin([0.0, 1.0]).all()
    for removed in REMOVED_LOW_PRIORITY_COLUMNS:
        assert removed not in out.columns


def test_build_features_does_not_require_removed_vi_or_short_sell_columns():
    cfg = AppConfig()
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=30, freq="B"),
            "Symbol": ["AAA"] * 30,
            "Open": [100 + i for i in range(30)],
            "High": [101 + i for i in range(30)],
            "Low": [99 + i for i in range(30)],
            "Close": [100 + i for i in range(30)],
            "Volume": [100000 + i * 100 for i in range(30)],
            "외국인순매수": [1000] * 30,
            "기관순매수": [500] * 30,
            "공시점수": [0.6] * 30,
            "뉴스점수": [0.7] * 30,
            "뉴스관련도": [0.8] * 30,
            "뉴스영향점수": [0.3] * 30,
            "뉴스건수": [2] * 30,
        }
    )

    out = build_features(df, cfg.feature)

    assert "vi_flag" not in out.columns
    assert "short_sell_ratio" not in out.columns
    assert "vi_after_return" not in out.columns
    assert "short_sell_event_score" not in out.columns
    assert "foreign_net_buy" in out.columns
    assert "smart_money_buy_signal" in out.columns
