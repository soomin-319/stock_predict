import pandas as pd

from src.config.settings import AppConfig
from src.features.price_features import build_features
from src.pipeline import _feature_columns


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
                "개인순매수": 1500 if i % 2 == 1 else -800,
                "외국인보유비중": 12.5,
                "프로그램순매수": 200.0,
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

    for c in [
        "market_type_kosdaq",
        "venue_nxt",
        "session_offhours",
        "days_since_listing",
        "is_newly_listed_60d",
        "individual_net_buy",
        "foreign_net_buy",
        "institution_net_buy",
        "foreign_ownership_ratio",
        "program_trading_flow",
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
        "limit_event_flag",
        "short_sell_event_score",
        "shareholder_return_score",
    ]:
        assert c in out.columns
        assert c in feature_cols

    assert out["foreign_buy_signal"].isin([0.0, 1.0]).all()
    for removed in [
        "individual_net_buy",
        "program_trading_flow",
        "warning_level",
        "short_sell_ratio",
        "buyback_flag",
        "market_type_kosdaq",
    ]:
        assert removed not in out.columns
