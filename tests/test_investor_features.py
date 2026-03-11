import pandas as pd

from src.config.settings import AppConfig
from src.features.price_features import build_features


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
            }
        )

    df = pd.DataFrame(rows)
    out = build_features(df, cfg.feature)

    for c in [
        "foreign_net_buy",
        "institution_net_buy",
        "disclosure_score",
        "news_sentiment",
        "value_traded",
        "is_top_turnover_10",
        "smart_money_buy_signal",
        "close_to_52w_high",
        "near_52w_high_flag",
        "investor_event_score",
    ]:
        assert c in out.columns

    assert out["foreign_buy_signal"].isin([0.0, 1.0]).all()
