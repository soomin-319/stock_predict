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
                "시장구분": "KOSDAQ",
                "거래소": "NXT" if i % 5 == 0 else "KRX",
                "세션": "정규장" if i % 7 else "시간외",
                "상장일": "2023-12-01",
                "투자경보단계": "투자경고" if i % 4 == 0 else "없음",
                "거래정지": 1 if i == 10 else 0,
                "VI발동": 1 if i % 6 == 0 else 0,
                "VI횟수": 2 if i % 6 == 0 else 0,
                "단기과열종목": 1 if i % 9 == 0 else 0,
                "공매도가능": 1,
                "공매도잔고": 10000 + i,
                "공매도비중": 0.03,
                "공매도과열종목": 1 if i % 8 == 0 else 0,
                "PBR": 0.9,
                "PER": 8.0,
                "ROE": 0.11,
                "배당수익률": 0.025,
                "자사주취득": 1 if i % 10 == 0 else 0,
                "자사주소각": 1 if i % 11 == 0 else 0,
                "밸류업공시": 1 if i % 12 == 0 else 0,
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
        "warning_level",
        "market_warning_flag",
        "halt_flag",
        "vi_flag",
        "vi_count",
        "vi_after_return",
        "short_term_overheat_flag",
        "short_sell_flag",
        "short_sell_balance",
        "short_sell_ratio",
        "short_sell_overheat_flag",
        "pbr",
        "per",
        "roe",
        "dividend_yield",
        "buyback_flag",
        "share_cancellation_flag",
        "value_up_disclosure_flag",
        "value_traded",
        "is_top_turnover_10",
        "individual_buy_signal",
        "smart_money_buy_signal",
        "retail_chase_signal",
        "news_positive_signal",
        "news_negative_signal",
        "close_to_52w_high",
        "near_52w_high_flag",
        "investor_event_score",
        "limit_event_flag",
        "short_sell_event_score",
        "shareholder_return_score",
    ]:
        assert c in out.columns
        assert c in feature_cols

    assert out["foreign_buy_signal"].isin([0.0, 1.0]).all()
    assert out["market_type_kosdaq"].eq(1.0).all()
