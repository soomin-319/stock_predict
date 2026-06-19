import inspect

import pandas as pd

from src.config.settings import InvestmentCriteriaConfig
from src.features import investment_signals
from src.features.investment_signals import add_investment_signal_features


def test_add_investment_signal_features_applies_threshold_flags_and_leader_confirmation():
    cfg = InvestmentCriteriaConfig(
        top_turnover_rank=15,
        high_conviction_net_buy_krw=100_000_000_000.0,
        nasdaq_tailwind_threshold=0.01,
        nasdaq_headwind_threshold=-0.01,
        rsi_buy_watch_low=30.0,
        rsi_buy_watch_high=35.0,
        rsi_overbought=70.0,
        near_52w_distance_threshold=0.03,
    )
    df = pd.DataFrame(
        [
            {
                "Date": "2026-03-28",
                "Symbol": "A",
                "value_traded": 900_000_000_000,
                "daily_return": 0.08,
                "foreign_net_buy": 120_000_000_000,
                "institution_net_buy": 130_000_000_000,
                "close_to_52w_high": 1.02,
                "nq_f_ret_1d": 0.012,
                "rsi_14": 33.0,
                "news_article_count": 3,
                "disclosure_score": 1.0,
            },
            {
                "Date": "2026-03-28",
                "Symbol": "B",
                "value_traded": 700_000_000_000,
                "daily_return": 0.03,
                "foreign_net_buy": 0.0,
                "institution_net_buy": 0.0,
                "close_to_52w_high": 0.99,
                "nq_f_ret_1d": 0.012,
                "rsi_14": 45.0,
                "news_article_count": 0,
                "disclosure_score": 0.0,
            },
        ]
    )

    out = add_investment_signal_features(df, cfg)
    row_a = out[out["Symbol"] == "A"].iloc[0]

    assert row_a["is_top_turnover_15"] == 1
    assert row_a["dual_high_conviction_buy_flag"] == 1
    assert row_a["breakout_52w_flag"] == 1
    assert row_a["nasdaq_tailwind_flag"] == 1
    assert row_a["rsi_buy_watch_flag"] == 1
    assert row_a["leader_confirmation_flag"] == 1
    assert row_a["news_same_day_signal"] == 1
    assert row_a["disclosure_same_day_signal"] == 1


def test_add_investment_signal_features_handles_missing_columns_without_crashing():
    cfg = InvestmentCriteriaConfig()
    df = pd.DataFrame([{"Date": "2026-03-28", "Symbol": "A"}])

    out = add_investment_signal_features(df, cfg)

    assert "is_top_turnover_15" in out.columns
    assert "dual_high_conviction_buy_flag" in out.columns
    assert "nasdaq_headwind_flag" in out.columns


def test_leader_confirmation_is_vectorized_without_group_item_loop():
    source = inspect.getsource(investment_signals._leader_confirmation)

    assert ".groups.items()" not in source
    assert "for _, idx in out.groupby" not in source


def test_leader_confirmation_populates_group_values_without_reordering_rows():
    cfg = InvestmentCriteriaConfig(leader_top_n=3, leader_min_co_movers=2, leader_min_return=0.02)
    df = pd.DataFrame(
        [
            {"Date": "2026-03-29", "Symbol": "D", "turnover_rank_daily": 4, "daily_return": -0.01},
            {"Date": "2026-03-28", "Symbol": "B", "turnover_rank_daily": 2, "daily_return": 0.03},
            {"Date": "2026-03-28", "Symbol": "A", "turnover_rank_daily": 1, "daily_return": 0.08},
            {"Date": "2026-03-28", "Symbol": "C", "turnover_rank_daily": 3, "daily_return": 0.01},
            {"Date": "2026-03-29", "Symbol": "E", "turnover_rank_daily": 1, "daily_return": 0.01},
            {"Date": "2026-03-29", "Symbol": "F", "turnover_rank_daily": 2, "daily_return": 0.04},
        ]
    )

    out = add_investment_signal_features(df, cfg)

    assert out["Symbol"].tolist() == df["Symbol"].tolist()
    day1 = out[out["Date"] == "2026-03-28"]
    assert day1["leader_1_return"].tolist() == [0.08, 0.08, 0.08]
    assert day1["leader_2_return"].tolist() == [0.03, 0.03, 0.03]
    assert day1["leader_3_return"].tolist() == [0.01, 0.01, 0.01]
    assert day1["leader_confirmation_flag"].tolist() == [1, 1, 1]
    day2 = out[out["Date"] == "2026-03-29"]
    assert day2["leader_confirmation_flag"].tolist() == [0, 0, 0]
