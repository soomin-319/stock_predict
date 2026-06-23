from __future__ import annotations

import pandas as pd


FEATURE_COLUMN_PREFIXES = (
    "ret_",
    "ma_",
    "close_to_ma_",
    "vol_",
    "ks",
    "kq",
    "gspc",
    "ixic",
    "nq_f",
    "sox",
    "vix",
    "krw",
    "tnx",
)

# Model-eligible feature names that are not already matched by FEATURE_COLUMN_PREFIXES.
# Display-only context names (news_*, disclosure_*, *_impact_*, and composite scores)
# must NOT be listed here; they live solely in DISPLAY_ONLY_CONTEXT_COLUMNS and are kept
# out of model features by is_display_only_context_column(). The invariant is locked by
# tests/test_feature_module_boundaries.py.
FEATURE_COLUMN_BASE = frozenset(
    {
        "daily_return",
        "gap_return",
        "intraday_return",
        "range_pct",
        "vol_ratio_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "stoch_k",
        "stoch_d",
        "cci_20",
        "obv",
        "obv_change_5d",
        "value_traded",
        "turnover_rank_daily",
        "is_top_turnover_3",
        "is_top_turnover_10",
        "market_type_kospi",
        "market_type_kosdaq",
        "market_type_konex",
        "venue_krx",
        "venue_nxt",
        "session_regular",
        "session_premarket",
        "session_aftermarket",
        "session_offhours",
        "days_since_listing",
        "is_newly_listed",
        "is_newly_listed_60d",
        "individual_net_buy",
        "foreign_net_buy",
        "institution_net_buy",
        "foreign_ownership_ratio",
        "program_trading_flow",
        "foreign_buy_signal",
        "institution_buy_signal",
        "smart_money_buy_signal",
        "foreign_buy_ratio",
        "institution_buy_ratio",
        "smart_money_strength",
        "foreign_net_buy_z20",
        "institution_net_buy_z20",
        "foreign_net_buy_3d",
        "foreign_net_buy_5d",
        "institution_net_buy_3d",
        "institution_net_buy_5d",
        "close_to_52w_high",
        "near_52w_high_flag",
        "breakout_52w_flag",
        "leader_confirmation_flag",
        "rsi_pullback_buy_flag",
        "rsi_overbought_sell_flag",
        "limit_hit_up_flag",
        "limit_hit_down_flag",
        "limit_event_flag",
        "pbr",
        "per",
        "roe",
        "dividend_yield",
        "buyback_flag",
        "share_cancellation_flag",
        "shareholder_return_score",
        "short_sell_event_score",
        "is_top_turnover_15",
        "foreign_high_conviction_buy_flag",
        "institution_high_conviction_buy_flag",
        "dual_high_conviction_buy_flag",
        "distance_to_52w_high",
        "nasdaq_tailwind_flag",
        "nasdaq_headwind_flag",
        "rsi_buy_watch_flag",
        "jongbae_score",
    }
)

# Single source of truth for display-only context column names. These are attached to
# results for user display only and must never become model features, rankings, or signals.
DISPLAY_ONLY_CONTEXT_COLUMNS = frozenset(
    {
        "disclosure_score",
        "news_sentiment",
        "news_relevance_score",
        "news_impact_score",
        "news_article_count",
        "news_positive_signal",
        "news_negative_signal",
        "news_same_day_signal",
        "disclosure_same_day_signal",
        # Composite score includes news/disclosure inputs, so it is context-only.
        "investor_event_score",
    }
)

DISPLAY_ONLY_CONTEXT_PREFIXES = ("news_", "disclosure_")
DISPLAY_ONLY_CONTEXT_SUBSTRINGS = ("_news_", "_impact_")

# Defense-in-depth: FEATURE_COLUMN_BASE already omits display-only names, but subtract
# again so any future accidental addition still cannot leak into model features.
MODEL_FEATURE_COLUMN_BASE = FEATURE_COLUMN_BASE - DISPLAY_ONLY_CONTEXT_COLUMNS


def is_display_only_context_column(column: str) -> bool:
    return (
        column in DISPLAY_ONLY_CONTEXT_COLUMNS
        or column.startswith(DISPLAY_ONLY_CONTEXT_PREFIXES)
        or any(pattern in column for pattern in DISPLAY_ONLY_CONTEXT_SUBSTRINGS)
    )


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if not is_display_only_context_column(c)
        and (c.startswith(FEATURE_COLUMN_PREFIXES) or c.endswith("_missing") or c in MODEL_FEATURE_COLUMN_BASE)
    ]


def display_context_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_display_only_context_column(c)]
