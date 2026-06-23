from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import FeatureConfig
from src.features.feature_selection import (
    DISPLAY_ONLY_CONTEXT_COLUMNS,
    FEATURE_COLUMN_BASE,
    FEATURE_COLUMN_PREFIXES,
    MODEL_FEATURE_COLUMN_BASE,
    display_context_columns,
    select_feature_columns,
)
from src.features.technical_indicators import (
    compute_technical_indicator_block,
    rolling_zscore as _rolling_zscore,
)


KRX_PRICE_LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-15")
NEUTRAL_FEATURE_VALUES = {
    "rsi_14": 50.0,
    "stoch_k": 50.0,
    "stoch_d": 50.0,
    "macd": 0.0,
    "macd_signal": 0.0,
    "macd_hist": 0.0,
}
MISSING_INDICATOR_SOURCE_COLUMNS = ("ma_120", "vol_60", "atr_14", "cci_20", "obv_change_5d")


def _coerce_numeric_series(df: pd.DataFrame, aliases: list[str], default: float = 0.0) -> pd.Series:
    src = next((c for c in aliases if c in df.columns), None)
    if src is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[src], errors="coerce").fillna(default)


def _price_limit_pct(df: pd.DataFrame) -> pd.Series:
    explicit = next((column for column in ("price_limit_pct", "PriceLimitPct") if column in df.columns), None)
    default = pd.Series(
        np.where(pd.to_datetime(df["Date"]) < KRX_PRICE_LIMIT_CHANGE_DATE, 0.15, 0.30),
        index=df.index,
        dtype=float,
    )
    if explicit is None:
        return default
    return pd.to_numeric(df[explicit], errors="coerce").fillna(default)


def feature_missing_rate_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {column: float(df[column].isna().mean()) for column in columns if column in df.columns}


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    out["_feature_input_order"] = np.arange(len(out))
    out = out.sort_values(["Symbol", "Date", "_feature_input_order"], kind="stable")
    grouped = out.groupby("Symbol", group_keys=False)

    # Keep only the high-priority investor/event inputs that drive the
    # requested selection buckets: top-turnover disclosures, favorable news,
    # foreign/institution buying, and 52-week-high trend strength.
    numeric_alias_map = {
        "foreign_net_buy": ["foreign_net_buy", "외국인순매수", "ForeignNetBuy"],
        "institution_net_buy": ["institution_net_buy", "기관순매수", "InstitutionNetBuy"],
        "disclosure_score": ["disclosure_score", "공시점수", "DisclosureScore"],
        "news_sentiment": ["news_sentiment", "뉴스점수", "NewsSentiment"],
        "news_relevance_score": ["news_relevance_score", "뉴스관련도", "NewsRelevanceScore"],
        "news_impact_score": ["news_impact_score", "뉴스영향점수", "NewsImpactScore"],
        "news_article_count": ["news_article_count", "뉴스건수", "NewsArticleCount"],
    }
    for canonical, aliases in numeric_alias_map.items():
        out[canonical] = _coerce_numeric_series(out, aliases)

    # Accumulate all technical indicator columns in a dict and merge once to
    # avoid incremental column-assignment fragmentation warnings.
    tech_cols: dict[str, pd.Series | np.ndarray] = {}

    close_group = grouped["Close"]
    high_group = grouped["High"]
    low_group = grouped["Low"]
    volume_group = grouped["Volume"]

    log_return_s = np.log(out["Close"] / close_group.shift(1))
    tech_cols["log_return"] = log_return_s
    tech_cols["daily_return"] = close_group.pct_change()
    tech_cols["gap_return"] = (out["Open"] / close_group.shift(1)) - 1
    tech_cols["intraday_return"] = (out["Close"] / out["Open"]) - 1
    tech_cols["range_pct"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)

    value_traded_s = out["Close"] * out["Volume"]
    turnover_rank_s = value_traded_s.groupby(out["Date"]).rank(method="first", ascending=False)
    tech_cols["value_traded"] = value_traded_s
    tech_cols["turnover_rank_daily"] = turnover_rank_s
    tech_cols["is_top_turnover_3"] = (turnover_rank_s <= 3).astype(float)
    tech_cols["is_top_turnover_10"] = (turnover_rank_s <= 10).astype(float)

    for window in cfg.lookback_windows:
        tech_cols[f"ret_{window}d"] = close_group.pct_change(window)

    for window in cfg.moving_average_windows:
        ma = grouped["Close"].transform(lambda x: x.rolling(window).mean())
        tech_cols[f"ma_{window}"] = ma
        tech_cols[f"close_to_ma_{window}"] = out["Close"] / ma - 1

    for window in cfg.volatility_windows:
        tech_cols[f"vol_{window}"] = log_return_s.groupby(out["Symbol"]).transform(lambda x: x.rolling(window).std())

    tech_cols["vol_ratio_20"] = out["Volume"] / volume_group.transform(lambda x: x.rolling(20).mean())

    technical_blocks = [
        compute_technical_indicator_block(
            group,
            rsi_period=cfg.rsi_period,
            stochastic_period=cfg.stochastic_period,
            cci_period=cfg.cci_period,
        )
        for _, group in out.groupby("Symbol", sort=False)
    ]
    if technical_blocks:
        technical = pd.concat(technical_blocks).sort_index()
        tech_cols.update({column: technical[column] for column in technical.columns})
    rsi_14_s = tech_cols["rsi_14"]
    tech_cols["rsi_pullback_buy_flag"] = rsi_14_s.between(30.0, 35.0, inclusive="both").astype(float)
    tech_cols["rsi_overbought_sell_flag"] = (rsi_14_s >= 70.0).astype(float)

    out = pd.concat([out, pd.DataFrame(tech_cols, index=out.index)], axis=1)
    grouped = out.groupby("Symbol", group_keys=False)

    # Investor-context engineered features
    feature_cols: dict[str, pd.Series | np.ndarray] = {}
    feature_cols["foreign_buy_signal"] = (out["foreign_net_buy"] > 0).astype(float)
    feature_cols["institution_buy_signal"] = (out["institution_net_buy"] > 0).astype(float)
    feature_cols["smart_money_buy_signal"] = ((out["foreign_net_buy"] + out["institution_net_buy"]) > 0).astype(float)
    value_traded_safe = out["value_traded"].replace(0, np.nan)
    feature_cols["foreign_buy_ratio"] = (out["foreign_net_buy"] / value_traded_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols["institution_buy_ratio"] = (out["institution_net_buy"] / value_traded_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols["smart_money_strength"] = (
        (out["foreign_net_buy"] + out["institution_net_buy"]) / value_traded_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols["foreign_net_buy_z20"] = grouped["foreign_net_buy"].transform(lambda x: _rolling_zscore(x, 20))
    feature_cols["institution_net_buy_z20"] = grouped["institution_net_buy"].transform(lambda x: _rolling_zscore(x, 20))
    feature_cols["foreign_net_buy_3d"] = grouped["foreign_net_buy"].transform(lambda x: x.rolling(3).sum()).fillna(0.0)
    feature_cols["foreign_net_buy_5d"] = grouped["foreign_net_buy"].transform(lambda x: x.rolling(5).sum()).fillna(0.0)
    feature_cols["institution_net_buy_3d"] = grouped["institution_net_buy"].transform(lambda x: x.rolling(3).sum()).fillna(0.0)
    feature_cols["institution_net_buy_5d"] = grouped["institution_net_buy"].transform(lambda x: x.rolling(5).sum()).fillna(0.0)
    feature_cols["news_positive_signal"] = (
        out["news_relevance_score"] * (out["news_sentiment"] - 0.5).clip(lower=0.0) * 2.0
    )
    feature_cols["news_negative_signal"] = (
        out["news_relevance_score"] * (0.5 - out["news_sentiment"]).clip(lower=0.0) * 2.0
    )

    rolling_high_252 = grouped["Close"].transform(lambda x: x.rolling(252, min_periods=20).max())
    prev_rolling_high_252 = grouped["Close"].transform(lambda x: x.shift(1).rolling(252, min_periods=20).max())
    close_to_52w_high = out["Close"] / rolling_high_252.replace(0, np.nan)
    feature_cols["close_to_52w_high"] = close_to_52w_high
    feature_cols["breakout_52w_flag"] = (out["Close"] >= prev_rolling_high_252.fillna(np.inf)).astype(float)
    top3_positive_count = (
        ((out["turnover_rank_daily"] <= 3) & (out["daily_return"] > 0)).astype(float).groupby(out["Date"]).transform("sum")
    )
    feature_cols["leader_confirmation_flag"] = (
        (out["turnover_rank_daily"] == 1)
        & (out["daily_return"] > 0)
        & (top3_positive_count >= 3)
    ).astype(float)

    feature_cols["investor_event_score"] = (
        0.35 * out["is_top_turnover_10"]
        + 0.20 * out["disclosure_score"]
        + 0.20 * feature_cols["news_positive_signal"]
        + 0.25 * feature_cols["smart_money_buy_signal"]
    )
    price_limit_threshold = _price_limit_pct(out) - 0.005
    limit_hit_up_flag = out["daily_return"].ge(price_limit_threshold).astype(float)
    limit_hit_down_flag = out["daily_return"].le(-price_limit_threshold).astype(float)
    feature_cols["limit_hit_up_flag"] = limit_hit_up_flag
    feature_cols["limit_hit_down_flag"] = limit_hit_down_flag
    feature_cols["limit_event_flag"] = ((limit_hit_up_flag + limit_hit_down_flag) > 0).astype(float)
    out = pd.concat([out, pd.DataFrame(feature_cols, index=out.index)], axis=1)

    drop_source_cols = [
        "개인순매수",
        "PersonalNetBuy",
        "외국인보유비중",
        "ForeignOwnershipRatio",
        "프로그램순매수",
        "ProgramTradingFlow",
        "시장구분",
        "MarketType",
        "거래소",
        "Venue",
        "세션",
        "Session",
        "상장일",
        "ListingDate",
        "상장후일수",
        "DaysSinceListing",
        "투자경보단계",
        "WarningLevel",
        "시장경보",
        "거래정지",
        "HaltFlag",
        "VI발동",
        "VIFlag",
        "VI횟수",
        "VICount",
        "단기과열종목",
        "ShortTermOverheatFlag",
        "공매도가능",
        "ShortSellFlag",
        "공매도잔고",
        "ShortSellBalance",
        "공매도비중",
        "ShortSellRatio",
        "공매도과열종목",
        "ShortSellOverheatFlag",
        "PBR",
        "PER",
        "ROE",
        "배당수익률",
        "DividendYield",
        "자사주취득",
        "BuybackFlag",
        "자사주소각",
        "ShareCancellationFlag",
        "밸류업공시",
        "ValueUpDisclosureFlag",
    ]
    out = out.drop(columns=[c for c in drop_source_cols if c in out.columns], errors="ignore")

    out = out.copy()
    out["target_log_return"] = grouped["Close"].transform(lambda x: np.log(x.shift(-1) / x))
    out["target_up"] = (out["target_log_return"] > 0).astype(int)
    out["target_close"] = out["Close"] * np.exp(out["target_log_return"])
    out = out.replace([np.inf, -np.inf], np.nan)
    for column in MISSING_INDICATOR_SOURCE_COLUMNS:
        if column in out.columns:
            out[f"{column}_missing"] = out[column].isna().astype(float)
    for column, neutral in NEUTRAL_FEATURE_VALUES.items():
        if column in out.columns:
            out[column] = out[column].fillna(neutral)
    return out.sort_values("_feature_input_order", kind="stable").drop(columns="_feature_input_order")
