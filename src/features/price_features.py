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
    compute_macd as _compute_macd,
    compute_rsi as _compute_rsi,
    rolling_zscore as _rolling_zscore,
)


WARNING_LEVEL_MAP = {
    "none": 0.0,
    "normal": 0.0,
    "투자주의": 1.0,
    "주의": 1.0,
    "investment_caution": 1.0,
    "투자경고": 2.0,
    "경고": 2.0,
    "investment_warning": 2.0,
    "투자위험": 3.0,
    "위험": 3.0,
    "investment_risk": 3.0,
}


def _coerce_numeric_series(df: pd.DataFrame, aliases: list[str], default: float = 0.0) -> pd.Series:
    src = next((c for c in aliases if c in df.columns), None)
    if src is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[src], errors="coerce").fillna(default)


def _coerce_flag_series(df: pd.DataFrame, aliases: list[str], truthy: set[str] | None = None) -> pd.Series:
    src = next((c for c in aliases if c in df.columns), None)
    if src is None:
        return pd.Series(0.0, index=df.index, dtype=float)

    values = df[src]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(float)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).gt(0).astype(float)

    valid = truthy or {"1", "y", "yes", "true", "t", "on", "발동", "지정", "해당", "krx", "nxt"}
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin(valid).astype(float)


def _coerce_category_series(df: pd.DataFrame, aliases: list[str], default: str) -> pd.Series:
    src = next((c for c in aliases if c in df.columns), None)
    if src is None:
        return pd.Series(default, index=df.index, dtype="object")
    return df[src].astype(str).str.strip().replace({"": default}).fillna(default)


def _warning_level_series(df: pd.DataFrame) -> pd.Series:
    src = next((c for c in ["warning_level", "시장경보", "투자경보단계", "WarningLevel"] if c in df.columns), None)
    if src is None:
        return pd.Series(0.0, index=df.index, dtype=float)

    values = df[src]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0)

    normalized = values.astype(str).str.strip().str.lower()
    mapped = normalized.map(WARNING_LEVEL_MAP)
    return mapped.fillna(0.0).astype(float)


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("Symbol", group_keys=False)

    # Backward-compatible defaults for legacy low-priority fields that may
    # still be referenced in older local copies or partially-updated branches.
    # They are dropped again before returning so the final feature set remains
    # limited to the requested key catalyst signals.
    legacy_removed_default_map = {
        "individual_net_buy": 0.0,
        "foreign_ownership_ratio": 0.0,
        "program_trading_flow": 0.0,
        "market_type_kospi": 0.0,
        "market_type_kosdaq": 0.0,
        "market_type_konex": 0.0,
        "venue_krx": 0.0,
        "venue_nxt": 0.0,
        "session_regular": 0.0,
        "session_premarket": 0.0,
        "session_aftermarket": 0.0,
        "session_offhours": 0.0,
        "days_since_listing": 9999.0,
        "is_newly_listed": 0.0,
        "is_newly_listed_60d": 0.0,
        "warning_level": 0.0,
        "market_warning_flag": 0.0,
        "halt_flag": 0.0,
        "vi_flag": 0.0,
        "vi_count": 0.0,
        "short_term_overheat_flag": 0.0,
        "short_sell_flag": 0.0,
        "short_sell_balance": 0.0,
        "short_sell_ratio": 0.0,
        "short_sell_overheat_flag": 0.0,
        "individual_buy_signal": 0.0,
        "retail_chase_signal": 0.0,
        "limit_hit_up_flag": 0.0,
        "limit_hit_down_flag": 0.0,
        "limit_event_flag": 0.0,
        "vi_after_return": 0.0,
        "vi_after_volume_spike": 0.0,
        "pbr": 0.0,
        "per": 0.0,
        "roe": 0.0,
        "dividend_yield": 0.0,
        "buyback_flag": 0.0,
        "share_cancellation_flag": 0.0,
        "value_up_disclosure_flag": 0.0,
        "shareholder_return_score": 0.0,
        "short_sell_event_score": 0.0,
    }
    missing_defaults = {
        column: pd.Series(default, index=out.index, dtype=float)
        for column, default in legacy_removed_default_map.items()
        if column not in out.columns
    }
    if missing_defaults:
        out = pd.concat([out, pd.DataFrame(missing_defaults, index=out.index)], axis=1)
        grouped = out.groupby("Symbol", group_keys=False)

    # Keep only the high-priority investor/event inputs that drive the
    # requested selection buckets: top-turnover disclosures, favorable news,
    # foreign/institution buying, and 52-week-high trend strength.
    numeric_alias_map = {
        "foreign_net_buy": ["foreign_net_buy", "외국인순매수", "ForeignNetBuy"],
        "institution_net_buy": ["institution_net_buy", "기관순매수", "InstitutionNetBuy"],
        "foreign_ownership_ratio": ["foreign_ownership_ratio", "외국인보유비중", "ForeignOwnershipRatio"],
        "program_trading_flow": ["program_trading_flow", "프로그램순매수", "ProgramTradingFlow"],
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

    rsi_14_s = grouped["Close"].transform(lambda x: _compute_rsi(x, cfg.rsi_period))
    tech_cols["rsi_14"] = rsi_14_s
    tech_cols["rsi_pullback_buy_flag"] = rsi_14_s.between(30.0, 35.0, inclusive="both").astype(float)
    tech_cols["rsi_overbought_sell_flag"] = (rsi_14_s >= 70.0).astype(float)

    def _macd_group(g: pd.DataFrame) -> pd.DataFrame:
        m, s, h = _compute_macd(g["Close"])
        return pd.DataFrame({"macd": m.values, "macd_signal": s.values, "macd_hist": h.values}, index=g.index)

    macd_df = out.groupby("Symbol", group_keys=False)[["Close"]].apply(_macd_group)
    tech_cols["macd"] = macd_df["macd"]
    tech_cols["macd_signal"] = macd_df["macd_signal"]
    tech_cols["macd_hist"] = macd_df["macd_hist"]

    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - close_group.shift(1)).abs()
    tr3 = (out["Low"] - close_group.shift(1)).abs()
    tech_cols["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).groupby(out["Symbol"]).transform(
        lambda x: x.rolling(14).mean()
    )

    low_min = low_group.transform(lambda x: x.rolling(cfg.stochastic_period).min())
    high_max = high_group.transform(lambda x: x.rolling(cfg.stochastic_period).max())
    stoch_k_s = 100 * (out["Close"] - low_min) / (high_max - low_min + 1e-9)
    tech_cols["stoch_k"] = stoch_k_s
    tech_cols["stoch_d"] = stoch_k_s.groupby(out["Symbol"]).transform(lambda x: x.rolling(3).mean())

    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    tp_ma = typical_price.groupby(out["Symbol"]).transform(lambda x: x.rolling(cfg.cci_period).mean())
    tp_mad = typical_price.groupby(out["Symbol"]).transform(
        lambda x: x.rolling(cfg.cci_period).apply(lambda y: np.mean(np.abs(y - np.mean(y))), raw=True)
    )
    tech_cols["cci_20"] = (typical_price - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    price_direction = np.sign(close_group.diff().fillna(0))
    obv_s = (price_direction * out["Volume"].fillna(0)).groupby(out["Symbol"]).cumsum()
    tech_cols["obv"] = obv_s
    tech_cols["obv_change_5d"] = obv_s.groupby(out["Symbol"]).transform(lambda x: x.pct_change(5))

    out = pd.concat([out, pd.DataFrame(tech_cols, index=out.index)], axis=1)

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
    near_52w_high_flag = (close_to_52w_high >= 0.95).astype(float)
    feature_cols["close_to_52w_high"] = close_to_52w_high
    feature_cols["near_52w_high_flag"] = near_52w_high_flag
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
        + 0.15 * feature_cols["smart_money_buy_signal"]
        + 0.10 * near_52w_high_flag
    )
    limit_hit_up_flag = (out["daily_return"] >= 0.295).astype(float)
    limit_hit_down_flag = (out["daily_return"] <= -0.295).astype(float)
    feature_cols["limit_hit_up_flag"] = limit_hit_up_flag
    feature_cols["limit_hit_down_flag"] = limit_hit_down_flag
    feature_cols["limit_event_flag"] = ((limit_hit_up_flag + limit_hit_down_flag) > 0).astype(float)
    feature_cols["vi_after_return"] = out["daily_return"].fillna(0.0) * out["vi_flag"]
    feature_cols["vi_after_volume_spike"] = out["vol_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0) * out["vi_flag"]
    feature_cols["short_sell_event_score"] = (
        0.5 * out["short_sell_overheat_flag"]
        + 0.3 * out["short_sell_flag"]
        + 0.2 * (out["short_sell_ratio"] > 0).astype(float)
    )
    feature_cols["shareholder_return_score"] = (
        0.4 * out["buyback_flag"]
        + 0.3 * out["share_cancellation_flag"]
        + 0.3 * out["value_up_disclosure_flag"]
    )
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
    out = out.drop(columns=list(legacy_removed_default_map.keys()), errors="ignore")

    out = out.copy()
    out["target_log_return"] = grouped["Close"].transform(lambda x: np.log(x.shift(-1) / x))
    out["target_up"] = (out["target_log_return"] > 0).astype(int)
    out["target_close"] = out["Close"] * np.exp(out["target_log_return"])
    return out
