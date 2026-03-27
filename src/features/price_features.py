from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import FeatureConfig


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


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    k = 100 * (close - ll) / (hh - ll + 1e-9)
    d = k.rolling(3).mean()
    return k, d


def _compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


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
    for column, default in legacy_removed_default_map.items():
        if column not in out.columns:
            out[column] = default

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

    out["log_return"] = grouped["Close"].transform(lambda x: np.log(x / x.shift(1)))
    out["daily_return"] = grouped["Close"].transform(lambda x: x.pct_change())
    out["gap_return"] = (out["Open"] / grouped["Close"].shift(1)) - 1
    out["intraday_return"] = (out["Close"] / out["Open"]) - 1
    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    out["value_traded"] = out["Close"] * out["Volume"]
    out["turnover_rank_daily"] = out.groupby("Date")["value_traded"].rank(method="first", ascending=False)
    out["is_top_turnover_3"] = (out["turnover_rank_daily"] <= 3).astype(float)
    out["is_top_turnover_10"] = (out["turnover_rank_daily"] <= 10).astype(float)

    for window in cfg.lookback_windows:
        out[f"ret_{window}d"] = grouped["Close"].transform(lambda x: x.pct_change(window))

    for window in cfg.moving_average_windows:
        ma = grouped["Close"].transform(lambda x: x.rolling(window).mean())
        out[f"ma_{window}"] = ma
        out[f"close_to_ma_{window}"] = out["Close"] / ma - 1

    for window in cfg.volatility_windows:
        out[f"vol_{window}"] = grouped["log_return"].transform(lambda x: x.rolling(window).std())

    out["vol_ratio_20"] = out["Volume"] / grouped["Volume"].transform(lambda x: x.rolling(20).mean())
    out["rsi_14"] = grouped["Close"].transform(lambda x: _compute_rsi(x, cfg.rsi_period))
    out["rsi_pullback_buy_flag"] = out["rsi_14"].between(30.0, 35.0, inclusive="both").astype(float)
    out["rsi_overbought_sell_flag"] = (out["rsi_14"] >= 70.0).astype(float)

    macd = grouped["Close"].transform(lambda x: _compute_macd(x)[0])
    macd_sig = grouped["Close"].transform(lambda x: _compute_macd(x)[1])
    macd_hist = grouped["Close"].transform(lambda x: _compute_macd(x)[2])
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    high_group = grouped["High"]
    low_group = grouped["Low"]
    close_group = grouped["Close"]

    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - close_group.shift(1)).abs()
    tr3 = (out["Low"] - close_group.shift(1)).abs()
    out["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).groupby(out["Symbol"]).transform(
        lambda x: x.rolling(14).mean()
    )

    low_min = low_group.transform(lambda x: x.rolling(cfg.stochastic_period).min())
    high_max = high_group.transform(lambda x: x.rolling(cfg.stochastic_period).max())
    out["stoch_k"] = 100 * (out["Close"] - low_min) / (high_max - low_min + 1e-9)
    out["stoch_d"] = out.groupby("Symbol", group_keys=False)["stoch_k"].transform(lambda x: x.rolling(3).mean())

    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    tp_ma = typical_price.groupby(out["Symbol"]).transform(lambda x: x.rolling(cfg.cci_period).mean())
    tp_mad = typical_price.groupby(out["Symbol"]).transform(
        lambda x: x.rolling(cfg.cci_period).apply(lambda y: np.mean(np.abs(y - np.mean(y))), raw=True)
    )
    out["cci_20"] = (typical_price - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    price_direction = np.sign(close_group.diff().fillna(0))
    out["obv"] = (price_direction * out["Volume"].fillna(0)).groupby(out["Symbol"]).cumsum()
    out["obv_change_5d"] = grouped["obv"].transform(lambda x: x.pct_change(5))

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
    horizon_map = {
        1: ("target_log_return", "target_up", "target_close"),
        5: ("target_log_return_5d", "target_up_5d", "target_close_5d"),
        20: ("target_log_return_20d", "target_up_20d", "target_close_20d"),
    }
    for horizon, (return_col, up_col, close_col) in horizon_map.items():
        out[return_col] = grouped["Close"].transform(lambda x, h=horizon: np.log(x.shift(-h) / x))
        out[up_col] = (out[return_col] > 0).astype(int)
        out[close_col] = out["Close"] * np.exp(out[return_col])
    return out
