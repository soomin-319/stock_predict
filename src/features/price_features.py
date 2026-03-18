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


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("Symbol", group_keys=False)

    # Optional investor-context columns (if provided by upstream data pipeline)
    numeric_alias_map = {
        "individual_net_buy": ["individual_net_buy", "개인순매수", "PersonalNetBuy"],
        "foreign_net_buy": ["foreign_net_buy", "외국인순매수", "ForeignNetBuy"],
        "institution_net_buy": ["institution_net_buy", "기관순매수", "InstitutionNetBuy"],
        "foreign_ownership_ratio": ["foreign_ownership_ratio", "외국인보유비중", "ForeignOwnershipRatio"],
        "program_trading_flow": ["program_trading_flow", "프로그램순매수", "ProgramTradingFlow"],
        "disclosure_score": ["disclosure_score", "공시점수", "DisclosureScore"],
        "news_sentiment": ["news_sentiment", "뉴스점수", "NewsSentiment"],
        "news_relevance_score": ["news_relevance_score", "뉴스관련도", "NewsRelevanceScore"],
        "news_impact_score": ["news_impact_score", "뉴스영향점수", "NewsImpactScore"],
        "news_article_count": ["news_article_count", "뉴스건수", "NewsArticleCount"],
        "short_sell_balance": ["short_sell_balance", "공매도잔고", "ShortSellBalance"],
        "short_sell_ratio": ["short_sell_ratio", "공매도비중", "ShortSellRatio"],
        "vi_count": ["vi_count", "VI횟수", "VICount"],
        "pbr": ["pbr", "PBR"],
        "per": ["per", "PER"],
        "roe": ["roe", "ROE"],
        "dividend_yield": ["dividend_yield", "배당수익률", "DividendYield"],
    }
    for canonical, aliases in numeric_alias_map.items():
        out[canonical] = _coerce_numeric_series(out, aliases)

    market_type_series = _coerce_category_series(out, ["market_type", "시장구분", "MarketType"], default="")
    if not market_type_series.ne("").any():
        market_type_series = out["Symbol"].astype(str).map(
            lambda s: "KOSPI" if s.endswith(".KS") else ("KOSDAQ" if s.endswith(".KQ") else "UNKNOWN")
        )
    market_type_normalized = market_type_series.str.upper()
    out["market_type_kospi"] = (market_type_normalized == "KOSPI").astype(float)
    out["market_type_kosdaq"] = (market_type_normalized == "KOSDAQ").astype(float)
    out["market_type_konex"] = (market_type_normalized == "KONEX").astype(float)

    venue_series = _coerce_category_series(out, ["venue", "거래소", "Venue"], default="KRX").str.upper()
    out["venue_krx"] = (venue_series == "KRX").astype(float)
    out["venue_nxt"] = (venue_series == "NXT").astype(float)

    session_map = {
        "정규장": "REGULAR",
        "regular": "REGULAR",
        "프리마켓": "PREMARKET",
        "premarket": "PREMARKET",
        "장전": "PREMARKET",
        "애프터마켓": "AFTERMARKET",
        "aftermarket": "AFTERMARKET",
        "시간외": "OFFHOURS",
        "offhours": "OFFHOURS",
    }
    session_series = _coerce_category_series(out, ["session", "세션", "Session"], default="REGULAR")
    session_normalized = session_series.astype(str).str.strip().str.lower().map(session_map).fillna("REGULAR")
    out["session_regular"] = (session_normalized == "REGULAR").astype(float)
    out["session_premarket"] = (session_normalized == "PREMARKET").astype(float)
    out["session_aftermarket"] = (session_normalized == "AFTERMARKET").astype(float)
    out["session_offhours"] = (session_normalized == "OFFHOURS").astype(float)

    listing_date_src = next((c for c in ["listing_date", "상장일", "ListingDate"] if c in out.columns), None)
    if listing_date_src is not None:
        listing_dates = pd.to_datetime(out[listing_date_src], errors="coerce")
        out["days_since_listing"] = (pd.to_datetime(out["Date"]) - listing_dates).dt.days.clip(lower=0).fillna(9999)
    else:
        out["days_since_listing"] = _coerce_numeric_series(out, ["days_since_listing", "상장후일수", "DaysSinceListing"], default=9999.0)
    out["is_newly_listed"] = (out["days_since_listing"] <= 20).astype(float)
    out["is_newly_listed_60d"] = (out["days_since_listing"] <= 60).astype(float)

    out["warning_level"] = _warning_level_series(out)
    out["market_warning_flag"] = (out["warning_level"] > 0).astype(float)
    out["halt_flag"] = _coerce_flag_series(out, ["halt_flag", "거래정지", "HaltFlag"])
    out["vi_flag"] = _coerce_flag_series(out, ["vi_flag", "VI발동", "VIFlag"])
    out["short_term_overheat_flag"] = _coerce_flag_series(
        out, ["short_term_overheat_flag", "단기과열종목", "ShortTermOverheatFlag"]
    )
    out["short_sell_flag"] = _coerce_flag_series(out, ["short_sell_flag", "공매도가능", "ShortSellFlag"])
    out["short_sell_overheat_flag"] = _coerce_flag_series(
        out, ["short_sell_overheat_flag", "공매도과열종목", "ShortSellOverheatFlag"]
    )
    out["buyback_flag"] = _coerce_flag_series(out, ["buyback_flag", "자사주취득", "BuybackFlag"])
    out["share_cancellation_flag"] = _coerce_flag_series(
        out, ["share_cancellation_flag", "자사주소각", "ShareCancellationFlag"]
    )
    out["value_up_disclosure_flag"] = _coerce_flag_series(
        out, ["value_up_disclosure_flag", "밸류업공시", "ValueUpDisclosureFlag"]
    )

    out["log_return"] = grouped["Close"].transform(lambda x: np.log(x / x.shift(1)))
    out["daily_return"] = grouped["Close"].transform(lambda x: x.pct_change())
    out["gap_return"] = (out["Open"] / grouped["Close"].shift(1)) - 1
    out["intraday_return"] = (out["Close"] / out["Open"]) - 1
    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    out["value_traded"] = out["Close"] * out["Volume"]
    out["turnover_rank_daily"] = out.groupby("Date")["value_traded"].rank(method="first", ascending=False)
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
    out["foreign_buy_signal"] = (out["foreign_net_buy"] > 0).astype(float)
    out["institution_buy_signal"] = (out["institution_net_buy"] > 0).astype(float)
    out["smart_money_buy_signal"] = ((out["foreign_net_buy"] + out["institution_net_buy"]) > 0).astype(float)
    out["individual_buy_signal"] = (out["individual_net_buy"] > 0).astype(float)
    out["retail_chase_signal"] = (
        (out["individual_net_buy"] > 0) & (out["foreign_net_buy"] < 0) & (out["institution_net_buy"] < 0)
    ).astype(float)
    out["news_positive_signal"] = (
        out["news_relevance_score"] * (out["news_sentiment"] - 0.5).clip(lower=0.0) * 2.0
    )
    out["news_negative_signal"] = (
        out["news_relevance_score"] * (0.5 - out["news_sentiment"]).clip(lower=0.0) * 2.0
    )

    rolling_high_252 = grouped["Close"].transform(lambda x: x.rolling(252, min_periods=20).max())
    prev_rolling_high_252 = grouped["Close"].transform(lambda x: x.shift(1).rolling(252, min_periods=20).max())
    out["close_to_52w_high"] = out["Close"] / rolling_high_252.replace(0, np.nan)
    out["near_52w_high_flag"] = (out["close_to_52w_high"] >= 0.95).astype(float)
    out["breakout_52w_flag"] = (out["Close"] >= prev_rolling_high_252.fillna(np.inf)).astype(float)

    out["investor_event_score"] = (
        0.35 * out["is_top_turnover_10"]
        + 0.20 * out["disclosure_score"]
        + 0.20 * out["news_positive_signal"]
        + 0.15 * out["smart_money_buy_signal"]
        + 0.10 * out["near_52w_high_flag"]
    )
    out["limit_hit_up_flag"] = (out["daily_return"] >= 0.295).astype(float)
    out["limit_hit_down_flag"] = (out["daily_return"] <= -0.295).astype(float)
    out["limit_event_flag"] = ((out["limit_hit_up_flag"] + out["limit_hit_down_flag"]) > 0).astype(float)
    out["vi_after_return"] = out["daily_return"].fillna(0.0) * out["vi_flag"]
    out["vi_after_volume_spike"] = out["vol_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0) * out["vi_flag"]
    out["short_sell_event_score"] = (
        0.5 * out["short_sell_overheat_flag"]
        + 0.3 * out["short_sell_flag"]
        + 0.2 * (out["short_sell_ratio"] > 0).astype(float)
    )
    out["shareholder_return_score"] = (
        0.4 * out["buyback_flag"]
        + 0.3 * out["share_cancellation_flag"]
        + 0.3 * out["value_up_disclosure_flag"]
    )

    out = out.copy()
    out["target_log_return"] = grouped["Close"].transform(lambda x: np.log(x.shift(-1) / x))
    out["target_up"] = (out["target_log_return"] > 0).astype(int)
    out["target_close"] = out["Close"] * np.exp(out["target_log_return"])
    return out
