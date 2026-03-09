from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import FeatureConfig


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

    out["log_return"] = grouped["Close"].transform(lambda x: np.log(x / x.shift(1)))
    out["daily_return"] = grouped["Close"].transform(lambda x: x.pct_change())
    out["gap_return"] = (out["Open"] / grouped["Close"].shift(1)) - 1
    out["intraday_return"] = (out["Close"] / out["Open"]) - 1
    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)

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

    out["atr_14"] = grouped.apply(
        lambda g: _compute_atr(g["High"], g["Low"], g["Close"], period=14)
    ).reset_index(level=0, drop=True)

    stoch_k = grouped.apply(
        lambda g: _compute_stochastic(g["High"], g["Low"], g["Close"], period=cfg.stochastic_period)[0]
    ).reset_index(level=0, drop=True)
    stoch_d = grouped.apply(
        lambda g: _compute_stochastic(g["High"], g["Low"], g["Close"], period=cfg.stochastic_period)[1]
    ).reset_index(level=0, drop=True)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    out["cci_20"] = grouped.apply(
        lambda g: _compute_cci(g["High"], g["Low"], g["Close"], period=cfg.cci_period)
    ).reset_index(level=0, drop=True)

    out["obv"] = grouped.apply(lambda g: _compute_obv(g["Close"], g["Volume"])).reset_index(level=0, drop=True)
    out["obv_change_5d"] = grouped["obv"].transform(lambda x: x.pct_change(5))

    out["target_log_return"] = grouped["Close"].transform(lambda x: np.log(x.shift(-1) / x))
    out["target_up"] = (out["target_log_return"] > 0).astype(int)
    out["target_close"] = out["Close"] * np.exp(out["target_log_return"])
    return out
