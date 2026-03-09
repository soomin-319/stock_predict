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

    out["target_log_return"] = grouped["Close"].transform(lambda x: np.log(x.shift(-1) / x))
    out["target_up"] = (out["target_log_return"] > 0).astype(int)
    out["target_close"] = out["Close"] * np.exp(out["target_log_return"])
    return out
