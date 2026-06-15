from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series]:
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    k = 100 * (close - ll) / (hh - ll + 1e-9)
    d = k.rolling(3).mean()
    return k, d


def compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


def finite_or_nan(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def compute_technical_indicator_block(
    frame: pd.DataFrame,
    *,
    rsi_period: int,
    stochastic_period: int,
    cci_period: int,
) -> pd.DataFrame:
    macd, macd_signal, macd_hist = compute_macd(frame["Close"])
    stoch_k, stoch_d = compute_stochastic(
        frame["High"],
        frame["Low"],
        frame["Close"],
        stochastic_period,
    )
    obv = compute_obv(frame["Close"], frame["Volume"])
    block = pd.DataFrame(
        {
            "rsi_14": compute_rsi(frame["Close"], rsi_period),
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "atr_14": compute_atr(frame["High"], frame["Low"], frame["Close"]),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "cci_20": compute_cci(frame["High"], frame["Low"], frame["Close"], cci_period),
            "obv": obv,
            "obv_change_5d": finite_or_nan(obv.pct_change(5)),
        },
        index=frame.index,
    )
    return block.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
