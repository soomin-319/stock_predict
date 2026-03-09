from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def fetch_real_ohlcv(symbols: Iterable[str], start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:
            continue
        df = _normalize_yf_columns(df).reset_index()

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in required):
            continue

        out = df[required].copy()
        out["Symbol"] = symbol
        frames.append(out)

    if not frames:
        raise RuntimeError("No data fetched from yfinance")

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    all_df = all_df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
    all_df = all_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return all_df


def save_real_ohlcv_csv(path: str | Path, symbols: Iterable[str], start: str = "2020-01-01", end: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_real_ohlcv(symbols=symbols, start=start, end=end)
    df.to_csv(p, index=False)
    return p
