from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def load_ohlcv_csv(path: str | Path, symbol: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if symbol is not None and "Symbol" not in df.columns:
        df["Symbol"] = symbol

    if "Symbol" not in df.columns:
        df["Symbol"] = "UNKNOWN"

    # Defend against pre-existing CSVs written before the fetcher deduped
    # (Date, Symbol) rows — duplicates here propagate into feature merges and
    # cause `InvalidIndexError` in downstream reindex/align calls.
    df = df.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
