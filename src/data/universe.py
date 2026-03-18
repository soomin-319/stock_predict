from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_universe_symbols(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(p)
    if "Symbol" not in df.columns:
        raise ValueError("Universe CSV must include 'Symbol' column")

    symbols = set(df["Symbol"].dropna().astype(str).unique())
    if not symbols:
        raise ValueError("Universe symbol set is empty")
    return symbols


def filter_by_universe(df: pd.DataFrame, universe: Iterable[str]) -> pd.DataFrame:
    universe_set = set(universe)
    return df[df["Symbol"].astype(str).isin(universe_set)].copy()
