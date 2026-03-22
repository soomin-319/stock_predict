from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_UNIVERSE_CSV = Path(__file__).resolve().parents[2] / "data" / "default_universe_kospi50_kosdaq50.csv"


def load_universe_symbols_list(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(p)
    if "Symbol" not in df.columns:
        raise ValueError("Universe CSV must include 'Symbol' column")

    symbols = [str(symbol).strip() for symbol in df["Symbol"].dropna().astype(str).tolist() if str(symbol).strip()]
    ordered_unique_symbols = list(dict.fromkeys(symbols))
    if not ordered_unique_symbols:
        raise ValueError("Universe symbol set is empty")
    return ordered_unique_symbols


def load_universe_symbols(path: str | Path) -> set[str]:
    return set(load_universe_symbols_list(path))


def load_default_universe_symbols() -> list[str]:
    return load_universe_symbols_list(DEFAULT_UNIVERSE_CSV)


def filter_by_universe(df: pd.DataFrame, universe: Iterable[str]) -> pd.DataFrame:
    universe_set = set(universe)
    return df[df["Symbol"].astype(str).isin(universe_set)].copy()
