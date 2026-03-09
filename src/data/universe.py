from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config.settings import UniverseConfig


def _fallback_universe(cfg: UniverseConfig) -> set[str]:
    kospi = {f"KOSPI_{i:03d}" for i in range(1, cfg.default_kospi_count + 1)}
    kosdaq = {f"KOSDAQ_{i:03d}" for i in range(1, cfg.default_kosdaq_count + 1)}
    return kospi | kosdaq


def load_universe_symbols(path: str | None, cfg: UniverseConfig) -> set[str]:
    if path is None:
        return _fallback_universe(cfg)

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
