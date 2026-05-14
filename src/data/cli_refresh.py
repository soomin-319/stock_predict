from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.universe import load_default_universe_symbols, load_universe_symbols


def fallback_symbols_from_input_or_default(input_csv: str, limit: int = 5) -> list[str]:
    """Return the repo-managed default fetch universe subset."""
    _ = input_csv
    symbols = load_default_universe_symbols()
    if limit <= 0:
        return symbols
    return symbols[:limit]


def resolve_fetch_symbols(
    real_symbols: list[str] | None,
    universe_csv: str | None,
    input_csv: str,
    *,
    universe_loader: Callable[[str], list[str]] = load_universe_symbols,
    fallback_loader: Callable[[str], list[str]] = fallback_symbols_from_input_or_default,
) -> list[str]:
    symbols = real_symbols
    if not symbols and universe_csv:
        try:
            symbols = universe_loader(universe_csv)
            print(f"Loaded symbols from universe CSV: {len(symbols)}")
        except Exception as exc:
            print(f"[경고] universe CSV 로드 실패: {exc}")

    if not symbols:
        symbols = fallback_loader(input_csv)
    return symbols


def resolve_incremental_fetch_start(input_csv: str, requested_start: str) -> str:
    path = Path(input_csv)
    if not path.exists():
        return requested_start
    try:
        base = pd.read_csv(path, usecols=["Date"])
        if base.empty:
            return requested_start
        parsed = pd.to_datetime(base["Date"], errors="coerce").dropna()
        if parsed.empty:
            return requested_start
        next_day = parsed.max() + timedelta(days=1)
        return max(requested_start, next_day.strftime("%Y-%m-%d"))
    except Exception:
        return requested_start
