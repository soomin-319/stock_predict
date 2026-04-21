from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable
import logging
import warnings
from functools import partial

import pandas as pd
import yfinance as yf

# Silence yfinance chatter once at import time. Mutating logger/warnings/stdout
# state per-call from ThreadPoolExecutor workers races on shared globals and
# can leave sys.stdout pointing at a detached StringIO, which looks like a
# deterministic hang because every subsequent print() output is swallowed.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")

_LOGGER = logging.getLogger(__name__)

def _to_yfinance_symbol(user_input: str) -> str:
    s = str(user_input).strip().upper()
    if not s:
        return s
    if "." in s:
        return s

    if s.isdigit() and len(s) == 6:
        return f"{s}.KS"
    return s



def normalize_user_symbols(symbol_inputs: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for item in symbol_inputs:
        chunks = [c for c in str(item).split(",") if c.strip()]
        parsed.extend(_to_yfinance_symbol(c) for c in chunks)
    return list(dict.fromkeys(s for s in parsed if s))



def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # yfinance 0.2.x occasionally emits duplicate column labels for single-ticker
    # downloads. Duplicates break `pd.concat(..., axis=0)` at `get_indexer` with
    # `InvalidIndexError: Reindexing only valid with uniquely valued Index`.
    duplicated = df.columns.duplicated(keep="first")
    if duplicated.any():
        dup_labels = df.columns[duplicated].tolist()
        _LOGGER.warning("yfinance returned duplicate columns %s; keeping first occurrence", dup_labels)
        df = df.loc[:, ~duplicated]
    return df



def _safe_download_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    try:
        return yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        _LOGGER.warning("yfinance download failed for %s: %s: %s", symbol, type(exc).__name__, exc)
        return pd.DataFrame()



def _fetch_single_symbol(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = _safe_download_ohlcv(symbol, start=start, end=end)
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_yf_columns(df).reset_index()
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    out = df[required].copy()
    out["Symbol"] = symbol
    return out



def fetch_real_ohlcv(symbols: Iterable[str], start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    normalized_symbols = list(dict.fromkeys(str(symbol) for symbol in symbols if str(symbol).strip()))
    if not normalized_symbols:
        raise RuntimeError("No symbols provided")

    max_workers = min(8, len(normalized_symbols))
    if max_workers <= 1:
        frames = [_fetch_single_symbol(symbol, start=start, end=end) for symbol in normalized_symbols]
    else:
        fetch_one = partial(_fetch_single_symbol, start=start, end=end)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            frames = list(executor.map(fetch_one, normalized_symbols))

    empty_symbols = [sym for sym, frame in zip(normalized_symbols, frames) if frame is None or frame.empty]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        raise RuntimeError(
            f"No data fetched from provider (yfinance). All {len(normalized_symbols)} symbols "
            f"returned empty. symbols={normalized_symbols!r}. Enable DEBUG logging on "
            f"'src.data.fetch_real_data' to see per-symbol errors."
        )
    if empty_symbols:
        _LOGGER.warning("yfinance returned empty frames for %d/%d symbols: %s",
                        len(empty_symbols), len(normalized_symbols), empty_symbols)

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    all_df = all_df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
    # yfinance can emit duplicate (Date, Symbol) rows; dedupe to match the
    # append path (`append_real_ohlcv_csv`) and prevent downstream pandas
    # operations (`reindex`/`align`) from raising `InvalidIndexError`.
    all_df = all_df.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    all_df = all_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return all_df



def _preserve_existing_optional_columns(fetched_df: pd.DataFrame, existing_df: pd.DataFrame | None) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        return fetched_df

    keys = ["Date", "Symbol"]
    existing = existing_df.copy()
    if "Date" in existing.columns:
        existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")

    extra_cols = [c for c in existing.columns if c not in {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}]
    if not extra_cols:
        return fetched_df

    preserved = existing[keys + extra_cols].drop_duplicates(subset=keys, keep="last")
    return fetched_df.merge(preserved, on=keys, how="left")



def save_real_ohlcv_csv(path: str | Path, symbols: Iterable[str], start: str = "2020-01-01", end: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_real_ohlcv(symbols=symbols, start=start, end=end)
    if p.exists():
        base = pd.read_csv(p)
        df = _preserve_existing_optional_columns(df, base)
    df.to_csv(p, index=False)
    return p



def append_real_ohlcv_csv(path: str | Path, symbols: Iterable[str], start: str = "2020-01-01", end: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        new_df = fetch_real_ohlcv(symbols=symbols, start=start, end=end)
    except RuntimeError:
        print("[경고] 추가 요청한 심볼 데이터를 가져오지 못해 기존 CSV를 유지합니다.")
        return p
    if p.exists():
        base = pd.read_csv(p)
        if "Date" in base.columns:
            base["Date"] = pd.to_datetime(base["Date"])
        new_df = _preserve_existing_optional_columns(new_df, base)
        merged = pd.concat([base, new_df], axis=0, ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    merged = merged.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    merged.to_csv(p, index=False)
    return p
