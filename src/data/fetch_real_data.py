from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Iterable
import logging
import sys
import time
import warnings
from functools import partial

import pandas as pd

from src.data.krx_universe import get_provider_symbol_for_ticker

warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")

_LOGGER = logging.getLogger(__name__)
_YF = None
DEFAULT_REAL_START_DATE = "2020-01-01"
MAX_DOWNLOAD_ATTEMPTS = 3
_LAST_FETCH_COVERAGE: dict = {}
_sleep = time.sleep


def _get_yfinance():
    global _YF
    if _YF is None:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yfinance is required for live OHLCV downloads. "
                "Install market extras or run with --disable-external."
            ) from exc
        # Silence yfinance chatter once after the optional import. Mutating
        # logger/warnings/stdout state per-call from ThreadPoolExecutor workers
        # races on shared globals.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        _YF = yf
    return _YF

def _to_yfinance_symbol(user_input: str) -> str:
    s = str(user_input).strip().upper()
    if not s:
        return s
    if "." in s:
        return s

    if s.isdigit() and len(s) == 6:
        return get_provider_symbol_for_ticker(s) or f"{s}.KS"
    return s



def normalize_user_symbols(symbol_inputs: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for item in symbol_inputs:
        chunks = [c for c in str(item).split(",") if c.strip()]
        parsed.extend(_to_yfinance_symbol(c) for c in chunks)
    return list(dict.fromkeys(s for s in parsed if s))



_PRICE_COLS = frozenset({"Open", "High", "Low", "Close", "Adj Close", "Volume"})


def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        # yfinance 0.2.x returns a MultiIndex whose price-type level can be 0 or 1
        # depending on version and whether auto_adjust is used. Find the level that
        # actually contains price names instead of blindly taking level 0, which
        # would produce duplicate ticker-name labels or miss the right level.
        extracted = None
        for level in range(df.columns.nlevels):
            vals = df.columns.get_level_values(level)
            if _PRICE_COLS.intersection(vals):
                extracted = vals
                break
        df.columns = extracted if extracted is not None else df.columns.get_level_values(0)

    # yfinance occasionally emits duplicate column labels (e.g. when a Korean
    # ticker resolves to multiple internal listings). Duplicates break
    # `pd.concat(..., axis=0)` with `InvalidIndexError`.
    duplicated = df.columns.duplicated(keep="first")
    if duplicated.any():
        dup_labels = df.columns[duplicated].tolist()
        _LOGGER.warning(
            "yfinance returned duplicate columns %s; keeping first occurrence", dup_labels
        )
        df = df.loc[:, ~duplicated]
    return df



def _safe_download_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            yf = _get_yfinance()
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.attrs["download_attempts"] = attempt
                df.attrs["retry_count"] = attempt - 1
                return df
        except Exception as exc:
            last_exc = exc
        if attempt < MAX_DOWNLOAD_ATTEMPTS:
            _sleep(2 ** (attempt - 1))

    if last_exc is not None:
        _LOGGER.warning(
            "yfinance download failed for %s after %d attempts: %s: %s",
            symbol,
            MAX_DOWNLOAD_ATTEMPTS,
            type(last_exc).__name__,
            last_exc,
        )
    empty = pd.DataFrame()
    empty.attrs["download_attempts"] = MAX_DOWNLOAD_ATTEMPTS
    empty.attrs["retry_count"] = MAX_DOWNLOAD_ATTEMPTS - 1
    return empty


def _provider_symbol_candidates(symbol: str) -> list[str]:
    if symbol.endswith(".KS"):
        return [symbol, f"{symbol[:-3]}.KQ"]
    if symbol.endswith(".KQ"):
        return [symbol, f"{symbol[:-3]}.KS"]
    return [symbol]



def _fetch_single_symbol(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    used_symbol = symbol
    df = pd.DataFrame()
    total_attempts = 0
    retry_count = 0
    for candidate in _provider_symbol_candidates(symbol):
        df = _safe_download_ohlcv(candidate, start=start, end=end)
        total_attempts += int(df.attrs.get("download_attempts", 1))
        retry_count += int(df.attrs.get("retry_count", 0))
        if df is not None and not df.empty:
            used_symbol = candidate
            break
    if df is None or df.empty:
        empty = pd.DataFrame()
        empty.attrs.update(
            requested_symbol=symbol,
            resolved_symbol=None,
            total_attempts=total_attempts,
            retry_count=retry_count,
            fallback_used=False,
        )
        return empty

    df = _normalize_yf_columns(df).reset_index()
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    out = df[required].copy()
    out["Symbol"] = used_symbol
    out.attrs.update(
        requested_symbol=symbol,
        resolved_symbol=used_symbol,
        total_attempts=total_attempts,
        retry_count=retry_count,
        fallback_used=used_symbol != symbol,
    )
    return out



def get_last_fetch_coverage() -> dict:
    return deepcopy(_LAST_FETCH_COVERAGE)


def fetch_real_ohlcv(symbols: Iterable[str], start: str = DEFAULT_REAL_START_DATE, end: str | None = None) -> pd.DataFrame:
    global _LAST_FETCH_COVERAGE
    normalized_symbols = list(dict.fromkeys(str(symbol) for symbol in symbols if str(symbol).strip()))
    if not normalized_symbols:
        raise RuntimeError("No symbols provided")

    max_workers = min(8, len(normalized_symbols))
    _saved_stdout = sys.stdout
    try:
        if max_workers <= 1:
            frames = [_fetch_single_symbol(symbol, start=start, end=end) for symbol in normalized_symbols]
        else:
            fetch_one = partial(_fetch_single_symbol, start=start, end=end)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                frames = list(executor.map(fetch_one, normalized_symbols))
    finally:
        sys.stdout = _saved_stdout

    empty_symbols = [sym for sym, frame in zip(normalized_symbols, frames) if frame is None or frame.empty]
    details = []
    for sym, frame in zip(normalized_symbols, frames):
        attrs = {} if frame is None else frame.attrs
        details.append(
            {
                "symbol": sym,
                "status": "failed" if frame is None or frame.empty else "ok",
                "resolved_symbol": attrs.get("resolved_symbol"),
                "attempts": int(attrs.get("total_attempts", 1)),
                "retry_count": int(attrs.get("retry_count", 0)),
                "fallback_used": bool(attrs.get("fallback_used", False)),
            }
        )
    successful = len(normalized_symbols) - len(empty_symbols)
    _LAST_FETCH_COVERAGE = {
        "enabled": True,
        "requested": len(normalized_symbols),
        "successful": successful,
        "failed": len(empty_symbols),
        "success_ratio": successful / len(normalized_symbols),
        "failed_symbols": empty_symbols,
        "fallback_used": sum(1 for detail in details if detail["fallback_used"]),
        "retried_symbols": [
            detail["symbol"] for detail in details if detail["retry_count"] > 0
        ],
        "total_retry_count": sum(detail["retry_count"] for detail in details),
        "details": details,
    }
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



def save_real_ohlcv_csv(
    path: str | Path,
    symbols: Iterable[str],
    start: str = DEFAULT_REAL_START_DATE,
    end: str | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_real_ohlcv(symbols=symbols, start=start, end=end)
    if p.exists():
        base = pd.read_csv(p, encoding="utf-8-sig")
        df = _preserve_existing_optional_columns(df, base)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p



def append_real_ohlcv_csv(
    path: str | Path,
    symbols: Iterable[str],
    start: str = DEFAULT_REAL_START_DATE,
    end: str | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        new_df = fetch_real_ohlcv(symbols=symbols, start=start, end=end)
    except RuntimeError:
        print("[경고] 추가 요청한 심볼 데이터를 가져오지 못해 기존 CSV를 유지합니다.")
        return p
    if p.exists():
        base = pd.read_csv(p, encoding="utf-8-sig")
        if "Date" in base.columns:
            base["Date"] = pd.to_datetime(base["Date"])
        new_df = _preserve_existing_optional_columns(new_df, base)
        merged = pd.concat([base, new_df], axis=0, ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    merged = merged.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    merged.to_csv(p, index=False, encoding="utf-8-sig")
    return p
