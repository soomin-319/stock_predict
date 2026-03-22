from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable
from datetime import datetime
import contextlib
import io
import logging
import warnings
from functools import lru_cache, partial

import pandas as pd
import yfinance as yf

from src.data.pykrx_support import import_pykrx_stock


_import_pykrx_stock = import_pykrx_stock


@lru_cache(maxsize=1)
def _market_ticker_sets() -> tuple[set[str], set[str]]:
    stock = _import_pykrx_stock()
    if stock is None:
        return set(), set()
    try:
        kospi = set(stock.get_market_ticker_list(market="KOSPI"))
        kosdaq = set(stock.get_market_ticker_list(market="KOSDAQ"))
        return kospi, kosdaq
    except Exception:
        return set(), set()



def _to_yfinance_symbol(user_input: str) -> str:
    s = str(user_input).strip().upper()
    if not s:
        return s
    if "." in s:
        return s

    if s.isdigit() and len(s) == 6:
        kospi, kosdaq = _market_ticker_sets()
        if s in kospi:
            return f"{s}.KS"
        if s in kosdaq:
            return f"{s}.KQ"
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
    return df



def _safe_download_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    prev_disabled = yf_logger.disabled
    try:
        yf_logger.disabled = True
        yf_logger.setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return yf.download(
                    symbol,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
    except Exception:
        return pd.DataFrame()
    finally:
        yf_logger.disabled = prev_disabled
        yf_logger.setLevel(prev_level)



def _fetch_krx_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    if not symbol.endswith((".KS", ".KQ")):
        return pd.DataFrame()
    stock = _import_pykrx_stock()
    if stock is None:
        return pd.DataFrame()

    ticker = symbol.split(".")[0]
    start_s = pd.to_datetime(start).strftime("%Y%m%d")
    end_s = pd.to_datetime(end).strftime("%Y%m%d") if end else datetime.now().strftime("%Y%m%d")

    try:
        df = stock.get_market_ohlcv_by_date(start_s, end_s, ticker)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    renamed = df.rename(
        columns={
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
        }
    )
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in renamed.columns for c in required):
        return pd.DataFrame()

    out = renamed[required].copy().reset_index()
    first_col = out.columns[0]
    out = out.rename(columns={first_col: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    out["Symbol"] = symbol
    return out[["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]]



def _fetch_single_symbol(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = _safe_download_ohlcv(symbol, start=start, end=end)
    if df is None or df.empty:
        df = _fetch_krx_ohlcv(symbol, start=start, end=end)
        if df.empty:
            return pd.DataFrame()
        return df[["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]]

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

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        raise RuntimeError("No data fetched from providers (yfinance/pykrx)")

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    all_df = all_df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
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
