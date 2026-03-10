from __future__ import annotations

from pathlib import Path
from typing import Iterable
import contextlib
import io
import logging
import warnings

import pandas as pd
import yfinance as yf


def _import_pykrx_stock():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from pykrx import stock

        return stock
    except Exception:
        return None


def _to_yfinance_symbol(user_input: str) -> str:
    s = str(user_input).strip().upper()
    if not s:
        return s
    if "." in s:
        return s

    if s.isdigit() and len(s) == 6:
        stock = _import_pykrx_stock()
        if stock is not None:
            try:
                kospi = set(stock.get_market_ticker_list(market="KOSPI"))
                kosdaq = set(stock.get_market_ticker_list(market="KOSDAQ"))
                if s in kospi:
                    return f"{s}.KS"
                if s in kosdaq:
                    return f"{s}.KQ"
            except Exception:
                pass
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


def fetch_real_ohlcv(symbols: Iterable[str], start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        df = _safe_download_ohlcv(symbol, start=start, end=end)
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


def append_real_ohlcv_csv(path: str | Path, symbols: Iterable[str], start: str = "2020-01-01", end: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        new_df = fetch_real_ohlcv(symbols=symbols, start=start, end=end)
    except RuntimeError:
        return p
    if p.exists():
        base = pd.read_csv(p)
        if "Date" in base.columns:
            base["Date"] = pd.to_datetime(base["Date"])
        merged = pd.concat([base, new_df], axis=0, ignore_index=True)
    else:
        merged = new_df

    merged = merged.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    merged = merged.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    merged.to_csv(p, index=False)
    return p
