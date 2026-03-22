from __future__ import annotations

import contextlib
import io
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
import yfinance as yf



def _safe_download(symbol: str, start: str, end: str | None) -> pd.Series:
    """Download close series without leaking provider noise to console."""
    try:
        yf_logger = logging.getLogger("yfinance")
        prev_level = yf_logger.level
        prev_disabled = yf_logger.disabled
        yf_logger.disabled = True
        yf_logger.setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
        yf_logger.disabled = prev_disabled
        yf_logger.setLevel(prev_level)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return pd.Series(dtype=float)
    finally:
        try:
            yf_logger.disabled = prev_disabled
            yf_logger.setLevel(prev_level)
        except Exception:
            pass



def _symbol_candidates(symbol: str) -> tuple[str, list[str]]:
    fallback = {
        "^SOX": ("sox", ["^SOX", "SOXX"]),
        "^VIX": ("vix", ["^VIX", "VIXY"]),
        "^KS11": ("ks11", ["^KS11", "EWY"]),
        "^KQ11": ("kq11", ["^KQ11", "KORU"]),
        "^IXIC": ("ixic", ["^IXIC", "QQQ"]),
        "NQ=F": ("nq_f", ["NQ=F", "QQQ"]),
        "^GSPC": ("gspc", ["^GSPC", "SPY"]),
        "KRW=X": ("krw_x", ["KRW=X", "USDKRW=X"]),
        "^TNX": ("tnx", ["^TNX", "IEF"]),
    }
    if symbol in fallback:
        return fallback[symbol]

    alias = symbol.replace("^", "").replace("=", "_").replace("-", "_").lower()
    return alias, [symbol]



def _download_external_symbol(sym: str, start: str, end: str) -> tuple[str, dict, pd.DataFrame | None]:
    col_base, candidates = _symbol_candidates(sym)

    s = pd.Series(dtype=float)
    used_candidate = None
    for candidate in candidates:
        s = _safe_download(candidate, start=start, end=end)
        if not s.empty:
            used_candidate = candidate
            break

    if s.empty:
        return sym, {"symbol": sym, "status": "failed", "used": None}, None

    e = s.reset_index()
    e.columns = ["Date", f"{col_base}_close"]
    e["Date"] = pd.to_datetime(e["Date"])
    e = e.sort_values("Date")
    e[f"{col_base}_ret_1d"] = e[f"{col_base}_close"].pct_change()
    e[f"{col_base}_ret_5d"] = e[f"{col_base}_close"].pct_change(5)
    e[f"{col_base}_vol_20"] = e[f"{col_base}_ret_1d"].rolling(20).std()
    return sym, {"symbol": sym, "status": "ok", "used": used_candidate}, e



def add_external_market_features_with_coverage(df: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    coverage = {
        "requested": len(symbols),
        "successful": 0,
        "failed": 0,
        "fallback_used": 0,
        "details": [],
    }

    if out.empty or not symbols:
        return out, coverage

    start = out["Date"].min().strftime("%Y-%m-%d")
    end = (out["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base_dates = pd.Series(sorted(out["Date"].unique()), name="Date")
    ext = pd.DataFrame({"Date": pd.to_datetime(base_dates)})

    worker_count = min(4, len(symbols))
    if worker_count <= 1:
        downloads = [_download_external_symbol(sym, start, end) for sym in symbols]
    else:
        download_one = partial(_download_external_symbol, start=start, end=end)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            downloads = list(executor.map(download_one, symbols))

    for sym, detail, frame in downloads:
        if detail["status"] == "failed" or frame is None:
            coverage["failed"] += 1
            coverage["details"].append(detail)
            continue

        coverage["successful"] += 1
        if detail["used"] != sym:
            coverage["fallback_used"] += 1
        coverage["details"].append(detail)
        ext = ext.merge(frame, on="Date", how="left")

    if coverage["successful"] == 0:
        return out, coverage

    ext = ext.sort_values("Date").ffill().bfill()
    out = out.merge(ext, on="Date", how="left")
    return out, coverage



def add_external_market_features(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    out, _ = add_external_market_features_with_coverage(df, symbols)
    return out
