from __future__ import annotations

import contextlib
import io
import warnings

import pandas as pd
import yfinance as yf


def _safe_download(symbol: str, start: str, end: str | None) -> pd.Series:
    """Download close series without leaking provider noise to console."""
    try:
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


def _symbol_candidates(symbol: str) -> tuple[str, list[str]]:
    fallback = {
        "^SOX": ("sox", ["^SOX", "SOXX"]),
        "^VIX": ("vix", ["^VIX", "VIXY"]),
        "^KS11": ("ks11", ["^KS11", "EWY"]),
        "^KQ11": ("kq11", ["^KQ11", "KORU"]),
        "^IXIC": ("ixic", ["^IXIC", "QQQ"]),
        "^GSPC": ("gspc", ["^GSPC", "SPY"]),
        "KRW=X": ("krw_x", ["KRW=X", "USDKRW=X"]),
        "^TNX": ("tnx", ["^TNX", "IEF"]),
    }
    if symbol in fallback:
        return fallback[symbol]

    alias = symbol.replace("^", "").replace("=", "_").replace("-", "_").lower()
    return alias, [symbol]


def add_external_market_features(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if out.empty or not symbols:
        return out

    start = out["Date"].min().strftime("%Y-%m-%d")
    end = (out["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base_dates = pd.Series(sorted(out["Date"].unique()), name="Date")
    ext = pd.DataFrame({"Date": pd.to_datetime(base_dates)})

    success_count = 0
    for sym in symbols:
        col_base, candidates = _symbol_candidates(sym)

        s = pd.Series(dtype=float)
        for candidate in candidates:
            s = _safe_download(candidate, start=start, end=end)
            if not s.empty:
                break

        if s.empty:
            continue

        success_count += 1
        e = s.reset_index()
        e.columns = ["Date", f"{col_base}_close"]
        e["Date"] = pd.to_datetime(e["Date"])
        e = e.sort_values("Date")
        e[f"{col_base}_ret_1d"] = e[f"{col_base}_close"].pct_change()
        e[f"{col_base}_ret_5d"] = e[f"{col_base}_close"].pct_change(5)
        e[f"{col_base}_vol_20"] = e[f"{col_base}_ret_1d"].rolling(20).std()
        ext = ext.merge(e, on="Date", how="left")

    if success_count == 0:
        return out

    ext = ext.sort_values("Date").ffill().bfill()
    out = out.merge(ext, on="Date", how="left")
    return out
