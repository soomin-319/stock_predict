from __future__ import annotations

import pandas as pd
import yfinance as yf


def _safe_download(symbol: str, start: str, end: str | None) -> pd.Series:
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return pd.Series(dtype=float)


def add_external_market_features(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if out.empty or not symbols:
        return out

    start = out["Date"].min().strftime("%Y-%m-%d")
    end = (out["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base_dates = pd.Series(sorted(out["Date"].unique()), name="Date")
    ext = pd.DataFrame({"Date": pd.to_datetime(base_dates)})

    for sym in symbols:
        s = _safe_download(sym, start=start, end=end)
        if s.empty:
            continue
        col_base = sym.replace("^", "").replace("=", "_").replace("-", "_").lower()
        e = s.reset_index()
        e.columns = ["Date", f"{col_base}_close"]
        e["Date"] = pd.to_datetime(e["Date"])
        e = e.sort_values("Date")
        e[f"{col_base}_ret_1d"] = e[f"{col_base}_close"].pct_change()
        e[f"{col_base}_ret_5d"] = e[f"{col_base}_close"].pct_change(5)
        e[f"{col_base}_vol_20"] = e[f"{col_base}_ret_1d"].rolling(20).std()
        ext = ext.merge(e, on="Date", how="left")

    ext = ext.sort_values("Date").ffill().bfill()
    out = out.merge(ext, on="Date", how="left")
    return out
