from __future__ import annotations

import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if "Symbol" not in out.columns:
        out["Symbol"] = "UNKNOWN"
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"])

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    out = out[(out["Open"] > 0) & (out["High"] > 0) & (out["Low"] > 0) & (out["Close"] > 0)]
    out = out[out["Volume"] >= 0]
    out = out[out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)]
    out = out[out["Low"] <= out[["Open", "Close", "High"]].min(axis=1)]

    out["_input_order"] = range(len(out))
    out = out.sort_values(["Date", "Symbol", "Volume", "_input_order"])
    out = out.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    out = out.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    out["is_zero_volume"] = out["Volume"].eq(0)
    returns = out.groupby("Symbol", sort=False)["Close"].pct_change(fill_method=None)
    out["is_extreme_return"] = returns.abs().gt(0.40)
    out = out.drop(columns="_input_order")
    return out
