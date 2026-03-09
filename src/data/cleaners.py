from __future__ import annotations

import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"])

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    out = out[(out["Open"] > 0) & (out["High"] > 0) & (out["Low"] > 0) & (out["Close"] > 0)]
    out = out[out["Volume"] >= 0]
    out = out[out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)]
    out = out[out["Low"] <= out[["Open", "Close", "High"]].min(axis=1)]

    out = out.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    out = out.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return out
