from __future__ import annotations

import pandas as pd


def _expanding_volatility_threshold(out: pd.DataFrame, vol: pd.Series) -> pd.Series:
    if "Symbol" not in out.columns or "Date" not in out.columns:
        return vol.expanding(min_periods=1).quantile(0.75).reindex(out.index)

    order = out[["Symbol", "Date"]].copy()
    order["_original_index"] = out.index
    order["_vol_20"] = vol
    order = order.sort_values(["Symbol", "Date", "_original_index"], kind="mergesort")
    thresholds = (
        order.groupby("Symbol", sort=False)["_vol_20"]
        .expanding(min_periods=1)
        .quantile(0.75)
        .reset_index(level=0, drop=True)
    )
    return thresholds.reindex(out.index)


def annotate_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    trend = out["close_to_ma_20"].fillna(0)
    vol = out["vol_20"].fillna(0)

    trend_state = pd.Series("sideways", index=out.index)
    trend_state[trend > 0.01] = "uptrend"
    trend_state[trend < -0.01] = "downtrend"

    vol_threshold = _expanding_volatility_threshold(out, vol)
    vol_state = pd.Series("low_vol", index=out.index)
    vol_state[vol > vol_threshold] = "high_vol"

    out["market_regime"] = trend_state + "_" + vol_state
    return out
