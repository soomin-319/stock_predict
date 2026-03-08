from __future__ import annotations

import pandas as pd


def annotate_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    trend = out["close_to_ma_20"].fillna(0)
    vol = out["vol_20"].fillna(0)

    trend_state = pd.Series("sideways", index=out.index)
    trend_state[trend > 0.01] = "uptrend"
    trend_state[trend < -0.01] = "downtrend"

    vol_state = pd.Series("low_vol", index=out.index)
    vol_state[vol > vol.quantile(0.75)] = "high_vol"

    out["market_regime"] = trend_state + "_" + vol_state
    return out
