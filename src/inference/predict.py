from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import SignalConfig
from src.models.lgbm_heads import MultiHeadPrediction


def normalize_series(values: pd.Series) -> pd.Series:
    std = values.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def build_prediction_frame(
    latest_df: pd.DataFrame,
    pred: MultiHeadPrediction,
    signal_cfg: SignalConfig,
) -> pd.DataFrame:
    out = latest_df[["Date", "Symbol", "Close", "market_regime"]].copy()
    out["predicted_log_return"] = pred.predicted_return
    out["predicted_return"] = np.expm1(out["predicted_log_return"]) * 100.0
    out["predicted_close"] = out["Close"] * np.exp(out["predicted_log_return"])
    out["up_probability"] = pred.up_probability
    out["uncertainty_width"] = pred.quantile_high - pred.quantile_low
    out["uncertainty_band"] = pd.Series(pred.quantile_low, index=out.index).astype(str) + " ~ " + pd.Series(pred.quantile_high, index=out.index).astype(str)

    out["rel_strength"] = normalize_series(out["predicted_log_return"])
    out["uncertainty_score"] = normalize_series(out["uncertainty_width"]).clip(lower=0)
    out["norm_return"] = normalize_series(out["predicted_log_return"])

    out["signal_score"] = (
        signal_cfg.return_weight * out["norm_return"]
        + signal_cfg.up_prob_weight * out["up_probability"]
        + signal_cfg.rel_strength_weight * out["rel_strength"]
        - signal_cfg.uncertainty_penalty * out["uncertainty_score"]
    )
    out["signal_label"] = pd.cut(
        out["signal_score"],
        bins=[-np.inf, 0.25, 0.45, 0.55, 0.75, np.inf],
        labels=["strong_negative", "weak_negative", "neutral", "weak_positive", "strong_positive"],
    )
    return out
