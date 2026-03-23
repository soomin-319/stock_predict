from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.settings import SignalConfig
from src.domain.signal_policy import build_prediction_policy_frame, confidence_label, vectorized_event_signal_boost
from src.inference.predict import build_prediction_frame, signal_label_series
from src.models.lgbm_heads import MultiHeadPrediction


OPTIONAL_PREDICTION_COLUMNS = [
    "target_log_return",
    "vol_ratio_20",
    "value_traded",
    "turnover_rank_daily",
    "foreign_net_buy",
    "institution_net_buy",
    "nq_f_ret_1d",
    "rsi_14",
    "near_52w_high_flag",
    "breakout_52w_flag",
    "leader_confirmation_flag",
    "ks11_ret_1d",
    "market_type",
]


@dataclass(slots=True)
class PredictionFrameContext:
    external_coverage_ratio: float = 1.0
    investor_coverage_ratio: float = 1.0
    min_liquidity_threshold: float = 0.0


def _copy_optional_prediction_columns(scored: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    for column in OPTIONAL_PREDICTION_COLUMNS:
        if column in source.columns:
            scored[column] = source[column].values
    return scored


def _market_headwind_score(values: pd.Series | None, index: pd.Index) -> pd.Series:
    numeric = pd.to_numeric(values if values is not None else pd.Series(0.0, index=index), errors="coerce").fillna(0.0)
    return (numeric < -0.01).astype(float) * -1.0


def build_scored_prediction_frame(
    latest_df: pd.DataFrame,
    pred: MultiHeadPrediction,
    signal_cfg: SignalConfig,
    context: PredictionFrameContext,
    *,
    calibration_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scored = build_prediction_frame(latest_df, pred, signal_cfg)
    scored = _copy_optional_prediction_columns(scored, latest_df)
    scored["external_coverage_ratio"] = float(context.external_coverage_ratio)
    scored["investor_coverage_ratio"] = float(context.investor_coverage_ratio)
    scored["min_liquidity_threshold"] = float(context.min_liquidity_threshold)
    nq_source = latest_df["nq_f_ret_1d"] if "nq_f_ret_1d" in latest_df.columns else None
    scored["market_headwind_score"] = _market_headwind_score(nq_source, scored.index)
    if calibration_source is not None:
        for column in [
            "turnover_rank_daily",
            "foreign_net_buy",
            "institution_net_buy",
            "nq_f_ret_1d",
            "rsi_14",
            "near_52w_high_flag",
            "breakout_52w_flag",
            "leader_confirmation_flag",
        ]:
            if column in calibration_source.columns and column not in scored.columns:
                scored[column] = calibration_source[column].values
    scored = vectorized_event_signal_boost(scored)
    if "signal_score" in scored.columns:
        scored["signal_label"] = signal_label_series(scored["signal_score"])
    return scored


def build_symbol_history_accuracy(scored_oof: pd.DataFrame) -> pd.DataFrame:
    if not {"Symbol", "target_log_return", "predicted_log_return"}.issubset(set(scored_oof.columns)):
        return pd.DataFrame(columns=["Symbol", "history_direction_accuracy"])
    tmp = scored_oof[["Symbol", "target_log_return", "predicted_log_return"]].copy()
    tmp["history_direction_accuracy"] = (
        (tmp["target_log_return"] > 0).astype(int) == (tmp["predicted_log_return"] > 0).astype(int)
    ).astype(float)
    return tmp.groupby("Symbol", as_index=False)["history_direction_accuracy"].mean()


def finalize_latest_prediction_frame(pred_df: pd.DataFrame, symbol_name_map: dict[str, str]) -> pd.DataFrame:
    out = pred_df.copy()
    out["symbol_name"] = out["Symbol"].astype(str).map(symbol_name_map).fillna(out["Symbol"].astype(str))
    out["confidence_score"] = (1 - out["uncertainty_score"].fillna(1)).clip(lower=0, upper=1)
    out["confidence_label"] = out["confidence_score"].map(confidence_label)
    out = build_prediction_policy_frame(out)
    return out


__all__ = [
    "OPTIONAL_PREDICTION_COLUMNS",
    "PredictionFrameContext",
    "build_scored_prediction_frame",
    "build_symbol_history_accuracy",
    "finalize_latest_prediction_frame",
]
