from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.lgbm_heads import MultiHeadPrediction
from src.validation.metrics import probability_calibration_metrics


@dataclass
class TemporalOOFSplit:
    tune: pd.DataFrame
    eval: pd.DataFrame
    status: str
    reason: str | None
    diagnostics: dict

    def __iter__(self):
        yield self.tune
        yield self.eval


def split_oof_for_tuning_and_eval(
    scored_oof: pd.DataFrame,
    tune_ratio: float = 0.7,
    *,
    min_tune_dates: int = 5,
    min_eval_dates: int = 5,
) -> TemporalOOFSplit:
    normalized = pd.to_datetime(scored_oof["Date"], errors="coerce").dt.normalize()
    dates = sorted(normalized.dropna().unique())
    if len(dates) < min_tune_dates + min_eval_dates:
        tune_dates = set(dates[: min(len(dates), min_tune_dates)])
        tune_df = scored_oof[normalized.isin(tune_dates)].copy()
        eval_df = scored_oof.iloc[0:0].copy()
        return TemporalOOFSplit(
            tune=tune_df,
            eval=eval_df,
            status="insufficient_data",
            reason="insufficient_unique_oof_dates",
            diagnostics={
                "unique_date_count": len(dates),
                "tune_date_count": len(tune_dates),
                "eval_date_count": 0,
                "tune_row_count": int(len(tune_df)),
                "eval_row_count": 0,
            },
        )

    split_idx = max(min_tune_dates, min(len(dates) - min_eval_dates, int(len(dates) * tune_ratio)))
    tune_dates = set(dates[:split_idx])
    eval_dates = set(dates[split_idx:])

    tune_df = scored_oof[normalized.isin(tune_dates)].copy()
    eval_df = scored_oof[normalized.isin(eval_dates)].copy()
    return TemporalOOFSplit(
        tune=tune_df,
        eval=eval_df,
        status="ok",
        reason=None,
        diagnostics={
            "unique_date_count": len(dates),
            "tune_date_count": len(tune_dates),
            "eval_date_count": len(eval_dates),
            "tune_row_count": int(len(tune_df)),
            "eval_row_count": int(len(eval_df)),
        },
    )


def prediction_from_oof_df(oof: pd.DataFrame) -> MultiHeadPrediction:
    return MultiHeadPrediction(
        predicted_return=oof["predicted_return"].values,
        up_probability=oof["up_probability"].values,
        quantile_low=oof["quantile_low"].values,
        quantile_mid=oof["quantile_mid"].values,
        quantile_high=oof["quantile_high"].values,
    )


def compute_oof_diagnostics(scored_oof: pd.DataFrame) -> dict:
    if scored_oof.empty:
        return {}

    req = {"target_log_return", "rel_strength", "norm_return", "predicted_log_return", "uncertainty_score", "uncertainty_width"}
    if not req.issubset(set(scored_oof.columns)):
        return {}

    df = scored_oof[list(req)].copy().dropna()
    if df.empty:
        return {}

    actual_up = (df["target_log_return"] > 0).astype(int)

    rel_dir_acc = float(((df["rel_strength"] > 0).astype(int) == actual_up).mean())
    norm_dir_acc = float(((df["norm_return"] > 0.5).astype(int) == actual_up).mean())
    pred_dir_acc = float(((df["predicted_log_return"] > 0).astype(int) == actual_up).mean())

    abs_error = (df["predicted_log_return"] - df["target_log_return"]).abs()

    return {
        "direction_accuracy": {
            "predicted_log_return": pred_dir_acc,
            "rel_strength": rel_dir_acc,
            "norm_return": norm_dir_acc,
        },
        "uncertainty_diagnostics": {
            "corr_uncertainty_vs_abs_error": float(df["uncertainty_width"].corr(abs_error)),
            "corr_uncertainty_score_vs_abs_error": float(df["uncertainty_score"].corr(abs_error)),
            "uncertainty_score_zero_ratio": float((df["uncertainty_score"] == 0).mean()),
            "uncertainty_score_mean": float(df["uncertainty_score"].mean()),
        },
    }


@dataclass
class UpProbabilityCalibrator:
    model: object | None
    status: str
    reason: str | None

    def transform(self, probabilities: pd.Series | pd.Index | list | tuple) -> pd.Series:
        raw = pd.Series(probabilities, dtype=float).clip(0.0, 1.0)
        if self.model is None:
            return raw
        calibrated = pd.Series(self.model.predict(raw.values), dtype=float).clip(0.0, 1.0)
        if raw.round(6).nunique() >= 4 and calibrated.round(6).nunique() <= 2:
            return (0.3 * calibrated + 0.7 * raw).clip(0.0, 1.0)
        return calibrated


def fit_up_probability_calibrator(tune_oof: pd.DataFrame) -> UpProbabilityCalibrator:
    required = {"up_probability", "target_log_return"}
    if tune_oof.empty or not required.issubset(tune_oof.columns):
        return UpProbabilityCalibrator(None, "identity", "missing_or_empty_tune_oof")
    cal = tune_oof[["up_probability", "target_log_return"]].copy().dropna()
    if cal.empty or cal["up_probability"].nunique() < 3:
        return UpProbabilityCalibrator(None, "identity", "insufficient_probability_diversity")
    y = (cal["target_log_return"] > 0).astype(int)
    if y.nunique() < 2:
        return UpProbabilityCalibrator(None, "identity", "insufficient_label_diversity")
    try:
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(cal["up_probability"].astype(float).values, y.values)
        return UpProbabilityCalibrator(model, "fitted", None)
    except Exception as exc:
        return UpProbabilityCalibrator(None, "identity", f"fit_failed:{type(exc).__name__}")


def calibration_split_metrics(
    tune_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    calibrator: UpProbabilityCalibrator,
) -> dict:
    def metrics(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {**probability_calibration_metrics([], []), "sample_count": 0}
        probabilities = calibrator.transform(frame["up_probability"])
        labels = (frame["target_log_return"] > 0).astype(int)
        return {
            **probability_calibration_metrics(labels.values, probabilities.values),
            "sample_count": int(len(frame)),
        }

    return {
        "fit": {"status": calibrator.status, "reason": calibrator.reason},
        "tune": metrics(tune_df),
        "eval": metrics(eval_df),
    }


def calibrate_up_probability(oof_df: pd.DataFrame, up_probs: pd.Series | pd.Index | list | tuple) -> pd.Series:
    return fit_up_probability_calibrator(oof_df).transform(up_probs)
