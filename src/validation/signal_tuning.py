from __future__ import annotations

from itertools import product

import pandas as pd


DEFAULT_WEIGHTS = {
    "return_weight": 0.45,
    "up_prob_weight": 0.35,
    "rel_strength_weight": 0.20,
    "uncertainty_penalty": 0.25,
}


def _top_decile_return(df: pd.DataFrame, weights: dict[str, float]) -> float:
    if df.empty:
        return 0.0
    score = (
        weights["return_weight"] * df["norm_return"]
        + weights["up_prob_weight"] * df["up_probability"]
        + weights["rel_strength_weight"] * df["rel_strength"]
        - weights["uncertainty_penalty"] * df["uncertainty_score"]
    )
    tmp = df.assign(score=score)
    n = max(1, int(len(tmp) * 0.1))
    return float(tmp.nlargest(n, "score")["target_log_return"].mean())


def _time_split(df: pd.DataFrame, ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    ordered = df.sort_values("Date") if "Date" in df.columns else df.copy()
    if "Date" in ordered.columns:
        dates = sorted(ordered["Date"].dropna().unique())
        split_idx = min(max(1, int(len(dates) * ratio)), max(1, len(dates) - 1))
        split_date = dates[split_idx - 1]
        train = ordered[ordered["Date"] <= split_date]
        valid = ordered[ordered["Date"] > split_date]
    else:
        split_idx = min(max(1, int(len(ordered) * ratio)), max(1, len(ordered) - 1))
        train = ordered.iloc[:split_idx]
        valid = ordered.iloc[split_idx:]
    if valid.empty:
        return ordered, ordered
    return train, valid


def _simplicity_key(weights: dict[str, float]) -> tuple[float, float]:
    distance_from_default = sum(abs(weights[key] - DEFAULT_WEIGHTS[key]) for key in DEFAULT_WEIGHTS)
    total_weight = (
        weights["return_weight"]
        + weights["up_prob_weight"]
        + weights["rel_strength_weight"]
        + weights["uncertainty_penalty"]
    )
    return distance_from_default, total_weight


def tune_signal_weights(pred_df: pd.DataFrame) -> dict:
    """Tune signal weights on an internal time split to reduce in-sample overfit."""
    if pred_df.empty:
        return {
            "best_top_decile_return": 0.0,
            "train_top_decile_return": 0.0,
            "validation_top_decile_return": 0.0,
            "top_decile_generalization_gap": 0.0,
            **DEFAULT_WEIGHTS,
        }

    train_df, valid_df = _time_split(pred_df)
    candidates = []
    for rw, w_prob, uw in product(
        [0.3, 0.45, 0.6],
        [0.20, 0.35, 0.50],
        [0.15, 0.25, 0.35],
    ):
        w_rel = round(max(0.0, 1.0 - rw - w_prob), 10)
        if rw + w_prob > 1.0:
            continue
        weights = {
            "return_weight": rw,
            "up_prob_weight": w_prob,
            "rel_strength_weight": w_rel,
            "uncertainty_penalty": uw,
        }
        train_perf = _top_decile_return(train_df, weights)
        valid_perf = _top_decile_return(valid_df, weights)
        candidates.append((valid_perf, train_perf, _simplicity_key(weights), weights))

    best_valid, best_train, _simplicity, best_weights = max(
        candidates,
        key=lambda x: (x[0], -x[2][0], -x[2][1]),
    )
    return {
        "best_top_decile_return": float(best_valid),
        "train_top_decile_return": float(best_train),
        "validation_top_decile_return": float(best_valid),
        "top_decile_generalization_gap": float(best_train - best_valid),
        **best_weights,
    }
