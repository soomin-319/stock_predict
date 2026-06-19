from __future__ import annotations

from itertools import product

import pandas as pd


DEFAULT_WEIGHTS = {
    "return_weight": 0.65,
    "up_prob_weight": 0.35,
    "uncertainty_penalty": 0.25,
}

RANK_IC_WEIGHT = 0.10
DOWNSIDE_PENALTY_WEIGHT = 0.25
OVERFIT_GAP_THRESHOLD = 0.15
DEFAULT_OBJECTIVE_TOLERANCE = 0.02


def _score_series(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return (
        weights["return_weight"] * df["norm_return"]
        + weights["up_prob_weight"] * df["up_probability"]
        - weights["uncertainty_penalty"] * df["uncertainty_score"]
    )


def _top_decile_return(df: pd.DataFrame, weights: dict[str, float]) -> float:
    if df.empty:
        return 0.0
    tmp = df.assign(score=_score_series(df, weights))
    n = max(1, int(len(tmp) * 0.1))
    return float(tmp.nlargest(n, "score")["target_log_return"].mean())


def _rank_ic(df: pd.DataFrame, weights: dict[str, float]) -> float:
    if df.empty or len(df) < 2:
        return 0.0
    corr = _score_series(df, weights).corr(df["target_log_return"], method="spearman")
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _selected_downside_penalty(df: pd.DataFrame, weights: dict[str, float]) -> float:
    if df.empty:
        return 0.0
    tmp = df.assign(score=_score_series(df, weights))
    n = max(1, int(len(tmp) * 0.1))
    selected_returns = tmp.nlargest(n, "score")["target_log_return"]
    downside = selected_returns[selected_returns < 0.0]
    if downside.empty:
        return 0.0
    return float(-downside.mean())


def _objective_score(df: pd.DataFrame, weights: dict[str, float]) -> float:
    top_return = _top_decile_return(df, weights)
    rank_ic = _rank_ic(df, weights)
    downside_penalty = _selected_downside_penalty(df, weights)
    return float(
        top_return
        + RANK_IC_WEIGHT * rank_ic
        - DOWNSIDE_PENALTY_WEIGHT * downside_penalty
    )


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
            "train_rank_ic": 0.0,
            "validation_rank_ic": 0.0,
            "train_objective_score": 0.0,
            "validation_objective_score": 0.0,
            "objective_generalization_gap": 0.0,
            "overfit_fallback_applied": False,
            **DEFAULT_WEIGHTS,
        }

    train_df, valid_df = _time_split(pred_df)
    candidates = []
    for rw, w_prob, uw in product(
        [0.3, 0.45, 0.6, 0.65],
        [0.20, 0.35, 0.50],
        [0.15, 0.25, 0.35],
    ):
        weights = {
            "return_weight": rw,
            "up_prob_weight": w_prob,
            "uncertainty_penalty": uw,
        }
        train_perf = _top_decile_return(train_df, weights)
        valid_perf = _top_decile_return(valid_df, weights)
        train_objective = _objective_score(train_df, weights)
        valid_objective = _objective_score(valid_df, weights)
        candidates.append(
            (
                valid_objective,
                valid_perf,
                train_objective,
                train_perf,
                _simplicity_key(weights),
                weights,
            )
        )

    (
        best_valid_objective,
        best_valid,
        best_train_objective,
        best_train,
        _simplicity,
        best_weights,
    ) = max(
        candidates,
        key=lambda x: (x[0], x[1], -x[4][0], -x[4][1]),
    )
    default_train = _top_decile_return(train_df, DEFAULT_WEIGHTS)
    default_valid = _top_decile_return(valid_df, DEFAULT_WEIGHTS)
    default_train_objective = _objective_score(train_df, DEFAULT_WEIGHTS)
    default_valid_objective = _objective_score(valid_df, DEFAULT_WEIGHTS)
    objective_gap = best_train_objective - best_valid_objective
    default_is_acceptable = (
        default_valid_objective
        >= best_valid_objective - DEFAULT_OBJECTIVE_TOLERANCE
    )
    fallback_applied = (
        objective_gap > OVERFIT_GAP_THRESHOLD and default_is_acceptable
    )
    final_weights = DEFAULT_WEIGHTS if fallback_applied else best_weights
    final_train = default_train if fallback_applied else best_train
    final_valid = default_valid if fallback_applied else best_valid
    final_train_objective = (
        default_train_objective if fallback_applied else best_train_objective
    )
    final_valid_objective = (
        default_valid_objective if fallback_applied else best_valid_objective
    )
    return {
        "best_top_decile_return": float(final_valid),
        "train_top_decile_return": float(final_train),
        "validation_top_decile_return": float(final_valid),
        "top_decile_generalization_gap": float(final_train - final_valid),
        "train_rank_ic": _rank_ic(train_df, final_weights),
        "validation_rank_ic": _rank_ic(valid_df, final_weights),
        "train_objective_score": float(final_train_objective),
        "validation_objective_score": float(final_valid_objective),
        "objective_generalization_gap": float(
            final_train_objective - final_valid_objective
        ),
        "overfit_fallback_applied": fallback_applied,
        **final_weights,
    }
