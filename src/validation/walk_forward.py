from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, List, Tuple

import pandas as pd

from src.config.settings import TrainingConfig
from src.models.lgbm_heads import MultiHeadStockModel
from src.validation.metrics import classification_metrics, regression_metrics


@dataclass
class FoldResult:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    metrics: Dict[str, float]
    train_start: pd.Timestamp | None = None
    fold_id: int | None = None


FoldInput = Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]
NumberedFoldInput = Tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oof: pd.DataFrame
    oof_diagnostics: dict


def _iter_folds(df: pd.DataFrame, cfg: TrainingConfig):
    dates = sorted(df["Date"].dropna().unique())
    # Purge gap prevents rows whose forward target overlaps the validation
    # window from entering training (look-ahead bias on multi-horizon targets).
    purge_gap = max(0, int(getattr(cfg, "purge_gap_days", 0) or 0))
    embargo = max(0, int(getattr(cfg, "embargo_days", 0) or 0))
    for start in range(cfg.min_train_size, len(dates) - cfg.test_size + 1, cfg.step_size):
        train_cutoff_idx = start - 1 - purge_gap
        if train_cutoff_idx < 0:
            continue
        train_end_date = dates[train_cutoff_idx]
        valid_start_idx = start + embargo
        if valid_start_idx >= len(dates):
            continue
        valid_end_idx = min(valid_start_idx + cfg.test_size - 1, len(dates) - 1)
        valid_end_date = dates[valid_end_idx]
        valid_start_date = dates[valid_start_idx]

        train_df = df[df["Date"] <= train_end_date]
        lookback = max(0, int(getattr(cfg, "walk_forward_lookback_days", 0) or 0))
        if lookback > 0:
            train_dates = sorted(pd.Series(train_df["Date"].dropna().unique()))
            cutoff_dates = train_dates[-lookback:]
            train_df = train_df[train_df["Date"].isin(cutoff_dates)]
        valid_df = df[(df["Date"] >= valid_start_date) & (df["Date"] <= valid_end_date)]
        if valid_df.empty:
            continue
        yield train_end_date, valid_start_date, valid_end_date, train_df, valid_df



def _run_fold(fold: NumberedFoldInput, feature_columns: List[str], cfg: TrainingConfig) -> tuple[FoldResult, pd.DataFrame]:
    fold_id, train_end_date, valid_start_date, valid_end_date, train_df, valid_df = fold
    train_start_date = pd.to_datetime(train_df["Date"]).min()
    model = MultiHeadStockModel(
        random_state=cfg.random_state,
        n_jobs=cfg.model_n_jobs,
        use_gpu=cfg.use_gpu,
        head_n_jobs=getattr(cfg, "model_head_n_jobs", 1),
    )
    model.fit(train_df, feature_columns, cfg.quantiles)
    pred = model.predict(valid_df)

    reg = regression_metrics(valid_df["target_log_return"], pred.predicted_return)
    cls = classification_metrics(valid_df["target_up"], pred.up_probability)
    result = FoldResult(
        train_end=train_end_date,
        valid_start=valid_start_date,
        valid_end=valid_end_date,
        metrics={**reg, **cls},
        train_start=train_start_date,
        fold_id=fold_id,
    )

    optional_cols = [
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
    keep_cols = ["Date", "Symbol", "Close", "market_regime", "target_log_return", "target_up"] + [
        c for c in optional_cols if c in valid_df.columns
    ]
    oof = valid_df[keep_cols].copy()
    oof["predicted_return"] = pred.predicted_return
    oof["up_probability"] = pred.up_probability
    oof["quantile_low"] = pred.quantile_low
    oof["quantile_mid"] = pred.quantile_mid
    oof["quantile_high"] = pred.quantile_high
    oof["fold_id"] = fold_id
    oof["train_start"] = train_start_date
    oof["train_end"] = train_end_date
    oof["valid_start"] = valid_start_date
    oof["valid_end"] = valid_end_date
    return result, oof



def _execute_folds(folds: List[FoldInput], feature_columns: List[str], cfg: TrainingConfig) -> list[tuple[FoldResult, pd.DataFrame]]:
    if not folds:
        return []

    n_jobs = int(getattr(cfg, "walk_forward_n_jobs", 1) or 1)
    cpu_count = os.cpu_count() or 1
    worker_count = min(cpu_count if n_jobs == -1 else max(1, n_jobs), len(folds))

    numbered_folds = [(fold_id, *fold) for fold_id, fold in enumerate(folds)]
    if worker_count == 1:
        return [_run_fold(fold, feature_columns, cfg) for fold in numbered_folds]

    # 여러 프로세스가 동시에 실행될 때 모델 내부 스레드 수를 1로 제한해
    # CPU 과점유(worker_count × model_n_jobs)를 방지한다.
    parallel_cfg = replace(cfg, model_n_jobs=1, model_head_n_jobs=1)
    run_fold = partial(_run_fold, feature_columns=feature_columns, cfg=parallel_cfg)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(run_fold, numbered_folds))


def aggregate_oof_predictions(raw_oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    diagnostics = {
        "policy_version": "date_symbol_mean_v1",
        "raw_row_count": int(len(raw_oof)),
        "unique_row_count": 0,
        "duplicate_row_count": 0,
        "duplicate_ratio": 0.0,
    }
    if raw_oof.empty:
        return raw_oof.copy(), diagnostics

    key = ["Date", "Symbol"]
    target_cols = [c for c in ("target_log_return", "target_up") if c in raw_oof.columns]
    for column in target_cols:
        if raw_oof.groupby(key, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting OOF target values for Date + Symbol: {column}")

    prediction_cols = [
        c
        for c in ("predicted_return", "up_probability", "quantile_low", "quantile_mid", "quantile_high")
        if c in raw_oof.columns
    ]
    provenance_cols = [c for c in ("fold_id", "train_start", "train_end", "valid_start", "valid_end") if c in raw_oof.columns]
    stable_cols = [c for c in raw_oof.columns if c not in {*key, *prediction_cols, *provenance_cols}]
    for column in stable_cols:
        if raw_oof.groupby(key, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting OOF stable values for Date + Symbol: {column}")

    rows = []
    for values, group in raw_oof.groupby(key, sort=True, dropna=False):
        row = dict(zip(key, values))
        row.update({column: group[column].iloc[0] for column in stable_cols})
        row.update({column: float(pd.to_numeric(group[column], errors="coerce").mean()) for column in prediction_cols})
        row["oof_prediction_count"] = int(len(group))
        row["fold_ids"] = (
            sorted(pd.to_numeric(group["fold_id"], errors="coerce").dropna().astype(int).unique().tolist())
            if "fold_id" in group
            else []
        )
        for column in ("train_start", "train_end", "valid_start", "valid_end"):
            if column in group:
                row[f"{column}_values"] = sorted(pd.to_datetime(group[column]).dropna().unique().tolist())
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    raw_count = int(len(raw_oof))
    unique_count = int(len(aggregated))
    diagnostics.update(
        {
            "unique_row_count": unique_count,
            "duplicate_row_count": raw_count - unique_count,
            "duplicate_ratio": float((raw_count - unique_count) / raw_count),
        }
    )
    return aggregated, diagnostics



def walk_forward_validate(df: pd.DataFrame, feature_columns: List[str], cfg: TrainingConfig) -> List[FoldResult]:
    results, _ = walk_forward_validate_with_oof(df, feature_columns, cfg)
    return results



def walk_forward_oof_predictions(df: pd.DataFrame, feature_columns: List[str], cfg: TrainingConfig) -> pd.DataFrame:
    _, oof = walk_forward_validate_with_oof(df, feature_columns, cfg)
    return oof



def walk_forward_validate_with_oof(df: pd.DataFrame, feature_columns: List[str], cfg: TrainingConfig) -> tuple[List[FoldResult], pd.DataFrame]:
    result = walk_forward_validate_result(df, feature_columns, cfg)
    return result.folds, result.oof


def walk_forward_validate_result(df: pd.DataFrame, feature_columns: List[str], cfg: TrainingConfig) -> WalkForwardResult:
    executed = _execute_folds(list(_iter_folds(df, cfg)), feature_columns, cfg)
    if not executed:
        empty, diagnostics = aggregate_oof_predictions(pd.DataFrame())
        return WalkForwardResult([], empty, diagnostics)

    results = [result for result, _ in executed]
    raw_oof = pd.concat([fold for _, fold in executed], axis=0, ignore_index=True)
    oof, diagnostics = aggregate_oof_predictions(raw_oof)
    return WalkForwardResult(results, oof, diagnostics)
