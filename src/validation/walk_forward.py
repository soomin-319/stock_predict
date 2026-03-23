from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass
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


FoldInput = Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]


def _iter_folds(df: pd.DataFrame, cfg: TrainingConfig):
    dates = sorted(df["Date"].dropna().unique())
    for start in range(cfg.min_train_size, len(dates) - cfg.test_size + 1, cfg.step_size):
        train_end_date = dates[start - 1]
        valid_end_idx = min(start + cfg.test_size - 1, len(dates) - 1)
        valid_end_date = dates[valid_end_idx]
        valid_start_date = dates[start]

        train_df = df[df["Date"] <= train_end_date]
        valid_df = df[(df["Date"] >= valid_start_date) & (df["Date"] <= valid_end_date)]
        if valid_df.empty:
            continue
        yield train_end_date, valid_start_date, valid_end_date, train_df, valid_df



def _run_fold(fold: FoldInput, feature_columns: List[str], cfg: TrainingConfig) -> tuple[FoldResult, pd.DataFrame]:
    train_end_date, valid_start_date, valid_end_date, train_df, valid_df = fold
    model = MultiHeadStockModel(random_state=cfg.random_state, n_jobs=cfg.model_n_jobs, use_gpu=cfg.use_gpu)
    model.fit(train_df, feature_columns, cfg.quantiles)
    pred = model.predict(valid_df)

    reg = regression_metrics(valid_df["target_log_return"], pred.predicted_return)
    cls = classification_metrics(valid_df["target_up"], pred.up_probability)
    result = FoldResult(
        train_end=train_end_date,
        valid_start=valid_start_date,
        valid_end=valid_end_date,
        metrics={**reg, **cls},
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
        "target_log_return_5d",
        "target_up_5d",
        "target_log_return_20d",
        "target_up_20d",
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
    for horizon, values in pred.horizon_predicted_return.items():
        oof[f"predicted_return_{horizon}d"] = values
    for horizon, values in pred.horizon_up_probability.items():
        oof[f"up_probability_{horizon}d"] = values
    return result, oof



def _execute_folds(folds: List[FoldInput], feature_columns: List[str], cfg: TrainingConfig) -> list[tuple[FoldResult, pd.DataFrame]]:
    if not folds:
        return []

    max_workers = max(1, int(getattr(cfg, "walk_forward_n_jobs", 1) or 1))
    if max_workers == 1 or len(folds) == 1:
        return [_run_fold(fold, feature_columns, cfg) for fold in folds]

    worker_count = min(max_workers, len(folds))
    run_fold = partial(_run_fold, feature_columns=feature_columns, cfg=cfg)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(run_fold, folds))



def walk_forward_validate(df: pd.DataFrame, feature_columns: List[str], cfg: TrainingConfig) -> List[FoldResult]:
    results, _ = walk_forward_validate_with_oof(df, feature_columns, cfg)
    return results



def walk_forward_oof_predictions(df: pd.DataFrame, feature_columns: List[str], cfg: TrainingConfig) -> pd.DataFrame:
    _, oof = walk_forward_validate_with_oof(df, feature_columns, cfg)
    return oof



def walk_forward_validate_with_oof(df: pd.DataFrame, feature_columns: List[str], cfg: TrainingConfig) -> tuple[List[FoldResult], pd.DataFrame]:
    executed = _execute_folds(list(_iter_folds(df, cfg)), feature_columns, cfg)
    if not executed:
        return [], pd.DataFrame()

    results = [result for result, _ in executed]
    oof = pd.concat([fold for _, fold in executed], axis=0, ignore_index=True)
    return results, oof
