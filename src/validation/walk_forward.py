from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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


def walk_forward_validate(
    df: pd.DataFrame,
    feature_columns: List[str],
    cfg: TrainingConfig,
) -> List[FoldResult]:
    results: List[FoldResult] = []
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

        model = MultiHeadStockModel(random_state=cfg.random_state)
        model.fit(train_df, feature_columns, cfg.quantiles)
        pred = model.predict(valid_df)

        reg = regression_metrics(valid_df["target_log_return"], pred.predicted_return)
        cls = classification_metrics(valid_df["target_up"], pred.up_probability)
        result = {**reg, **cls}

        results.append(
            FoldResult(
                train_end=train_end_date,
                valid_start=valid_start_date,
                valid_end=valid_end_date,
                metrics=result,
            )
        )

    return results
