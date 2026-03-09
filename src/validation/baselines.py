from __future__ import annotations

import pandas as pd

from src.validation.metrics import classification_metrics, regression_metrics


def evaluate_baselines(valid_df: pd.DataFrame) -> dict:
    y = valid_df["target_log_return"]
    y_up = valid_df["target_up"]

    zero_pred = pd.Series(0.0, index=valid_df.index)
    prev_pred = valid_df["log_return"].fillna(0.0)

    return {
        "baseline_zero": {
            **regression_metrics(y, zero_pred),
            **classification_metrics(y_up, pd.Series(0.5, index=valid_df.index)),
        },
        "baseline_prev_return": {
            **regression_metrics(y, prev_pred),
            **classification_metrics(y_up, (prev_pred > 0).astype(float)),
        },
    }
