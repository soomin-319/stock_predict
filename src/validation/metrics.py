from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0,
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5):
    y_hat = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
    }
