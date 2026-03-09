from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    corr = 0.0
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "corr": corr,
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_hat = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_hat),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
    }
