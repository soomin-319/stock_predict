from __future__ import annotations

import numpy as np


def _roc_auc_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob)
    sorted_probs = y_prob[order]
    ranks = np.empty(len(y_prob), dtype=float)
    i = 0
    while i < len(y_prob):
        j = i + 1
        while j < len(y_prob) and sorted_probs[j] == sorted_probs[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else 0.0

    corr = 0.0
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))) if y_true.size else 0.0,
        "rmse": rmse,
        "corr": corr,
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_hat = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(np.mean(y_true == y_hat)) if y_true.size else 0.0,
        "roc_auc": float(_roc_auc_binary(y_true.astype(int), y_prob.astype(float))),
    }


def probability_calibration_metrics(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return {"brier": 0.0, "ece": 0.0}

    y_prob = np.clip(y_prob, 0.0, 1.0)
    brier = float(np.mean((y_prob - y_true) ** 2))

    bins = np.linspace(0.0, 1.0, max(2, n_bins) + 1)
    ece = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        if not np.any(m):
            continue
        conf = float(np.mean(y_prob[m]))
        acc = float(np.mean(y_true[m]))
        ece += float(np.mean(m)) * abs(acc - conf)

    return {"brier": brier, "ece": float(ece)}
