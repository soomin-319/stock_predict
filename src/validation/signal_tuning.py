from __future__ import annotations

from itertools import product

import pandas as pd


def tune_signal_weights(pred_df: pd.DataFrame) -> dict:
    """Grid-search simple signal weights by maximizing top-decile next-day return."""
    candidates = []
    for rw, uw in product([0.3, 0.45, 0.6], [0.15, 0.25, 0.35]):
        w_prob = 0.30
        w_rel = max(0.0, 1.0 - rw - w_prob)
        if w_rel < 0:
            continue
        score = rw * pred_df["norm_return"] + w_prob * pred_df["up_probability"] + w_rel * pred_df["rel_strength"] - uw * pred_df["uncertainty_score"]
        tmp = pred_df.copy()
        tmp["score"] = score
        n = max(1, int(len(tmp) * 0.1))
        top = tmp.nlargest(n, "score")
        perf = top["target_log_return"].mean()
        candidates.append((perf, rw, w_prob, w_rel, uw))

    best = max(candidates, key=lambda x: x[0])
    return {
        "best_top_decile_return": float(best[0]),
        "return_weight": best[1],
        "up_prob_weight": best[2],
        "rel_strength_weight": best[3],
        "uncertainty_penalty": best[4],
    }
