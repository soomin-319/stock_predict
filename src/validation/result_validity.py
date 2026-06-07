from __future__ import annotations

from typing import Any


def evaluate_backtest_validity(
    backtest: dict[str, Any],
    tradable_prediction_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    days = int(backtest.get("days") or 0)
    halted_days = int(backtest.get("halted_days") or 0)
    avg_selected_count = float(backtest.get("avg_selected_count") or 0.0)
    if int(tradable_prediction_count) <= 0:
        reasons.append("tradable_prediction_count_zero")
    if days <= 0:
        reasons.append("no_evaluation_days")
    if days > 0 and halted_days >= days:
        reasons.append("all_days_halted")
    if avg_selected_count <= 0:
        reasons.append("avg_selected_count_zero")
    return {
        "backtest_valid": not reasons,
        "blocking_reasons": reasons,
    }


__all__ = ["evaluate_backtest_validity"]
