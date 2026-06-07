from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ContextPolicyResult:
    allowed: bool
    input_as_of_date: str | None
    context_as_of_date: str | None
    gap_days: int | None
    reason: str | None


def _iso(value: pd.Timestamp | object) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def evaluate_context_policy(
    input_as_of_date: object,
    context_as_of_date: object,
    max_gap_days: int = 3,
) -> ContextPolicyResult:
    input_date = pd.to_datetime(input_as_of_date, errors="coerce")
    context_date = pd.to_datetime(context_as_of_date, errors="coerce")
    if pd.isna(input_date) or pd.isna(context_date):
        return ContextPolicyResult(
            False,
            _iso(input_date),
            _iso(context_date),
            None,
            "missing_context_date",
        )
    gap_days = abs((context_date.normalize() - input_date.normalize()).days)
    reason = None if gap_days <= max_gap_days else "context_date_gap_exceeded"
    return ContextPolicyResult(
        reason is None,
        _iso(input_date),
        _iso(context_date),
        gap_days,
        reason,
    )


__all__ = ["ContextPolicyResult", "evaluate_context_policy"]
