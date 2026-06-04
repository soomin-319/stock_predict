from __future__ import annotations

from dataclasses import dataclass

from src.news_impact.backtester import BacktestMetrics


@dataclass(frozen=True)
class PerformanceCriteria:
    min_overall_samples: int = 250
    min_bucket_samples: int = 30
    min_rank_ic: float = 0.0
    min_top_bottom_spread: float = 0.0


@dataclass(frozen=True)
class PerformanceValidationResult:
    passed: bool
    failed_checks: tuple[str, ...]


def evaluate_independent_performance(
    overall: BacktestMetrics,
    buckets: dict[str, BacktestMetrics],
    criteria: PerformanceCriteria | None = None,
) -> PerformanceValidationResult:
    active_criteria = criteria or PerformanceCriteria()
    failed_checks: list[str] = []
    if overall.sample_size < active_criteria.min_overall_samples:
        failed_checks.append("insufficient_overall_samples")
    if overall.rank_ic <= active_criteria.min_rank_ic:
        failed_checks.append("rank_ic_below_threshold")
    if overall.top_bottom_spread <= active_criteria.min_top_bottom_spread:
        failed_checks.append("top_bottom_spread_below_threshold")
    for bucket_name, metrics in buckets.items():
        if metrics.sample_size < active_criteria.min_bucket_samples:
            failed_checks.append(f"bucket_{bucket_name}_insufficient_samples")
    return PerformanceValidationResult(
        passed=not failed_checks,
        failed_checks=tuple(failed_checks),
    )
