from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.news_impact.schema import ImpactEvent


@dataclass(frozen=True)
class ScoringWeights:
    source_weight: float = 1.0
    relevance_weight: float = 1.0
    recency_weight: float = 1.0


@dataclass(frozen=True)
class ScoredEvent:
    event: ImpactEvent
    score: float
    raw_score: float
    risk_flags: tuple[str, ...]


@dataclass(frozen=True)
class ScoreSummary:
    news_disclosure_score: float
    positive_score: float
    negative_score: float
    uncertainty_score: float
    event_count: int
    llm_failed_count: int


def score_event(event: ImpactEvent, weights: ScoringWeights | None = None) -> ScoredEvent:
    del weights  # Kept for API compatibility; LLM impact_score is now canonical.
    score = _round_score(event.impact_score)
    return ScoredEvent(
        event=event,
        score=score,
        raw_score=score,
        risk_flags=tuple(event.risk_flags),
    )


def aggregate_scores(
    events: Iterable[ImpactEvent],
    weights: ScoringWeights | None = None,
    llm_failed_count: int = 0,
) -> ScoreSummary:
    del weights  # Kept for API compatibility; aggregate uses event.impact_score directly.
    event_list = list(events)
    cluster_scores = _cluster_scores(event_list)
    positive_score = _round_score(_clamp(sum(score for score in cluster_scores if score > 0), 0, 100))
    negative_score = _round_score(_clamp(sum(score for score in cluster_scores if score < 0), -100, 0))
    news_disclosure_score = _round_score(_clamp(positive_score + negative_score, -100, 100))
    conflicting_event_count = 1 if positive_score > 0 and negative_score < 0 else 0
    uncertainty_score = min(100.0, conflicting_event_count * 15.0 + llm_failed_count * 10.0)
    return ScoreSummary(
        news_disclosure_score=news_disclosure_score,
        positive_score=positive_score,
        negative_score=negative_score,
        uncertainty_score=uncertainty_score,
        event_count=len(event_list),
        llm_failed_count=llm_failed_count,
    )


def _cluster_scores(events: list[ImpactEvent]) -> list[float]:
    scores_by_cluster: dict[str, list[float]] = {}
    for event in events:
        scores_by_cluster.setdefault(event.cluster_id, []).append(_round_score(event.impact_score))

    cluster_scores: list[float] = []
    for scores in scores_by_cluster.values():
        positive = [score for score in scores if score > 0]
        negative = [score for score in scores if score < 0]
        if positive:
            ordered = sorted(positive, reverse=True)
            cluster_scores.append(_round_score(ordered[0] + 0.2 * sum(ordered[1:])))
        if negative:
            ordered = sorted(negative)
            cluster_scores.append(_round_score(ordered[0] + 0.2 * sum(ordered[1:])))
    return cluster_scores


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _round_score(value: float) -> float:
    return round(value, 6)
