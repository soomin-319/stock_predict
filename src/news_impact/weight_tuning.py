from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.news_impact.backtester import BacktestMetrics


WEIGHT_CANDIDATE_SCHEMA = "stock-news-impact.weight-candidates.v1"
WEIGHT_TUNING_RESULT_SCHEMA = "stock-news-impact.weight-tuning-result.v1"


@dataclass(frozen=True)
class WeightVariantEvaluation:
    variant_id: str
    description: str
    weights: dict[str, float]
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    bucket_metrics: dict[str, BacktestMetrics]
    scoring_version: str = "scoring.v1"
    train_period: tuple[str, str] = ("", "")
    test_period: tuple[str, str] = ("", "")


@dataclass(frozen=True)
class WeightCandidateConfig:
    variant_id: str
    description: str
    scoring_version: str
    weights: dict[str, float]
    train_period: tuple[str, str]
    test_period: tuple[str, str]

    def __post_init__(self) -> None:
        if not self.variant_id:
            raise ValueError("variant_id must be non-empty")
        if not self.scoring_version:
            raise ValueError("scoring_version must be non-empty")
        _validate_period_order(self.train_period, self.test_period)


@dataclass(frozen=True)
class WeightTuningResult:
    selected: WeightVariantEvaluation | None
    rejected: tuple[WeightVariantEvaluation, ...]
    rejection_reasons: dict[str, str]


def select_weight_variant(
    baseline_test_metrics: BacktestMetrics,
    candidates: list[WeightVariantEvaluation] | tuple[WeightVariantEvaluation, ...],
    min_bucket_rank_ic: float = 0.0,
    max_weight: float = 1.2,
) -> WeightTuningResult:
    accepted: list[WeightVariantEvaluation] = []
    rejected: list[WeightVariantEvaluation] = []
    rejection_reasons: dict[str, str] = {}
    for candidate in candidates:
        rejection_reason = _rejection_reason(
            baseline_test_metrics,
            candidate,
            min_bucket_rank_ic=min_bucket_rank_ic,
            max_weight=max_weight,
        )
        if rejection_reason is None:
            accepted.append(candidate)
        else:
            rejected.append(candidate)
            rejection_reasons[candidate.variant_id] = rejection_reason
    selected = max(accepted, key=lambda item: item.test_metrics.rank_ic, default=None)
    return WeightTuningResult(
        selected=selected,
        rejected=tuple(rejected),
        rejection_reasons=rejection_reasons,
    )


def load_weight_candidate_configs(path: str | Path) -> list[WeightCandidateConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weight candidate config root must be a JSON object")
    if payload.get("schema") != WEIGHT_CANDIDATE_SCHEMA:
        raise ValueError(f"schema must be {WEIGHT_CANDIDATE_SCHEMA}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    return [_candidate_config_from_dict(item) for item in candidates]


def write_weight_tuning_result(
    result: WeightTuningResult,
    output_path: str | Path,
    *,
    baseline_scoring_version: str,
) -> None:
    payload = serialize_weight_tuning_result(
        result,
        baseline_scoring_version=baseline_scoring_version,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def serialize_weight_tuning_result(
    result: WeightTuningResult,
    *,
    baseline_scoring_version: str,
) -> dict[str, Any]:
    selected = _evaluation_to_dict(result.selected) if result.selected is not None else None
    rejected = [_evaluation_to_dict(item) for item in result.rejected]
    selected_scoring_version = (
        result.selected.scoring_version if result.selected is not None else baseline_scoring_version
    )
    return {
        "schema": WEIGHT_TUNING_RESULT_SCHEMA,
        "selected": selected,
        "rejected": rejected,
        "rejection_reasons": dict(result.rejection_reasons),
        "audit": {
            "baseline_scoring_version": baseline_scoring_version,
            "selected_variant_id": result.selected.variant_id if result.selected else None,
            "selected_scoring_version": selected_scoring_version,
            "rejected_variant_ids": [item.variant_id for item in result.rejected],
            "adoption_status": "accepted" if result.selected is not None else "rejected_all",
        },
    }


def _rejection_reason(
    baseline_test_metrics: BacktestMetrics,
    candidate: WeightVariantEvaluation,
    min_bucket_rank_ic: float,
    max_weight: float,
) -> str | None:
    if any(weight < 0.0 or weight > max_weight for weight in candidate.weights.values()):
        return "weight_outside_policy_range"
    if candidate.test_metrics.rank_ic <= baseline_test_metrics.rank_ic:
        return "no_out_of_sample_rank_ic_improvement"
    if candidate.test_metrics.top_bottom_spread < baseline_test_metrics.top_bottom_spread:
        return "worse_out_of_sample_spread"
    if any(metrics.rank_ic < min_bucket_rank_ic for metrics in candidate.bucket_metrics.values()):
        return "unstable_bucket_rank_ic"
    return None


def _candidate_config_from_dict(item: object) -> WeightCandidateConfig:
    if not isinstance(item, dict):
        raise ValueError("candidate must be a JSON object")
    weights = item.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("weights must be a JSON object")
    return WeightCandidateConfig(
        variant_id=str(item.get("variant_id", "")),
        description=str(item.get("description", "")),
        scoring_version=str(item.get("scoring_version", "")),
        weights={str(key): float(value) for key, value in weights.items()},
        train_period=_period_from_dict(item.get("train_period"), "train_period"),
        test_period=_period_from_dict(item.get("test_period"), "test_period"),
    )


def _period_from_dict(value: object, field_name: str) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    start = str(value.get("start", ""))
    end = str(value.get("end", ""))
    _parse_date(start)
    _parse_date(end)
    if start > end:
        raise ValueError(f"{field_name} start must be on or before end")
    return (start, end)


def _validate_period_order(train_period: tuple[str, str], test_period: tuple[str, str]) -> None:
    train_end = _parse_date(train_period[1])
    test_start = _parse_date(test_period[0])
    if test_start <= train_end:
        raise ValueError("test_period must start after train_period")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _evaluation_to_dict(item: WeightVariantEvaluation) -> dict[str, Any]:
    return {
        "variant_id": item.variant_id,
        "description": item.description,
        "scoring_version": item.scoring_version,
        "weights": dict(item.weights),
        "train_period": _period_to_dict(item.train_period),
        "test_period": _period_to_dict(item.test_period),
        "train_metrics": asdict(item.train_metrics),
        "test_metrics": asdict(item.test_metrics),
        "bucket_metrics": {
            bucket: asdict(metrics)
            for bucket, metrics in item.bucket_metrics.items()
        },
    }


def _period_to_dict(value: tuple[str, str]) -> dict[str, str]:
    return {"start": value[0], "end": value[1]}
