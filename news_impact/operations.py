from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROVIDER_RUNTIME_POLICY_SCHEMA = "stock-news-impact.provider-runtime-policy.v1"
EXTERNAL_REPORT_CONSUMER_SCHEMA = "stock-news-impact.external-report-consumer.v1"

_STORAGE_POLICIES = {"metadata_only", "summary_allowed", "raw_allowed"}
_ALLOWED_EXTERNAL_ARTIFACTS = {
    "report.json",
    "report.csv",
    "audit.json",
    "backtest_validation.json",
    "weight_tuning_result.json",
}
_REQUIRED_FORBIDDEN_ACTIONS = {
    "place_orders",
    "generate_buy_sell_recommendations",
}


@dataclass(frozen=True)
class ProviderRuntimePolicy:
    provider: str
    max_requests_per_minute: int
    backoff_seconds: tuple[float, ...]
    cache_retention_days: int
    storage_policy: str
    terms_reviewed_at: str

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider is required")
        if self.max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        if not self.backoff_seconds:
            raise ValueError("backoff_seconds is required")
        if any(value <= 0 for value in self.backoff_seconds):
            raise ValueError("backoff_seconds values must be positive")
        if tuple(sorted(self.backoff_seconds)) != self.backoff_seconds:
            raise ValueError("backoff_seconds must be non-decreasing")
        if self.cache_retention_days <= 0:
            raise ValueError("cache_retention_days must be positive")
        if self.storage_policy not in _STORAGE_POLICIES:
            allowed = ", ".join(sorted(_STORAGE_POLICIES))
            raise ValueError(f"storage_policy must be one of: {allowed}")
        if not self.terms_reviewed_at:
            raise ValueError("terms_reviewed_at is required")


@dataclass(frozen=True)
class ExternalReportConsumerContract:
    consumer_project: str
    transport: str
    mode: str
    allowed_artifacts: tuple[str, ...]
    automated_trading_enabled: bool
    forbidden_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.consumer_project:
            raise ValueError("consumer_project is required")
        if self.transport != "file_json":
            raise ValueError("transport must be file_json")
        if self.mode != "read_only_report_consumption":
            raise ValueError("mode must be read_only_report_consumption")
        if not self.allowed_artifacts:
            raise ValueError("allowed_artifacts is required")
        unknown = set(self.allowed_artifacts) - _ALLOWED_EXTERNAL_ARTIFACTS
        if unknown:
            raise ValueError(f"unknown allowed_artifacts: {', '.join(sorted(unknown))}")
        if self.automated_trading_enabled:
            raise ValueError("automated trading is forbidden")
        missing = _REQUIRED_FORBIDDEN_ACTIONS - set(self.forbidden_actions)
        if missing:
            raise ValueError(f"forbidden_actions missing: {', '.join(sorted(missing))}")


def load_provider_runtime_policy(path: str | Path) -> ProviderRuntimePolicy:
    payload = _read_json_object(path)
    _require_schema(payload, PROVIDER_RUNTIME_POLICY_SCHEMA)
    return ProviderRuntimePolicy(
        provider=str(payload.get("provider", "")),
        max_requests_per_minute=int(payload.get("max_requests_per_minute", 0)),
        backoff_seconds=tuple(float(value) for value in payload.get("backoff_seconds", ())),
        cache_retention_days=int(payload.get("cache_retention_days", 0)),
        storage_policy=str(payload.get("storage_policy", "")),
        terms_reviewed_at=str(payload.get("terms_reviewed_at", "")),
    )


def load_external_report_consumer_contract(path: str | Path) -> ExternalReportConsumerContract:
    payload = _read_json_object(path)
    _require_schema(payload, EXTERNAL_REPORT_CONSUMER_SCHEMA)
    return ExternalReportConsumerContract(
        consumer_project=str(payload.get("consumer_project", "")),
        transport=str(payload.get("transport", "")),
        mode=str(payload.get("mode", "")),
        allowed_artifacts=tuple(str(value) for value in payload.get("allowed_artifacts", ())),
        automated_trading_enabled=bool(payload.get("automated_trading_enabled", False)),
        forbidden_actions=tuple(str(value) for value in payload.get("forbidden_actions", ())),
    )


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("policy file must be a JSON object")
    return payload


def _require_schema(payload: dict[str, Any], expected_schema: str) -> None:
    if payload.get("schema") != expected_schema:
        raise ValueError(f"schema must be {expected_schema}")
