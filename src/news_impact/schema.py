from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


MarketSession = Literal["pre_market", "regular", "after_market", "holiday"]
StoragePolicy = Literal["metadata_only", "summary_allowed", "raw_allowed"]
PublisherDomainSource = Literal["originallink", "link", "manual"]
TimestampSource = Literal["naver_pubDate", "article_meta", "manual", "unknown"]
ImpactEventType = Literal[
    "earnings",
    "contract",
    "capital_raise",
    "legal",
    "policy",
    "macro",
    "sector",
    "product",
    "partnership",
    "other",
]
ImpactDirection = Literal["positive", "negative", "neutral", "mixed"]
TimeHorizon = Literal["intraday", "next_day", "short_term", "mid_term", "long_term"]
Expectedness = Literal["surprise", "partly_expected", "expected", "unknown"]


class SchemaValidationError(ValueError):
    """Raised when a canonical schema object violates docs/SCHEMA_SPEC.md."""


@dataclass(frozen=True)
class NewsItem:
    source: str
    title: str
    summary: str
    url: str
    original_url: str | None
    publisher_domain: str | None
    publisher_domain_source: PublisherDomainSource | None
    publisher_confidence: float
    published_at: datetime
    timestamp_source: TimestampSource
    collected_at: datetime
    signal_at: datetime
    market_session: MarketSession
    raw_text: str | None
    storage_policy: StoragePolicy
    quality_flags: list[str] | tuple[str, ...]

    def __post_init__(self) -> None:
        _require_allowed("market_session", self.market_session, _MARKET_SESSIONS)
        _require_allowed("storage_policy", self.storage_policy, _STORAGE_POLICIES)
        _require_allowed("timestamp_source", self.timestamp_source, _TIMESTAMP_SOURCES)
        if self.publisher_domain_source is not None:
            _require_allowed(
                "publisher_domain_source",
                self.publisher_domain_source,
                _PUBLISHER_DOMAIN_SOURCES,
            )
        _require_aware("published_at", self.published_at)
        _require_aware("collected_at", self.collected_at)
        _require_aware("signal_at", self.signal_at)
        _require_score("publisher_confidence", self.publisher_confidence)
        if self.signal_at != compute_signal_at(self.published_at, self.collected_at):
            raise SchemaValidationError("signal_at must equal max(published_at, collected_at)")
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class DisclosureItem:
    source: str
    receipt_no: str
    corp_code: str
    ticker: str
    disclosure_title: str
    disclosure_at: datetime
    collected_at: datetime
    signal_at: datetime
    is_correction: bool
    original_receipt_no: str | None
    url: str
    quality_flags: list[str] | tuple[str, ...]

    def __post_init__(self) -> None:
        _require_ticker(self.ticker)
        _require_aware("disclosure_at", self.disclosure_at)
        _require_aware("collected_at", self.collected_at)
        _require_aware("signal_at", self.signal_at)
        if self.signal_at != compute_signal_at(self.disclosure_at, self.collected_at):
            raise SchemaValidationError("signal_at must equal max(disclosure_at, collected_at)")
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class ImpactEvent:
    event_id: str
    cluster_id: str
    ticker: str
    company: str
    sector: str
    event_type: ImpactEventType
    impact_direction: ImpactDirection
    impact_strength: float
    impact_score: float
    time_horizon: TimeHorizon
    confidence: float
    expectedness: Expectedness
    novelty_score: float
    already_reflected_price_move: float
    reason: str
    why_may_be_wrong: str
    risk_flags: list[str] | tuple[str, ...]
    evidence_urls: list[str] | tuple[str, ...]

    def __post_init__(self) -> None:
        _require_ticker(self.ticker)
        _require_allowed("event_type", self.event_type, _IMPACT_EVENT_TYPES)
        _require_allowed("impact_direction", self.impact_direction, _IMPACT_DIRECTIONS)
        _require_allowed("time_horizon", self.time_horizon, _TIME_HORIZONS)
        _require_allowed("expectedness", self.expectedness, _EXPECTEDNESS)
        _require_score("impact_strength", self.impact_strength)
        _require_score_range("impact_score", self.impact_score, -100.0, 100.0)
        _require_score("confidence", self.confidence)
        _require_score("novelty_score", self.novelty_score)
        _require_score("already_reflected_price_move", self.already_reflected_price_move)
        object.__setattr__(self, "risk_flags", tuple(self.risk_flags))
        object.__setattr__(self, "evidence_urls", tuple(self.evidence_urls))

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class RunAudit:
    run_id: str
    run_started_at: datetime
    git_commit: str
    config_hash: str
    watchlist_hash: str
    company_master_snapshot_id: str
    data_snapshot_id: str
    llm_provider: str
    llm_model_requested: str
    llm_model_returned: str
    llm_temperature: float
    llm_prompt_hash: str
    scoring_version: str
    backtest_version: str

    def __post_init__(self) -> None:
        _require_aware("run_started_at", self.run_started_at)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def compute_signal_at(published_at: datetime, collected_at: datetime) -> datetime:
    _require_aware("published_at", published_at)
    _require_aware("collected_at", collected_at)
    if collected_at > published_at:
        return collected_at
    return published_at


def _require_aware(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SchemaValidationError(f"{field_name} must be timezone-aware")


def _require_score(field_name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise SchemaValidationError(f"{field_name} must be between 0.0 and 1.0")


def _require_score_range(field_name: str, value: float, minimum: float, maximum: float) -> None:
    if not minimum <= value <= maximum:
        raise SchemaValidationError(f"{field_name} must be between {minimum} and {maximum}")


def _require_ticker(value: str) -> None:
    if len(value) != 6:
        raise SchemaValidationError("ticker must be a six-character string")


def _require_allowed(field_name: str, value: str, allowed_values: set[str]) -> None:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise SchemaValidationError(f"{field_name} must be one of: {allowed}")


def _dataclass_to_dict(item: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in item.__dict__.items():
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


_MARKET_SESSIONS = {"pre_market", "regular", "after_market", "holiday"}
_STORAGE_POLICIES = {"metadata_only", "summary_allowed", "raw_allowed"}
_PUBLISHER_DOMAIN_SOURCES = {"originallink", "link", "manual"}
_TIMESTAMP_SOURCES = {"naver_pubDate", "article_meta", "manual", "unknown"}
_IMPACT_EVENT_TYPES = {
    "earnings",
    "contract",
    "capital_raise",
    "legal",
    "policy",
    "macro",
    "sector",
    "product",
    "partnership",
    "other",
}
_IMPACT_DIRECTIONS = {"positive", "negative", "neutral", "mixed"}
_TIME_HORIZONS = {"intraday", "next_day", "short_term", "mid_term", "long_term"}
_EXPECTEDNESS = {"surprise", "partly_expected", "expected", "unknown"}
