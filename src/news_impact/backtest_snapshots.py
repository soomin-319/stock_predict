from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.news_impact.backtester import (
    BacktestObservation,
    BacktestSignal,
    PriceBar,
    calculate_metrics,
    match_signal_returns,
    summarize_by_bucket,
)
from src.news_impact.performance_validation import (
    PerformanceCriteria,
    evaluate_independent_performance,
)


PRICE_BAR_SCHEMA = "stock-news-impact.price-bars.v1"
SIGNAL_SCHEMA = "stock-news-impact.backtest-signals.v1"
VALIDATION_SCHEMA = "stock-news-impact.backtest-validation.v1"


@dataclass(frozen=True)
class PriceBarSnapshot:
    schema: str
    provider: str
    source: str
    adjustment: str
    rows: list[PriceBar]

    def __post_init__(self) -> None:
        if self.schema != PRICE_BAR_SCHEMA:
            raise ValueError(f"price bar snapshot schema must be {PRICE_BAR_SCHEMA}")
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if self.source not in {"krx", "data.go.kr", "fixture"}:
            raise ValueError("source must be one of: krx, data.go.kr, fixture")
        if self.adjustment not in {"raw", "split_adjusted", "corporate_action_adjusted"}:
            raise ValueError("adjustment must be raw, split_adjusted, or corporate_action_adjusted")


@dataclass(frozen=True)
class SignalSnapshot:
    schema: str
    scoring_version: str
    rows: list[BacktestSignal]

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_SCHEMA:
            raise ValueError(f"signal snapshot schema must be {SIGNAL_SCHEMA}")
        if not self.scoring_version:
            raise ValueError("scoring_version must be non-empty")


def load_price_bar_snapshot(path: str | Path) -> PriceBarSnapshot:
    payload = _read_json_object(path)
    rows = [_price_bar_from_dict(item) for item in _require_list(payload, "rows")]
    return PriceBarSnapshot(
        schema=str(payload.get("schema", "")),
        provider=str(payload.get("provider", "")),
        source=str(payload.get("source", "")),
        adjustment=str(payload.get("adjustment", "")),
        rows=rows,
    )


def load_signal_snapshot(path: str | Path) -> SignalSnapshot:
    payload = _read_json_object(path)
    rows = [_signal_from_dict(item) for item in _require_list(payload, "rows")]
    return SignalSnapshot(
        schema=str(payload.get("schema", "")),
        scoring_version=str(payload.get("scoring_version", "")),
        rows=rows,
    )


def build_observations_from_snapshots(
    signals: SignalSnapshot,
    prices: PriceBarSnapshot,
    *,
    round_trip_cost_bps: float = 26.0,
    include_missing: bool = False,
) -> list[BacktestObservation]:
    return match_signal_returns(
        signals=signals.rows,
        prices=prices.rows,
        round_trip_cost_bps=round_trip_cost_bps,
        include_missing=include_missing,
    )


def build_validation_report(
    observations: list[BacktestObservation] | tuple[BacktestObservation, ...],
    *,
    criteria: PerformanceCriteria | None = None,
    basket_size: int = 5,
) -> dict[str, Any]:
    overall = calculate_metrics(observations, basket_size=basket_size)
    by_day = _summarize_by_day(observations)
    by_sector = summarize_by_bucket(observations, "sector")
    by_market_cap = summarize_by_bucket(observations, "market_cap_bucket")
    gate_buckets = {**by_day, **by_sector, **by_market_cap}
    validation = evaluate_independent_performance(overall, gate_buckets, criteria=criteria)
    return {
        "schema": VALIDATION_SCHEMA,
        "overall": _metrics_to_dict(overall),
        "by_day": _metrics_mapping_to_dict(by_day),
        "by_sector": _metrics_mapping_to_dict(by_sector),
        "by_market_cap": _metrics_mapping_to_dict(by_market_cap),
        "validation": {
            "passed": validation.passed,
            "failed_checks": list(validation.failed_checks),
        },
    }


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summarize_by_day(
    observations: list[BacktestObservation] | tuple[BacktestObservation, ...],
) -> dict[str, Any]:
    grouped: dict[str, list[BacktestObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.entry_day.isoformat(), []).append(observation)
    return {key: calculate_metrics(values, basket_size=1) for key, values in grouped.items()}


def _metrics_mapping_to_dict(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: _metrics_to_dict(value) for key, value in metrics.items()}


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    return asdict(metrics)


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot root must be a JSON object")
    return payload


def _require_list(payload: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must contain JSON objects")
    return value


def _price_bar_from_dict(item: dict[str, Any]) -> PriceBar:
    return PriceBar(
        ticker=str(item["ticker"]),
        trading_day=_parse_date(str(item["trading_day"])),
        open=float(item["open"]),
        close=float(item["close"]),
        market_return=float(item.get("market_return", 0.0)),
        sector_return=float(item.get("sector_return", 0.0)),
    )


def _signal_from_dict(item: dict[str, Any]) -> BacktestSignal:
    return BacktestSignal(
        ticker=str(item["ticker"]),
        score=float(item["score"]),
        signal_day=_parse_date(str(item["signal_day"])),
        effective_trading_day=_parse_date(str(item["effective_trading_day"])),
        market_session=str(item.get("market_session", "regular")),
        tradeability_status=str(item.get("tradeability_status", "tradable")),
        sector=str(item.get("sector", "unknown")),
        market_cap_bucket=str(item.get("market_cap_bucket", "unknown")),
        event_type=str(item.get("event_type", "unknown")),
    )


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)
