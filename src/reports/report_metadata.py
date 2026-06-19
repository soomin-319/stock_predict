from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
KRX_CALENDAR_EXPIRY_WARNING_DAYS = 60

ARTIFACT_SCHEMA_VERSIONS = {
    "result_simple": SCHEMA_VERSION,
    "result_detail": SCHEMA_VERSION,
    "result_news": SCHEMA_VERSION,
    "result_disclosure": SCHEMA_VERSION,
}

KOREA_MARKET_HOLIDAYS = {
    "2025-01-01",
    "2025-01-28",
    "2025-01-29",
    "2025-01-30",
    "2025-03-03",
    "2025-05-05",
    "2025-05-06",
    "2025-06-03",
    "2025-06-06",
    "2025-08-15",
    "2025-10-03",
    "2025-10-06",
    "2025-10-07",
    "2025-10-08",
    "2025-10-09",
    "2025-12-25",
    "2025-12-31",
    "2026-01-01",
    "2026-02-16",
    "2026-02-17",
    "2026-02-18",
    "2026-03-02",
    "2026-05-05",
    "2026-05-25",
    "2026-08-17",
    "2026-09-24",
    "2026-09-25",
    "2026-10-05",
    "2026-10-09",
    "2026-12-25",
    "2026-12-31",
}
KOREA_MARKET_HOLIDAY_COVERAGE_END = max(KOREA_MARKET_HOLIDAYS)


def _parse_iso_date(value: str | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10]).date()
    except ValueError:
        return None


def artifact_schema_version(schema_kind: str) -> str:
    return ARTIFACT_SCHEMA_VERSIONS.get(str(schema_kind), SCHEMA_VERSION)


def evaluate_krx_calendar_coverage(reference_date: str | datetime | None) -> dict[str, Any]:
    coverage_end = _parse_iso_date(KOREA_MARKET_HOLIDAY_COVERAGE_END)
    reference = _parse_iso_date(reference_date)
    if coverage_end is None or reference is None:
        return {
            "status": "unknown",
            "coverage_end": KOREA_MARKET_HOLIDAY_COVERAGE_END,
            "warnings": [],
        }

    if reference > coverage_end:
        warning = f"krx_calendar_coverage_expired:{KOREA_MARKET_HOLIDAY_COVERAGE_END}"
        return {
            "status": "expired",
            "coverage_end": KOREA_MARKET_HOLIDAY_COVERAGE_END,
            "warnings": [warning],
        }

    days_remaining = (coverage_end - reference).days
    if days_remaining <= KRX_CALENDAR_EXPIRY_WARNING_DAYS:
        warning = f"krx_calendar_coverage_near_expiry:{KOREA_MARKET_HOLIDAY_COVERAGE_END}"
        return {
            "status": "near_expiry",
            "coverage_end": KOREA_MARKET_HOLIDAY_COVERAGE_END,
            "warnings": [warning],
        }

    return {
        "status": "ok",
        "coverage_end": KOREA_MARKET_HOLIDAY_COVERAGE_END,
        "warnings": [],
    }


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return f"{current.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def next_krx_business_day(input_as_of_date: str | datetime | None) -> str | None:
    current = _parse_iso_date(input_as_of_date)
    if current is None:
        return None
    candidate = current + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate.isoformat() in KOREA_MARKET_HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate.isoformat()

def detect_git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def hash_config(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_report_metadata(
    *,
    run_id: str,
    environment: str,
    data_mode: str,
    input_as_of_date: str | None,
    prediction_for_date: str | None,
    context_as_of_date: str | None,
    config_payload: dict[str, Any],
    status: str = "pass",
    blocking_reasons: tuple[str, ...] | list[str] = (),
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[2]
    calendar_coverage = evaluate_krx_calendar_coverage(prediction_for_date or input_as_of_date)
    calendar_warnings = list(calendar_coverage["warnings"])
    merged_reasons = list(dict.fromkeys([*blocking_reasons, *calendar_warnings]))
    report_status = str(status)
    if report_status == "pass" and calendar_warnings:
        report_status = "warning"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "environment": str(environment),
        "data_mode": str(data_mode),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_as_of_date": input_as_of_date,
        "prediction_for_date": prediction_for_date,
        "context_as_of_date": context_as_of_date,
        "git_commit": detect_git_commit(root),
        "config_hash": hash_config(config_payload),
        "status": report_status,
        "blocking_reasons": merged_reasons,
        "calendar_status": calendar_coverage["status"],
        "calendar_coverage_end": calendar_coverage["coverage_end"],
        "calendar_warnings": calendar_warnings,
    }


__all__ = [
    "ARTIFACT_SCHEMA_VERSIONS",
    "KOREA_MARKET_HOLIDAY_COVERAGE_END",
    "KRX_CALENDAR_EXPIRY_WARNING_DAYS",
    "SCHEMA_VERSION",
    "artifact_schema_version",
    "build_report_metadata",
    "detect_git_commit",
    "evaluate_krx_calendar_coverage",
    "generate_run_id",
    "hash_config",
    "next_krx_business_day",
]
