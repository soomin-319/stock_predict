from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

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


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return f"{current.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def next_krx_business_day(input_as_of_date: str | datetime | None) -> str | None:
    if input_as_of_date is None:
        return None
    if isinstance(input_as_of_date, datetime):
        current = input_as_of_date.date()
    else:
        raw = str(input_as_of_date).strip()
        if not raw:
            return None
        normalized = raw[:10]
        try:
            current = datetime.fromisoformat(normalized).date()
        except ValueError:
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
        "status": str(status),
        "blocking_reasons": list(blocking_reasons),
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_report_metadata",
    "detect_git_commit",
    "generate_run_id",
    "hash_config",
    "next_krx_business_day",
]
