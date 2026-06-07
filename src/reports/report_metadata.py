from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def generate_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return f"{current.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def detect_git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            text=True,
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
]
