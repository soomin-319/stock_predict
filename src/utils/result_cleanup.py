from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetentionPolicy:
    successful_run_count: int = 10
    successful_run_days: int = 30
    failed_run_days: int = 30
    runtime_log_days: int = 14


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def prune_registry(
    data: dict[str, Any],
    *,
    timestamp_field: str,
    ttl: timedelta,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    result = {}
    for key, value in data.items():
        timestamp = parse_utc(value.get(timestamp_field)) if isinstance(value, dict) else None
        if timestamp is not None and current - timestamp <= ttl:
            result[key] = value
    return result


def _inside(target: Path, root: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return target.resolve() != root.resolve()
    except ValueError:
        return False


def _remove(target: Path, allowed_root: Path) -> str:
    if not _inside(target, allowed_root):
        raise ValueError(f"refusing cleanup outside allowed root: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return str(target)


def _modified_at(path: Path) -> datetime:
    marker = path / "manifest.json" if path.is_dir() and (path / "manifest.json").exists() else path
    return datetime.fromtimestamp(marker.stat().st_mtime, tz=timezone.utc)


def cleanup_runs(runs_root: Path, policy: RetentionPolicy, now: datetime) -> list[str]:
    if not runs_root.exists():
        return []
    successful: list[Path] = []
    failed: list[Path] = []
    for run in (path for path in runs_root.iterdir() if path.is_dir()):
        try:
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            manifest = {"status": "fail"}
        (failed if manifest.get("status") == "fail" else successful).append(run)
    successful.sort(key=_modified_at, reverse=True)
    removed = []
    for index, run in enumerate(successful):
        age = now - _modified_at(run)
        if index >= policy.successful_run_count or age > timedelta(days=policy.successful_run_days):
            removed.append(_remove(run, runs_root))
    for run in failed:
        if now - _modified_at(run) > timedelta(days=policy.failed_run_days):
            removed.append(_remove(run, runs_root))
    return removed


def cleanup_logs(log_root: Path, policy: RetentionPolicy, now: datetime) -> list[str]:
    if not log_root.exists():
        return []
    removed = []
    for path in (item for item in log_root.iterdir() if item.is_file()):
        if now - _modified_at(path) > timedelta(days=policy.runtime_log_days):
            removed.append(_remove(path, log_root))
    return removed


def cleanup_test_artifacts(test_root: Path, policy: RetentionPolicy, now: datetime) -> list[str]:
    if not test_root.exists():
        return []
    removed = []
    for artifact in test_root.iterdir():
        file_times = [_modified_at(path) for path in artifact.rglob("*") if path.is_file()] if artifact.is_dir() else []
        modified_at = max(file_times) if file_times else _modified_at(artifact)
        if now - modified_at > timedelta(days=policy.failed_run_days):
            removed.append(_remove(artifact, test_root))
    return removed


def cleanup_result_artifacts(
    result_root: Path,
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    result_root = Path(result_root)
    removed = []
    removed.extend(cleanup_runs(result_root / "runs", policy, current))
    removed.extend(cleanup_test_artifacts(result_root / "test", policy, current))
    removed.extend(cleanup_logs(result_root / "runtime" / "logs", policy, current))
    return {"removed": removed, "removed_count": len(removed)}


__all__ = [
    "RetentionPolicy",
    "cleanup_result_artifacts",
    "cleanup_logs",
    "cleanup_runs",
    "cleanup_test_artifacts",
    "parse_utc",
    "prune_registry",
]
