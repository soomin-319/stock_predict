import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.utils.result_cleanup import RetentionPolicy, cleanup_result_artifacts, prune_registry


def _dated_file(path: Path, now: datetime, age_days: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    timestamp = (now - timedelta(days=age_days)).timestamp()
    os.utime(path, (timestamp, timestamp))


def _run(path: Path, now: datetime, age_days: int, status: str) -> None:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    timestamp = (now - timedelta(days=age_days)).timestamp()
    os.utime(path / "manifest.json", (timestamp, timestamp))
    os.utime(path, (timestamp, timestamp))


def test_cleanup_removes_only_expired_allowed_artifacts(tmp_path: Path):
    now = datetime(2026, 6, 7, tzinfo=timezone.utc)
    result = tmp_path / "result"
    latest = result / "latest"
    latest.mkdir(parents=True)
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("keep", encoding="utf-8")
    _run(result / "runs" / "old-success", now, 40, "pass")
    _run(result / "runs" / "current-success", now, 1, "pass")
    _run(result / "runs" / "old-fail", now, 40, "fail")
    _dated_file(result / "runtime" / "logs" / "old.log", now, 20)
    _dated_file(result / "runtime" / "logs" / "current.log", now, 1)
    (result / "runtime" / "chatbot_jobs.json").write_text("{}", encoding="utf-8")

    report = cleanup_result_artifacts(result, RetentionPolicy(), now=now)

    assert report["removed_count"] == 3
    assert not (result / "runs" / "old-success").exists()
    assert not (result / "runs" / "old-fail").exists()
    assert not (result / "runtime" / "logs" / "old.log").exists()
    assert (result / "runs" / "current-success").exists()
    assert (result / "runtime" / "logs" / "current.log").exists()
    assert latest.exists()
    assert sentinel.exists()
    assert (result / "runtime" / "chatbot_jobs.json").exists()


def test_prune_registry_removes_expired_entries():
    now = datetime(2026, 6, 7, tzinfo=timezone.utc)
    data = {
        "old": {"updated_at": (now - timedelta(days=10)).isoformat()},
        "current": {"updated_at": (now - timedelta(hours=1)).isoformat()},
    }

    result = prune_registry(data, timestamp_field="updated_at", ttl=timedelta(days=1), now=now)

    assert set(result) == {"current"}
