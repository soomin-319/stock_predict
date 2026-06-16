from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.utils.result_cleanup import RetentionPolicy, cleanup_runs


def _write_run(runs_root: Path, run_id: str, *, status: str = "pass", days_old: int = 0) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps({"run_id": run_id, "status": status}), encoding="utf-8")
    timestamp = (datetime(2026, 6, 16, tzinfo=timezone.utc) - timedelta(days=days_old)).timestamp()
    os.utime(manifest, (timestamp, timestamp))
    os.utime(run_dir, (timestamp, timestamp))
    return run_dir


def test_cleanup_runs_preserves_run_referenced_by_latest_manifest(tmp_path: Path):
    runs_root = tmp_path / "runs"
    protected = _write_run(runs_root, "run-protected", days_old=20)
    _write_run(runs_root, "run-new", days_old=0)
    (tmp_path / "latest").mkdir()
    (tmp_path / "latest" / "manifest.json").write_text(
        json.dumps({"run_id": "run-protected", "promoted": True}),
        encoding="utf-8",
    )

    removed = cleanup_runs(
        runs_root,
        RetentionPolicy(successful_run_count=1, successful_run_days=10, failed_run_days=10),
        datetime(2026, 6, 16, tzinfo=timezone.utc),
    )

    assert protected.exists()
    assert str(protected) not in removed
