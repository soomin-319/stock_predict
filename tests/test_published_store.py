from __future__ import annotations

from pathlib import Path

from src.ops.published_store import (
    PUBLISHED_ARTIFACTS,
    resolve_published_dir,
)


def test_published_artifacts_set():
    assert PUBLISHED_ARTIFACTS == (
        "csv/result_simple.csv",
        "csv/result_detail.csv",
        "csv/result_news.csv",
        "csv/result_disclosure.csv",
        "manifest.json",
        "pipeline_report.json",
    )


def test_resolve_published_dir_latest_and_history(tmp_path: Path):
    root = tmp_path / "published"
    assert resolve_published_dir(root, None) == root / "latest"
    assert resolve_published_dir(root, "2026-06-17") == root / "history" / "2026-06-17"
