from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ops.published_store import (
    PUBLISHED_ARTIFACTS,
    PublishMeta,
    copy_published_set,
    load_published_simple,
    read_index,
    read_publish_meta,
    resolve_published_dir,
    update_index,
    write_publish_meta,
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


def _make_run_dir(run_dir: Path) -> None:
    (run_dir / "csv").mkdir(parents=True)
    (run_dir / "csv" / "result_simple.csv").write_text("종목코드\n005930\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_detail.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_news.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_disclosure.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "manifest.json").write_text('{"run_id": "rid-1", "promoted": true}', encoding="utf-8")
    (run_dir / "pipeline_report.json").write_text('{"ok": true}', encoding="utf-8")


def test_copy_published_set_copies_all_artifacts(tmp_path: Path):
    run_dir = tmp_path / "runs" / "rid-1"
    _make_run_dir(run_dir)
    dest = tmp_path / "published" / "latest"

    copy_published_set(run_dir, dest)

    assert (dest / "csv" / "result_simple.csv").read_text(encoding="utf-8-sig") == "종목코드\n005930\n"
    assert json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["run_id"] == "rid-1"
    for rel in (
        "csv/result_detail.csv",
        "csv/result_news.csv",
        "csv/result_disclosure.csv",
        "pipeline_report.json",
    ):
        assert (dest / rel).exists()


def test_copy_published_set_overwrites_existing(tmp_path: Path):
    run_dir = tmp_path / "runs" / "rid-1"
    _make_run_dir(run_dir)
    dest = tmp_path / "published" / "history" / "2026-06-17"
    dest.mkdir(parents=True)
    (dest / "stale.txt").write_text("old", encoding="utf-8")
    (dest / "csv").mkdir()
    (dest / "csv" / "result_simple.csv").write_text("종목코드\n000660\n", encoding="utf-8-sig")

    copy_published_set(run_dir, dest)

    assert (dest / "csv" / "result_simple.csv").read_text(encoding="utf-8-sig") == "종목코드\n005930\n"
    assert (dest / "stale.txt").read_text(encoding="utf-8") == "old"


def test_publish_meta_to_dict_shape():
    meta = PublishMeta(
        generated_at_kst="2026-06-17T18:05:00+09:00",
        trading_date="2026-06-17",
        news_mode="gemma",
        source_run_id="rid-1",
        symbol_count=200,
        git_commit="abc123",
        git_branch="main",
    )
    assert meta.to_dict() == {
        "generated_at_kst": "2026-06-17T18:05:00+09:00",
        "trading_date": "2026-06-17",
        "news_mode": "gemma",
        "source_run_id": "rid-1",
        "symbol_count": 200,
        "git": {"commit": "abc123", "branch": "main"},
    }


def _meta(date: str, mode: str = "gemma") -> PublishMeta:
    return PublishMeta(
        generated_at_kst=f"{date}T18:05:00+09:00",
        trading_date=date,
        news_mode=mode,
        source_run_id=f"rid-{date}",
        symbol_count=200,
    )


def test_write_publish_meta(tmp_path: Path):
    dest = tmp_path / "published" / "latest"
    dest.mkdir(parents=True)
    write_publish_meta(dest, _meta("2026-06-17"))
    payload = json.loads((dest / "publish_meta.json").read_text(encoding="utf-8"))
    assert payload["trading_date"] == "2026-06-17"
    assert payload["git"] == {"commit": None, "branch": None}


def test_update_index_dedups_and_sorts(tmp_path: Path):
    root = tmp_path / "published"
    root.mkdir()
    update_index(root, _meta("2026-06-16"))
    update_index(root, _meta("2026-06-17"))
    update_index(root, _meta("2026-06-17", mode="rule_based"))

    index = read_index(root)
    assert index["latest"] == "2026-06-17"
    dates = [e["trading_date"] for e in index["entries"]]
    assert dates == ["2026-06-17", "2026-06-16"]
    latest_entry = index["entries"][0]
    assert latest_entry["news_mode"] == "rule_based"
    assert latest_entry["symbol_count"] == 200


def test_load_published_simple_reads_codes_as_str(tmp_path: Path):
    dest = tmp_path / "published" / "latest"
    (dest / "csv").mkdir(parents=True)
    (dest / "csv" / "result_simple.csv").write_text(
        "종목코드,종목명\n005930,삼성전자\n", encoding="utf-8-sig"
    )
    df = load_published_simple(dest)
    assert list(df["종목코드"]) == ["005930"]
    assert df["종목코드"].dtype == object


def test_load_published_simple_missing_returns_empty(tmp_path: Path):
    df = load_published_simple(tmp_path / "nope")
    assert df.empty


def test_read_publish_meta_missing_returns_empty(tmp_path: Path):
    assert read_publish_meta(tmp_path / "nope") == {}
