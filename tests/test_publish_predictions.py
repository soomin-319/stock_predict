from __future__ import annotations

import json
from pathlib import Path

from src.ops.publish_predictions import publish_artifacts
from src.ops.published_store import read_index


def _make_run_dir(run_dir: Path) -> None:
    (run_dir / "csv").mkdir(parents=True)
    detail = "Symbol,Date\n005930.KS,2026-06-17\n000660.KS,2026-06-17\n"
    (run_dir / "csv" / "result_simple.csv").write_text(
        "종목코드,종목명\n005930,삼성전자\n000660,SK하이닉스\n", encoding="utf-8-sig"
    )
    (run_dir / "csv" / "result_detail.csv").write_text(detail, encoding="utf-8-sig")
    (run_dir / "csv" / "result_news.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_disclosure.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "manifest.json").write_text('{"run_id": "rid-1", "promoted": true}', encoding="utf-8")
    (run_dir / "pipeline_report.json").write_text('{"ok": true}', encoding="utf-8")


def test_publish_artifacts_writes_latest_history_index(tmp_path: Path):
    run_dir = tmp_path / "result" / "runs" / "rid-1"
    _make_run_dir(run_dir)
    published_root = tmp_path / "published"

    meta = publish_artifacts(
        run_dir=run_dir,
        published_root=published_root,
        trading_date="2026-06-17",
        news_mode="gemma",
        source_run_id="rid-1",
        symbol_count=2,
    )

    assert meta.trading_date == "2026-06-17"
    latest_simple = published_root / "latest" / "csv" / "result_simple.csv"
    hist_simple = published_root / "history" / "2026-06-17" / "csv" / "result_simple.csv"
    assert latest_simple.exists() and hist_simple.exists()
    assert json.loads((published_root / "latest" / "publish_meta.json").read_text(encoding="utf-8"))["news_mode"] == "gemma"
    assert read_index(published_root)["latest"] == "2026-06-17"


import pytest

from src.ops.publish_predictions import infer_trading_date, ensure_operational_manifest


def test_infer_trading_date_from_detail(tmp_path: Path):
    run_dir = tmp_path / "runs" / "rid-1"
    (run_dir / "csv").mkdir(parents=True)
    (run_dir / "csv" / "result_detail.csv").write_text(
        "Symbol,Date\n005930.KS,2026-06-16\n005930.KS,2026-06-17\n", encoding="utf-8-sig"
    )
    assert infer_trading_date(run_dir) == "2026-06-17"


def test_ensure_operational_manifest_accepts_promoted_pass():
    ensure_operational_manifest(
        {"promoted": True, "status": "pass", "environment": "production", "data_mode": "real"}
    )


def test_ensure_operational_manifest_rejects_non_operational():
    with pytest.raises(ValueError):
        ensure_operational_manifest({"promoted": False, "status": "fail"})
