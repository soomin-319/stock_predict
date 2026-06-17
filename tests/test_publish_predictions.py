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
