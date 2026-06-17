from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ops.publish_predictions import _effective_news_mode, publish_artifacts
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


from dataclasses import dataclass

from src.ops.publish_predictions import run_publish


@dataclass
class _Args:
    news_mode: str = "gemma"
    full_refresh: bool = False
    no_push: bool = True
    dry_run: bool = False
    config_json: str | None = None


def test_run_publish_invokes_pipeline_and_publishes(tmp_path: Path):
    project_root = tmp_path
    result_root = project_root / "result"
    run_dir = result_root / "runs" / "rid-9"
    _make_run_dir(run_dir)
    (result_root / "latest_manifest.json").write_text('{"run_id": "rid-9"}', encoding="utf-8")

    pipeline_calls = []

    def fake_pipeline(news_impact_llm_config, full_refresh, config_json=None):
        pipeline_calls.append({"cfg": news_impact_llm_config, "full": full_refresh})
        return {"manifest": {"promoted": True, "status": "pass", "run_id": "rid-9"}}

    git_calls = []

    result = run_publish(
        _Args(),
        project_root=project_root,
        pipeline_fn=fake_pipeline,
        git_fn=lambda *a, **k: git_calls.append((a, k)),
    )

    assert pipeline_calls[0]["cfg"].endswith("news_impact.gemma.example.json")
    assert (project_root / "published" / "latest" / "csv" / "result_simple.csv").exists()
    assert result["trading_date"] == "2026-06-17"
    assert git_calls == []


def test_run_publish_rule_mode_uses_no_llm_config(tmp_path: Path):
    project_root = tmp_path
    run_dir = project_root / "result" / "runs" / "rid-r"
    _make_run_dir(run_dir)
    (project_root / "result" / "latest_manifest.json").write_text('{"run_id": "rid-r"}', encoding="utf-8")

    captured = {}

    def fake_pipeline(news_impact_llm_config, full_refresh, config_json=None):
        captured["cfg"] = news_impact_llm_config
        return {"manifest": {"promoted": True, "status": "pass", "run_id": "rid-r"}}

    run_publish(
        _Args(news_mode="rule"),
        project_root=project_root,
        pipeline_fn=fake_pipeline,
        git_fn=lambda *a, **k: None,
    )
    assert captured["cfg"] is None


def test_effective_news_mode_rule_aliases_and_default():
    assert _effective_news_mode({"news_impact": {"mode": "rule"}}) == "rule_based"
    assert _effective_news_mode({"news_impact": {"mode": "rule_based"}}) == "rule_based"
    assert _effective_news_mode({"news_impact": {"mode": "heuristic"}}) == "rule_based"
    assert _effective_news_mode({}) == "gemma"
    assert _effective_news_mode({"news_impact": {"mode": "llm"}}) == "gemma"
