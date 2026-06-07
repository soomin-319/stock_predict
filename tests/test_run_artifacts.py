from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.reports.run_artifacts import RunArtifactManager


def _metadata(run_id: str, *, environment: str = "production", data_mode: str = "real") -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "environment": environment,
        "data_mode": data_mode,
        "generated_at": "2026-06-07T12:00:00+00:00",
        "status": "pass",
        "blocking_reasons": [],
    }


def _write_required(manager: RunArtifactManager, marker: str) -> None:
    frame = pd.DataFrame([{"marker": marker}])
    for name in ("result_simple.csv", "result_detail.csv", "result_news.csv", "result_disclosure.csv"):
        manager.write_csv(f"csv/{name}", frame)
    manager.write_json("pm_report.json", {**manager.metadata, "marker": marker})
    manager.write_json("pipeline_report.json", {**manager.metadata, "marker": marker})


def test_artifacts_share_run_id_and_manifest_has_hashes(tmp_path: Path):
    manager = RunArtifactManager(tmp_path, _metadata("run-1"))
    _write_required(manager, "one")

    manifest = manager.finalize()

    assert manifest["run_id"] == "run-1"
    assert manifest["promoted"] is True
    assert all(item["sha256"] for item in manifest["artifacts"])
    assert all(item["relative_path"] != "manifest.json" for item in manifest["artifacts"])
    assert json.loads((tmp_path / "latest" / "pipeline_report.json").read_text(encoding="utf-8"))["run_id"] == "run-1"


def test_latest_promoted_only_after_success(tmp_path: Path):
    first = RunArtifactManager(tmp_path, _metadata("run-1"))
    _write_required(first, "old")
    first.finalize()
    second = RunArtifactManager(tmp_path, _metadata("run-2"))
    second.write_json("pipeline_report.json", second.metadata)

    manifest = second.finalize()

    assert manifest["promoted"] is False
    assert manifest["status"] == "fail"
    assert json.loads((tmp_path / "latest" / "pipeline_report.json").read_text(encoding="utf-8"))["marker"] == "old"


def test_smoke_output_cannot_replace_production_latest_or_compatibility_copy(tmp_path: Path):
    production = RunArtifactManager(tmp_path, _metadata("run-prod"))
    _write_required(production, "prod")
    production.finalize()
    smoke = RunArtifactManager(tmp_path, _metadata("run-smoke", environment="smoke", data_mode="sample"))
    _write_required(smoke, "smoke")

    manifest = smoke.finalize()

    assert manifest["promoted"] is False
    assert json.loads((tmp_path / "latest" / "pipeline_report.json").read_text(encoding="utf-8"))["marker"] == "prod"
    legacy = pd.read_csv(tmp_path / "result_simple.csv", encoding="utf-8-sig")
    assert legacy.loc[0, "marker"] == "prod"
