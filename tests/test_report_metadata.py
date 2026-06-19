from __future__ import annotations

import json
from pathlib import Path

from src.reports.pm_report import save_pm_report, validate_pm_report_schema
from src.reports.report_metadata import (
    build_report_metadata,
    evaluate_krx_calendar_coverage,
    generate_run_id,
    next_krx_business_day,
)


def test_report_metadata_contains_required_identity_fields():
    run_id = generate_run_id()

    metadata = build_report_metadata(
        run_id=run_id,
        environment="smoke",
        data_mode="sample",
        input_as_of_date="2023-08-10",
        prediction_for_date="2023-08-11",
        context_as_of_date=None,
        config_payload={"한글": 1},
    )

    assert metadata["schema_version"] == "1.0"
    assert metadata["run_id"] == run_id
    assert metadata["environment"] == "smoke"
    assert metadata["data_mode"] == "sample"
    assert len(metadata["config_hash"]) == 64
    assert metadata["status"] == "pass"


def test_pm_report_is_written_atomically_as_utf8(tmp_path: Path):
    path = tmp_path / "pm_report.json"

    save_pm_report({"상태": "정상"}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"상태": "정상"}


def test_pm_report_schema_validator_requires_contract_fields():
    ok, missing = validate_pm_report_schema(
        {
            "schema_version": "1.0",
            "run_id": "run-1",
            "environment": "production",
            "data_mode": "real",
            "generated_at": "2026-06-18T00:00:00+00:00",
            "input_as_of_date": "2026-06-17",
            "prediction_for_date": "2026-06-18",
            "context_as_of_date": "2026-06-17",
            "git_commit": "abc123",
            "config_hash": "x" * 64,
            "status": "pass",
            "blocking_reasons": [],
            "coverage_gate": {},
            "pm_summary": {},
            "risk_flag_counts": {},
            "horizon_summary": {},
            "top_buy_candidates": [],
        }
    )
    assert ok is True
    assert missing == []

    ok2, missing2 = validate_pm_report_schema({"run_id": "run-1"})
    assert ok2 is False
    assert "schema_version" in missing2
    assert "top_buy_candidates" in missing2


def test_next_krx_business_day_skips_weekends():
    assert next_krx_business_day("2025-06-13") == "2025-06-16"


def test_next_krx_business_day_skips_korean_holiday_cluster():
    assert next_krx_business_day("2025-10-02") == "2025-10-10"


def test_report_metadata_warns_when_krx_calendar_coverage_expired():
    metadata = build_report_metadata(
        run_id="run-calendar-expired",
        environment="production",
        data_mode="real",
        input_as_of_date="2026-12-31",
        prediction_for_date="2027-01-01",
        context_as_of_date=None,
        config_payload={},
    )

    assert metadata["calendar_status"] == "expired"
    assert metadata["calendar_coverage_end"] == "2026-12-31"
    assert metadata["calendar_warnings"] == ["krx_calendar_coverage_expired:2026-12-31"]
    assert metadata["status"] == "warning"
    assert "krx_calendar_coverage_expired:2026-12-31" in metadata["blocking_reasons"]


def test_krx_calendar_coverage_warns_near_expiry():
    coverage = evaluate_krx_calendar_coverage("2026-12-15")

    assert coverage["status"] == "near_expiry"
    assert coverage["coverage_end"] == "2026-12-31"
    assert coverage["warnings"] == ["krx_calendar_coverage_near_expiry:2026-12-31"]
