from __future__ import annotations

import json
from pathlib import Path

from src.reports.pm_report import save_pm_report
from src.reports.report_metadata import build_report_metadata, generate_run_id, next_krx_business_day


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


def test_next_krx_business_day_skips_weekends():
    assert next_krx_business_day("2025-06-13") == "2025-06-16"


def test_next_krx_business_day_skips_korean_holiday_cluster():
    assert next_krx_business_day("2025-10-02") == "2025-10-10"
