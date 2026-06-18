from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.news_impact.pipeline import _build_audit_payload
from src.news_impact.schema import RunAudit


def _make_audit(**overrides) -> RunAudit:
    base = dict(
        run_id="run-1",
        run_started_at=datetime(2026, 6, 18, tzinfo=timezone.utc),
        git_commit="abc123",
        config_hash="cfg",
        watchlist_hash="wl",
        company_master_snapshot_id="cm",
        data_snapshot_id="ds",
        llm_provider="openai",
        llm_model_requested="gpt-5-mini",
        llm_model_returned="gpt-5-mini",
        llm_temperature=0.1,
        llm_prompt_hash="deadbeef",
        scoring_version="scoring.v1",
        backtest_version="backtest.v1",
    )
    base.update(overrides)
    return RunAudit(**base)


def test_run_audit_to_dict_includes_llm_reproducibility_fields():
    data = _make_audit().to_dict()

    assert data["llm_temperature"] == 0.1
    assert data["llm_prompt_hash"] == "deadbeef"


def test_build_audit_payload_replay_exposes_llm_reproducibility(tmp_path: Path):
    payload = _build_audit_payload(
        audit=_make_audit(),
        output_dir=tmp_path,
        watchlist_ticker_count=1,
        news_count=0,
        disclosure_count=0,
        impact_event_count=0,
        report_row_count=0,
        llm_failed_count=0,
        semantic_cluster_metrics={
            "failed_count": 0,
            "failure_rate": 0.0,
            "failure_causes": [],
        },
        risk_flags=[],
    )

    assert payload["llm_temperature"] == 0.1
    assert payload["llm_prompt_hash"] == "deadbeef"
    assert payload["replay"]["llm_temperature"] == 0.1
    assert payload["replay"]["llm_prompt_hash"] == "deadbeef"
    assert payload["replay"]["llm_model_requested"] == "gpt-5-mini"
