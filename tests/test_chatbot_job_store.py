from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from src.chatbot.job_store import ChatbotJobStore, PredictionJobState


def test_job_store_sets_running_state_and_redacts_secret(tmp_path):
    path = tmp_path / "jobs.json"
    store = ChatbotJobStore(path=path, registry={}, lock=threading.RLock(), secret_values=("SECRET",))

    state = PredictionJobState(
        symbol="005930.KS",
        display_code="005930",
        command=["python", "--token", "SECRET"],
        log_path="result/runtime/005930.log",
        submitted_at="2026-06-23T00:00:00+00:00",
        pid=123,
    )
    store.set("005930.KS", state)

    assert store.data["005930.KS"]["status"] == "running"
    assert store.data["005930.KS"]["pid"] == 123
    assert "SECRET" not in path.read_text(encoding="utf-8")
    assert "[REDACTED]" in path.read_text(encoding="utf-8")


def test_job_store_mark_failed_strips_command_and_pid(tmp_path):
    path = tmp_path / "jobs.json"
    store = ChatbotJobStore(
        path=path,
        registry={"005930.KS": {"status": "running", "command": ["python"], "pid": 123}},
        lock=threading.RLock(),
        secret_values=(),
    )

    state = store.mark_failed("005930.KS", exit_code=-2, note="stale_running_state")

    assert state["status"] == "failed"
    assert state["exit_code"] == -2
    assert state["failure_note"] == "stale_running_state"
    assert "completed_at" in state
    assert "command" not in state
    assert "pid" not in state
    assert path.exists()


def test_job_store_mark_completed_strips_runtime_fields(tmp_path):
    path = tmp_path / "jobs.json"
    store = ChatbotJobStore(
        path=path,
        registry={"005930.KS": {"status": "running", "command": ["python"], "pid": 123}},
        lock=threading.RLock(),
        secret_values=(),
    )

    state = store.mark_completed("005930.KS", exit_code=0)

    assert state["status"] == "completed"
    assert state["exit_code"] == 0
    assert "completed_at" in state
    assert "command" not in state
    assert "pid" not in state


def test_job_store_elapsed_seconds_handles_valid_missing_and_invalid(tmp_path):
    store = ChatbotJobStore(path=tmp_path / "jobs.json", registry={}, lock=threading.RLock(), secret_values=())
    completed_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()

    assert store.elapsed_seconds({"completed_at": completed_at}) is not None
    assert store.elapsed_seconds({}) is None
    assert store.elapsed_seconds({"completed_at": "not-a-date"}) is None


def test_job_store_running_count_excludes_bootstrap(tmp_path):
    store = ChatbotJobStore(
        path=tmp_path / "jobs.json",
        registry={
            "005930.KS": {"status": "running"},
            "000660.KS": {"status": "completed"},
            "__bootstrap__": {"status": "running"},
        },
        lock=threading.RLock(),
        secret_values=(),
    )

    assert store.running_prediction_count("__bootstrap__") == 1


def test_job_store_marks_stale_running_jobs_on_startup(tmp_path):
    path = tmp_path / "jobs.json"
    store = ChatbotJobStore(
        path=path,
        registry={
            "005930.KS": {"status": "running", "command": ["python"], "pid": 123},
            "000660.KS": {"status": "completed", "pid": 456},
        },
        lock=threading.RLock(),
        secret_values=(),
    )

    changed = store.mark_stale_running_on_startup()

    assert changed is True
    stale = store.data["005930.KS"]
    assert stale["status"] == "failed"
    assert stale["exit_code"] == -2
    assert stale["failure_note"] == "stale_running_on_startup"
    assert "command" not in stale
    assert "pid" not in stale
    assert store.data["000660.KS"]["status"] == "completed"
    assert path.exists()
