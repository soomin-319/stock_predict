# P1 Kakao Job Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract persistent Kakao prediction job registry operations into a focused store while preserving bot runtime behavior.

**Architecture:** Add `src/chatbot/job_store.py` for job state dataclass, registry access, terminal updates, elapsed-time parsing, and stale-startup cleanup. `KakaoColabPredictionBot` keeps subprocess/thread orchestration and delegates persistent state mutations to the store, while retaining `_job_registry` as a compatibility alias.

**Tech Stack:** Python 3.10+, dataclasses, pathlib, pytest, existing `session_store.load_registry/save_registry`, existing secret redaction/atomic file helpers.

---

## File structure

- Create: `src/chatbot/job_store.py`
  - Owns `PredictionJobState` and `ChatbotJobStore`.
  - Reuses `save_registry()` from `src/chatbot/session_store.py`.
- Modify: `src/chatbot/kakao_colab_bot.py`
  - Imports `ChatbotJobStore` and `PredictionJobState`.
  - Removes local `PredictionJobState`.
  - Initializes `_job_store`.
  - Delegates job registry mutations/read helpers to `_job_store`.
- Create: `tests/test_chatbot_job_store.py`
  - Unit tests for store-only behavior.
- Modify: `tests/test_kakao_colab_bot.py`
  - Add delegation coverage.
- Verification:
  - Targeted tests.
  - Impacted chatbot + pipeline smoke.
  - Sample pipeline command.
  - Full `pytest -q`.

---

### Task 1: Lock job-store behavior with failing tests

**Files:**
- Create: `tests/test_chatbot_job_store.py`

- [ ] **Step 1: Add focused store tests**

Create `tests/test_chatbot_job_store.py` with:

```python
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
```

- [ ] **Step 2: Run tests and confirm red**

Run:

```powershell
pytest tests/test_chatbot_job_store.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'src.chatbot.job_store'`.

- [ ] **Step 3: Commit red tests**

Run:

```powershell
git add tests/test_chatbot_job_store.py
git commit -m "Lock Kakao job store behavior"
```

---

### Task 2: Add `ChatbotJobStore`

**Files:**
- Create: `src/chatbot/job_store.py`

- [ ] **Step 1: Implement store**

Create `src/chatbot/job_store.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.chatbot.session_store import LockLike, save_registry


@dataclass(slots=True)
class PredictionJobState:
    symbol: str
    display_code: str
    command: list[str]
    log_path: str
    submitted_at: str
    status: str = "running"
    pid: int | None = None
    exit_code: int | None = None
    completed_at: str | None = None


class ChatbotJobStore:
    def __init__(
        self,
        path: Path,
        registry: dict[str, dict[str, Any]] | None,
        lock: LockLike,
        secret_values: Iterable[str],
    ) -> None:
        self.path = path
        self._registry = registry if registry is not None else {}
        self._lock = lock
        self._secret_values = tuple(secret_values)

    @property
    def data(self) -> dict[str, dict[str, Any]]:
        return self._registry

    def save(self) -> None:
        save_registry(self.path, self._registry, self._secret_values)

    def get(self, symbol: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._registry.get(symbol, {}))

    def set(self, symbol: str, state: PredictionJobState | dict[str, Any]) -> dict[str, Any]:
        next_state = asdict(state) if isinstance(state, PredictionJobState) else dict(state)
        with self._lock:
            self._registry[symbol] = next_state
            self.save()
            return dict(next_state)

    def running_prediction_count(self, bootstrap_key: str) -> int:
        with self._lock:
            return sum(
                1
                for key, state in self._registry.items()
                if key != bootstrap_key and state.get("status") == "running"
            )

    def mark_failed(self, symbol: str, exit_code: int, note: str = "") -> dict[str, Any]:
        with self._lock:
            job_state = dict(self._registry.get(symbol, {}))
            job_state.update(
                {
                    "status": "failed",
                    "exit_code": int(exit_code),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "failure_note": str(note or ""),
                }
            )
            job_state.pop("command", None)
            job_state.pop("pid", None)
            self._registry[symbol] = job_state
            self.save()
            return dict(job_state)

    def mark_completed(self, symbol: str, exit_code: int) -> dict[str, Any]:
        status = "completed" if exit_code == 0 else "failed"
        with self._lock:
            job_state = dict(self._registry.get(symbol, {}))
            job_state.update(
                {
                    "status": status,
                    "exit_code": int(exit_code),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            job_state.pop("command", None)
            job_state.pop("pid", None)
            self._registry[symbol] = job_state
            self.save()
            return dict(job_state)

    def elapsed_seconds(self, job_state: dict[str, Any]) -> float | None:
        completed_at = str(job_state.get("completed_at") or "").strip()
        if not completed_at:
            return None
        try:
            completed_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except Exception:
            return None
        now_utc = datetime.now(timezone.utc)
        if completed_dt.tzinfo is None:
            completed_dt = completed_dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now_utc - completed_dt.astimezone(timezone.utc)).total_seconds())

    def mark_stale_running_on_startup(self) -> bool:
        changed = False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for state in self._registry.values():
                if state.get("status") != "running":
                    continue
                state["status"] = "failed"
                state["exit_code"] = -2
                state["completed_at"] = now
                state["failure_note"] = "stale_running_on_startup"
                state.pop("command", None)
                state.pop("pid", None)
                changed = True
            if changed:
                self.save()
        return changed
```

- [ ] **Step 2: Run store tests**

Run:

```powershell
pytest tests/test_chatbot_job_store.py -q
```

Expected: `6 passed`.

- [ ] **Step 3: Commit implementation**

Run:

```powershell
git add src/chatbot/job_store.py tests/test_chatbot_job_store.py
git commit -m "Add Kakao job store"
```

---

### Task 3: Delegate bot job state operations

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`

- [ ] **Step 1: Update imports and remove local dataclass**

Change imports:

```python
from dataclasses import asdict, dataclass, field
```

to:

```python
from dataclasses import dataclass, field
```

Add:

```python
from src.chatbot.job_store import ChatbotJobStore, PredictionJobState
```

Remove the local `PredictionJobState` dataclass block.

- [ ] **Step 2: Initialize store in `__init__`**

Replace:

```python
self._job_registry = self._load_registry(state_load_path)
```

with:

```python
loaded_job_registry = self._load_registry(state_load_path)
```

After `_state_lock` and before migration saves, add:

```python
self._job_store = ChatbotJobStore(
    path=self.state_path,
    registry=loaded_job_registry,
    lock=self._state_lock,
    secret_values=self.runtime_config.secret_values(),
)
self._job_registry = self._job_store.data
```

Keep the existing migration save condition, but save through the store:

```python
if state_load_path != self.state_path:
    self._job_store.save()
```

- [ ] **Step 3: Delegate startup stale cleanup**

Replace `_cleanup_stale_running_jobs_on_startup()` body with:

```python
def _cleanup_stale_running_jobs_on_startup(self) -> None:
    self._job_store.mark_stale_running_on_startup()
```

- [ ] **Step 4: Delegate terminal helpers**

In `_finalize_process()`, replace manual job state update/save with:

```python
job_state = self._job_store.mark_completed(symbol, int(exit_code))
status = str(job_state.get("status") or "failed")
```

In `_mark_job_failed()`, replace body with:

```python
def _mark_job_failed(self, symbol: str, exit_code: int, note: str = ""):
    self._job_store.mark_failed(symbol, exit_code=exit_code, note=note)
```

In `_job_elapsed_seconds()`, replace body with:

```python
def _job_elapsed_seconds(self, job_state: dict[str, Any]) -> float | None:
    return self._job_store.elapsed_seconds(job_state)
```

- [ ] **Step 5: Delegate count and writes in job start paths**

In `_start_prediction_job()`, replace the running count expression with:

```python
running_count = self._job_store.running_prediction_count(self.BOOTSTRAP_JOB_KEY)
```

Replace failed start write with:

```python
self._job_store.set(symbol, failed_state)
```

Replace running state write with:

```python
self._job_store.set(
    symbol,
    PredictionJobState(
        symbol=symbol,
        display_code=display_code,
        command=safe_command,
        log_path=str(log_path.relative_to(self.project_root)),
        submitted_at=submitted_at,
        status="running",
        pid=getattr(process, "pid", None),
    ),
)
```

In `_start_bootstrap_job()`, replace the running bootstrap state write with:

```python
self._job_store.set(
    self.BOOTSTRAP_JOB_KEY,
    PredictionJobState(
        symbol=self.BOOTSTRAP_JOB_KEY,
        display_code="BOOTSTRAP",
        command=command,
        log_path=str(log_path.relative_to(self.project_root)),
        submitted_at=submitted_at,
        status="running",
    ),
)
```

In `_run_bootstrap_prewarm_worker()`, after updating `state`, replace manual assignment/save with:

```python
self._job_store.set(self.BOOTSTRAP_JOB_KEY, state)
```

- [ ] **Step 6: Run impacted bot tests**

Run:

```powershell
pytest tests/test_chatbot_job_store.py tests/test_chatbot_helpers.py::test_start_prediction_job_deduplicates_running_symbol tests/test_chatbot_helpers.py::test_start_prediction_job_respects_concurrency_limit tests/test_kakao_colab_bot.py::test_finalize_process_logs_completion_without_inline_formatting tests/test_kakao_colab_bot.py::test_finalize_process_handles_missing_log_handle -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit delegation**

Run:

```powershell
git add src/chatbot/kakao_colab_bot.py src/chatbot/job_store.py tests/test_chatbot_job_store.py
git commit -m "Delegate Kakao jobs to store"
```

---

### Task 4: Add bot delegation regression tests

**Files:**
- Modify: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Add delegation tests near existing session delegation test**

Append:

```python
def test_bot_job_helpers_delegate_to_job_store(tmp_path: Path, monkeypatch):
    bot = make_bot(tmp_path)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        bot._job_store,
        "mark_failed",
        lambda symbol, exit_code, note="": calls.append(("mark_failed", (symbol, exit_code, note))),
    )
    monkeypatch.setattr(
        bot._job_store,
        "elapsed_seconds",
        lambda job_state: calls.append(("elapsed_seconds", job_state)) or 12.5,
    )
    monkeypatch.setattr(
        bot._job_store,
        "mark_stale_running_on_startup",
        lambda: calls.append(("stale", None)) or True,
    )

    bot._mark_job_failed("005930.KS", -2, "stale_running_state")
    elapsed = bot._job_elapsed_seconds({"completed_at": "invalid"})
    bot._cleanup_stale_running_jobs_on_startup()

    assert elapsed == 12.5
    assert calls == [
        ("mark_failed", ("005930.KS", -2, "stale_running_state")),
        ("elapsed_seconds", {"completed_at": "invalid"}),
        ("stale", None),
    ]
```

- [ ] **Step 2: Run new regression test**

Run:

```powershell
pytest tests/test_kakao_colab_bot.py::test_bot_job_helpers_delegate_to_job_store -q
```

Expected: pass.

- [ ] **Step 3: Commit regression test**

Run:

```powershell
git add tests/test_kakao_colab_bot.py
git commit -m "Cover Kakao job store delegation"
```

---

### Task 5: Verification and PR

**Files:**
- No new source changes expected.

- [ ] **Step 1: Run impacted tests**

Run:

```powershell
pytest tests/test_chatbot_job_store.py tests/test_chatbot_helpers.py tests/test_kakao_colab_bot.py tests/test_pipeline_smoke.py -q
```

Expected: all pass.

- [ ] **Step 2: Run sample pipeline smoke**

Run:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit code 0 and sample recommendations print. Do not commit `pipeline_report_smoke.json`.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
pytest -q
```

Expected: all pass. Existing pandas warning is acceptable if unchanged.

- [ ] **Step 4: Push and create draft PR**

Run:

```powershell
git status --short
git push -u origin p1-kakao-job-store
```

Create a draft PR:
- Title: `Extract Kakao job store`
- Base: `p1-kakao-session-store`
- Summary:
  - Adds `ChatbotJobStore` and moves persistent job registry state helpers out of the bot.
  - Delegates failed/completed/stale/elapsed helpers while keeping subprocess orchestration in the bot.
  - Preserves recommendation/signal/ranking behavior and display-only news/disclosure guardrails.
- Tests:
  - targeted job-store tests
  - impacted chatbot tests
  - sample pipeline smoke
  - full `pytest -q`

---

## Self-review

- Spec coverage: store extraction, compatibility alias, stale startup, terminal updates, elapsed parsing, redaction, bot delegation, and verification are covered.
- Placeholder scan: no placeholder steps remain.
- Type consistency: `ChatbotJobStore`, `PredictionJobState`, `mark_failed`, `mark_completed`, `elapsed_seconds`, `mark_stale_running_on_startup`, `running_prediction_count`, and `set` signatures are consistent across tasks.
