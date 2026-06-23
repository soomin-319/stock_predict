# P1 Kakao Session Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract Kakao chatbot JSON registry/session persistence into a focused `session_store` module with no behavior changes.

**Architecture:** Add `src/chatbot/session_store.py` as a small persistence boundary. `KakaoColabPredictionBot` keeps existing wrappers and registry attributes, but delegates load/save/session lookups to the new module.

**Tech Stack:** Python 3.10+, dataclasses, pathlib, threading lock protocol, pytest, existing `atomic_write_text` and `redact_value` helpers.

---

## File Structure

- Create: `src/chatbot/session_store.py`
  - Owns generic registry JSON load/save and `ChatbotSessionStore`.
- Modify: `src/chatbot/kakao_colab_bot.py`
  - Imports store helpers.
  - Creates a store after loading session registry.
  - Delegates existing wrapper methods.
- Test: `tests/test_chatbot_session_store.py`
  - Direct unit tests for malformed registry handling, redacted save, update, lookup.
- Modify: `tests/test_kakao_colab_bot.py`
  - Add a delegation/parity test for bot session wrappers.

---

### Task 1: Add failing store unit tests

**Files:**
- Create: `tests/test_chatbot_session_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chatbot_session_store.py`:

```python
from __future__ import annotations

import json
import threading
from pathlib import Path

from src.chatbot.session_store import ChatbotSessionStore, load_registry, save_registry


def test_load_registry_returns_empty_for_missing_invalid_or_non_dict(tmp_path: Path):
    missing = tmp_path / "missing.json"
    assert load_registry(missing) == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad-json", encoding="utf-8")
    assert load_registry(invalid) == {}

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_registry(non_dict) == {}


def test_load_registry_filters_non_dict_entries(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"ok": {"status": "completed"}, "bad": "value"}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_registry(path) == {"ok": {"status": "completed"}}


def test_save_registry_redacts_secret_values(tmp_path: Path):
    path = tmp_path / "registry.json"

    save_registry(path, {"job": {"token": "secret-token", "status": "running"}}, ["secret-token"])

    saved = path.read_text(encoding="utf-8")
    assert "secret-token" not in saved
    assert "***REDACTED***" in saved


def test_chatbot_session_store_updates_and_reads_session(tmp_path: Path):
    path = tmp_path / "sessions.json"
    store = ChatbotSessionStore(path=path, registry={}, lock=threading.RLock(), secret_values=[])

    store.update("user-1", symbol="005930.KS", display_code="005930", intent="tracking")

    assert store.symbol_for("user-1") == "005930.KS"
    assert store.intent_for("user-1") == "tracking"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["user-1"]["last_symbol"] == "005930.KS"
    assert saved["user-1"]["last_display_code"] == "005930"
    assert saved["user-1"]["last_intent"] == "tracking"
    assert saved["user-1"]["updated_at"]


def test_chatbot_session_store_missing_user_defaults(tmp_path: Path):
    store = ChatbotSessionStore(path=tmp_path / "sessions.json", registry={}, lock=threading.RLock(), secret_values=[])

    assert store.symbol_for(None) is None
    assert store.symbol_for("missing") is None
    assert store.intent_for(None) == ""
    assert store.intent_for("missing") == ""
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
pytest tests/test_chatbot_session_store.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.chatbot.session_store'`.

---

### Task 2: Implement `session_store`

**Files:**
- Create: `src/chatbot/session_store.py`
- Test: `tests/test_chatbot_session_store.py`

- [ ] **Step 1: Write minimal implementation**

Create `src/chatbot/session_store.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from src.utils.atomic_files import atomic_write_text
from src.utils.secrets import redact_value


class LockLike(Protocol):
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


@dataclass(slots=True)
class UserSessionState:
    user_id: str
    last_symbol: str | None = None
    last_display_code: str | None = None
    last_intent: str = "idle"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def save_registry(path: Path, data: dict[str, dict[str, Any]], secret_values: Iterable[str]) -> None:
    safe_data = redact_value(data, secret_values)
    atomic_write_text(path, json.dumps(safe_data, ensure_ascii=False, indent=2), encoding="utf-8")


class ChatbotSessionStore:
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

    def update(self, user_id: str | None, symbol: str | None, display_code: str | None, intent: str) -> None:
        if not user_id:
            return
        with self._lock:
            self._registry[user_id] = asdict(
                UserSessionState(
                    user_id=user_id,
                    last_symbol=symbol,
                    last_display_code=display_code,
                    last_intent=intent,
                )
            )
            save_registry(self.path, self._registry, self._secret_values)

    def symbol_for(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        with self._lock:
            session = self._registry.get(user_id, {})
        symbol = session.get("last_symbol")
        return str(symbol) if symbol else None

    def intent_for(self, user_id: str | None) -> str:
        if not user_id:
            return ""
        with self._lock:
            session = self._registry.get(user_id, {})
        return str(session.get("last_intent") or "").strip()
```

- [ ] **Step 2: Run tests and verify pass**

Run:

```powershell
pytest tests/test_chatbot_session_store.py -q
```

Expected: `5 passed`.

- [ ] **Step 3: Commit**

Run:

```powershell
git add src/chatbot/session_store.py tests/test_chatbot_session_store.py
git commit -m "Add Kakao session store"
```

---

### Task 3: Delegate bot registry/session helpers

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`
- Test: existing chatbot tests

- [ ] **Step 1: Update imports**

In `src/chatbot/kakao_colab_bot.py`, add:

```python
from src.chatbot.session_store import ChatbotSessionStore, load_registry, save_registry
```

Keep existing `json`, `asdict`, `field`, `atomic_write_text`, and `redact_value` imports if still used elsewhere.

- [ ] **Step 2: Create store in `__init__`**

Replace the session registry load block:

```python
self._job_registry = self._load_registry(state_load_path)
self._session_registry = self._load_registry(session_load_path)
self._active_processes: dict[str, dict[str, Any]] = {}
self._save_registry(self.state_path, self._job_registry)
if session_load_path != self.session_path:
    self._save_registry(self.session_path, self._session_registry)
```

with:

```python
self._job_registry = self._load_registry(state_load_path)
loaded_session_registry = self._load_registry(session_load_path)
self._session_store = ChatbotSessionStore(
    path=self.session_path,
    registry=loaded_session_registry,
    lock=self._state_lock,
    secret_values=self.runtime_config.secret_values(),
)
self._session_registry = self._session_store.data
self._active_processes: dict[str, dict[str, Any]] = {}
self._save_registry(self.state_path, self._job_registry)
if session_load_path != self.session_path:
    self._save_registry(self.session_path, self._session_registry)
```

- [ ] **Step 3: Delegate wrappers**

Replace the five helper bodies near the bottom of `src/chatbot/kakao_colab_bot.py` with:

```python
def _update_session(self, user_id: str | None, symbol: str | None, intent: str):
    display_code = self._display_code(symbol) if symbol else None
    self._session_store.update(user_id, symbol=symbol, display_code=display_code, intent=intent)

def _symbol_from_session(self, user_id: str | None) -> str | None:
    return self._session_store.symbol_for(user_id)

def _session_intent(self, user_id: str | None) -> str:
    return self._session_store.intent_for(user_id)

def _load_registry(self, path: Path) -> dict[str, dict[str, Any]]:
    return load_registry(path)

def _save_registry(self, path: Path, data: dict[str, dict[str, Any]]):
    save_registry(path, data, self.runtime_config.secret_values())
```

- [ ] **Step 4: Run impacted tests**

Run:

```powershell
pytest tests/test_chatbot_session_store.py tests/test_chatbot_helpers.py tests/test_kakao_colab_bot.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/chatbot/kakao_colab_bot.py
git commit -m "Delegate Kakao sessions to store"
```

---

### Task 4: Add bot delegation coverage

**Files:**
- Modify: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Write the failing/guard test**

Add this test near other session/registry tests in `tests/test_kakao_colab_bot.py`:

```python
def test_bot_session_helpers_delegate_to_session_store(tmp_path: Path, monkeypatch):
    bot = make_bot(tmp_path)
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def fake_update(user_id, *, symbol, display_code, intent):
        calls.append(("update", (user_id,), {"symbol": symbol, "display_code": display_code, "intent": intent}))

    def fake_symbol_for(user_id):
        calls.append(("symbol_for", (user_id,), {}))
        return "005930.KS"

    def fake_intent_for(user_id):
        calls.append(("intent_for", (user_id,), {}))
        return "tracking"

    monkeypatch.setattr(bot._session_store, "update", fake_update)
    monkeypatch.setattr(bot._session_store, "symbol_for", fake_symbol_for)
    monkeypatch.setattr(bot._session_store, "intent_for", fake_intent_for)

    bot._update_session("user-1", "005930.KS", "running")

    assert bot._symbol_from_session("user-1") == "005930.KS"
    assert bot._session_intent("user-1") == "tracking"
    assert calls == [
        ("update", ("user-1",), {"symbol": "005930.KS", "display_code": "005930", "intent": "running"}),
        ("symbol_for", ("user-1",), {}),
        ("intent_for", ("user-1",), {}),
    ]
```

- [ ] **Step 2: Run the new test**

Run:

```powershell
pytest tests/test_kakao_colab_bot.py::test_bot_session_helpers_delegate_to_session_store -q
```

Expected: PASS if Task 3 already delegated; FAIL if wrappers still contain inline logic.

- [ ] **Step 3: Run impacted chatbot tests**

Run:

```powershell
pytest tests/test_chatbot_session_store.py tests/test_chatbot_helpers.py tests/test_kakao_colab_bot.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

Run:

```powershell
git add tests/test_kakao_colab_bot.py
git commit -m "Cover Kakao session store delegation"
```

---

### Task 5: Required verification and PR

**Files:**
- No source edits expected unless verification finds defects.

- [ ] **Step 1: Run impacted + smoke tests**

Run:

```powershell
pytest tests/test_chatbot_session_store.py tests/test_chatbot_helpers.py tests/test_kakao_colab_bot.py tests/test_pipeline_smoke.py -q
```

Expected: all pass.

- [ ] **Step 2: Run sample pipeline**

Run:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit code `0`; generated smoke report remains uncommitted.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
pytest -q
```

Expected: all pass.

- [ ] **Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected: no unstaged source/test/doc changes except ignored generated artifacts.

- [ ] **Step 5: Push and create draft PR**

Run:

```powershell
git push -u origin p1-kakao-session-store
```

Then create a draft PR with base `p1-kakao-message-formatter`, summary, tests, and note that no signal/recommendation/news ranking behavior changed.

---

## Self-Review

- Spec coverage: all goals covered by Tasks 1-4; verification and PR covered by Task 5.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: `ChatbotSessionStore.update`, `symbol_for`, `intent_for`, `load_registry`, and `save_registry` names are consistent across tasks.
- Scope check: job/subprocess and live context refactors are intentionally excluded.
