# Chatbot Operations Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instructions forbid subagents, so execute inline only.

**Goal:** Add configurable chatbot runtime persistence, stale running-job cleanup, Kakao recommendation listCard output, and optional webhook CIDR allowlisting.

**Architecture:** Keep changes small and localized. `PipelineRuntimeConfig` owns new operational config, `KakaoColabPredictionBot` derives runtime paths and startup cleanup, `responses.py` owns Kakao response shapes, and Flask webhook code enforces CIDR before shared-secret auth.

**Tech Stack:** Python 3.10+, Flask test client, `ipaddress`, pytest, pandas-backed existing chatbot helpers.

---

## File Structure

- Modify `src/chatbot/kakao_colab_bot.py`
  - Add config fields: `runtime_dir`, `allowed_webhook_cidrs`.
  - Derive runtime paths from `runtime_dir`.
  - Mark loaded stale `running` jobs failed on startup.
  - Use rich recommendation response helper when safe.
  - Enforce webhook CIDR allowlist.
  - Add CLI/env parsing for new options.
- Modify `src/chatbot/responses.py`
  - Add `list_card_response()` helper.
  - Keep existing simpleText helpers unchanged for compatibility.
- Modify `tests/test_chatbot_helpers.py`
  - Add tests for runtime paths, stale cleanup, listCard shape, recommendation rich response, CIDR allow/deny.
- Modify `docs/09_chatbot.md`
  - Document runtime directory, stale cleanup, listCard display, CIDR allowlist.

---

### Task 1: Runtime Directory Config and Stale Startup Cleanup

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`
- Test: `tests/test_chatbot_helpers.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_chatbot_helpers.py`:

```python
def test_runtime_dir_derives_default_state_session_logs(tmp_path):
    runtime_dir = tmp_path / "drive_runtime"
    cfg = PipelineRuntimeConfig(runtime_dir=str(runtime_dir))
    bot = KakaoColabPredictionBot(runtime_config=cfg)

    assert bot.state_path == runtime_dir / "chatbot_jobs.json"
    assert bot.session_path == runtime_dir / "chatbot_sessions.json"
    assert bot.prewarm_meta_path == runtime_dir / "prewarm_cache_meta.json"
    assert bot.log_dir == runtime_dir / "logs"


def test_startup_marks_stale_running_jobs_failed(tmp_path):
    state_path = tmp_path / "jobs.json"
    state_path.write_text(
        '{"005930.KS":{"symbol":"005930.KS","display_code":"005930.KS",'
        '"command":[],"log_path":"result/runtime/logs/old.log",'
        '"submitted_at":"2026-06-18T00:00:00+00:00","status":"running"}}',
        encoding="utf-8",
    )

    bot = make_test_bot(tmp_path, state_path=state_path)

    state = bot._job_registry["005930.KS"]
    assert state["status"] == "failed"
    assert state["exit_code"] == -2
    assert state["note"] == "stale_after_restart"
    assert state["completed_at"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_chatbot_helpers.py::test_runtime_dir_derives_default_state_session_logs tests/test_chatbot_helpers.py::test_startup_marks_stale_running_jobs_failed -q
```

Expected: fail because `runtime_dir` and stale cleanup do not exist.

- [ ] **Step 3: Implement minimal code**

In `PipelineRuntimeConfig`, add:

```python
runtime_dir: str = "result/runtime"
```

In `KakaoColabPredictionBot.__init__`, replace fixed runtime paths with:

```python
runtime_dir = Path(self.runtime_config.runtime_dir)
if not runtime_dir.is_absolute():
    runtime_dir = self.project_root / runtime_dir
self.runtime_dir = runtime_dir
self.state_path = self.project_root / state_path if state_path is not None else runtime_dir / "chatbot_jobs.json"
self.session_path = self.project_root / session_path if session_path is not None else runtime_dir / "chatbot_sessions.json"
self.prewarm_meta_path = runtime_dir / "prewarm_cache_meta.json"
self.log_dir = runtime_dir / "logs"
```

Use safe path handling so explicit absolute paths are preserved:

```python
def _resolve_project_path(self, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else self.project_root / path
```

After loading registries and migrations, call:

```python
self._cleanup_stale_running_jobs_on_startup()
```

Add method:

```python
def _cleanup_stale_running_jobs_on_startup(self) -> None:
    changed = False
    now = datetime.now(timezone.utc).isoformat()
    with self._state_lock:
        for state in self._job_registry.values():
            if state.get("status") != "running":
                continue
            state["status"] = "failed"
            state["exit_code"] = -2
            state["completed_at"] = now
            state["note"] = "stale_after_restart"
            state.pop("pid", None)
            changed = True
        if changed:
            self._save_registry(self.state_path, self._job_registry)
```

- [ ] **Step 4: Run tests and verify pass**

Run same pytest command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/chatbot/kakao_colab_bot.py tests/test_chatbot_helpers.py
git commit -m "Persist chatbot runtime state in configurable directory"
```

---

### Task 2: Kakao listCard Recommendation Display

**Files:**
- Modify: `src/chatbot/responses.py`
- Modify: `src/chatbot/kakao_colab_bot.py`
- Test: `tests/test_chatbot_helpers.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
from dataclasses import dataclass
from src.chatbot.responses import list_card_response


def test_list_card_response_shape_and_quick_replies():
    response = list_card_response(
        header_title="실시간 추천",
        items=[
            {"title": "1. 삼성전자", "description": "005930.KS | 점수 250"},
            {"title": "2. SK하이닉스", "description": "000660.KS | 점수 240"},
        ],
        quick_replies=[("다시 추천", "추천")],
    )

    assert response["version"] == "2.0"
    card = response["template"]["outputs"][0]["listCard"]
    assert card["header"]["title"] == "실시간 추천"
    assert card["items"][0]["title"] == "1. 삼성전자"
    assert response["template"]["quickReplies"][0]["messageText"] == "추천"


@dataclass
class _RecommendationItem:
    rank: int
    name: str
    symbol: str
    final_score: float


class _RecommendationService:
    def get_recommendations(self, **kwargs):
        return [
            _RecommendationItem(1, "삼성전자", "005930.KS", 250.5),
            _RecommendationItem(2, "SK하이닉스", "000660.KS", 240.0),
        ]


def test_recommendation_request_uses_list_card_when_items_available(tmp_path):
    bot = make_test_bot(tmp_path, recommendation_service=_RecommendationService())

    response = bot.handle_utterance("추천")

    assert "listCard" in response["template"]["outputs"][0]
    card = response["template"]["outputs"][0]["listCard"]
    assert card["items"][0]["title"] == "1. 삼성전자"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_chatbot_helpers.py::test_list_card_response_shape_and_quick_replies tests/test_chatbot_helpers.py::test_recommendation_request_uses_list_card_when_items_available -q
```

Expected: fail because `list_card_response` does not exist.

- [ ] **Step 3: Implement minimal code**

In `responses.py`, add:

```python
def list_card_response(
    header_title: str,
    items: list[dict[str, str]],
    quick_replies: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    safe_items = []
    for item in items[:5]:
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        if not title:
            continue
        safe_items.append({"title": title, "description": description})
    if not safe_items:
        return simple_text_response(header_title)
    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "listCard": {
                        "header": {"title": str(header_title or "추천 목록")},
                        "items": safe_items,
                    }
                }
            ]
        },
    }
    return attach_quick_replies(response, quick_replies[:10] if quick_replies else None)
```

In `kakao_colab_bot.py`, import it:

```python
from src.chatbot.responses import attach_quick_replies, list_card_response, simple_text_response
```

Add helper:

```python
def _recommendation_list_card_response(self, recommendations: list[Any] | tuple[Any, ...]) -> dict[str, Any] | None:
    items = []
    for item in list(recommendations)[:5]:
        rank = getattr(item, "rank", None)
        name = str(getattr(item, "name", "") or "").strip()
        symbol = str(getattr(item, "symbol", "") or "").strip()
        score = getattr(item, "final_score", None)
        if not name:
            continue
        title = f"{rank}. {name}" if rank not in (None, "") else name
        desc_parts = [part for part in (symbol, f"점수 {score:g}" if isinstance(score, (int, float)) else "") if part]
        items.append({"title": title, "description": " | ".join(desc_parts)})
    if not items:
        return None
    return list_card_response("실시간 추천", items, quick_replies=[("다시 추천", "추천"), ("도움말", "도움말")])
```

Use it in `_handle_recommendation_request()` after logging:

```python
rich_response = self._recommendation_list_card_response(recommendations)
if rich_response is not None:
    return rich_response
```

- [ ] **Step 4: Run tests and verify pass**

Run same pytest command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/chatbot/responses.py src/chatbot/kakao_colab_bot.py tests/test_chatbot_helpers.py
git commit -m "Show chatbot recommendations as Kakao list cards"
```

---

### Task 3: Webhook CIDR Allowlist

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`
- Test: `tests/test_chatbot_helpers.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
def test_kakao_webhook_rejects_disallowed_remote_addr(tmp_path):
    bot = make_test_bot(tmp_path)
    cfg = PipelineRuntimeConfig(allowed_webhook_cidrs=("10.0.0.0/8",))
    app = create_app(bot=bot, runtime_config=cfg)
    client = app.test_client()

    response = client.post("/kakao/webhook", json={}, environ_base={"REMOTE_ADDR": "127.0.0.1"})

    assert response.status_code == 403


def test_kakao_webhook_accepts_allowed_remote_addr(tmp_path):
    bot = make_test_bot(tmp_path)
    cfg = PipelineRuntimeConfig(allowed_webhook_cidrs=("127.0.0.1/32",))
    app = create_app(bot=bot, runtime_config=cfg)
    client = app.test_client()

    response = client.post("/kakao/webhook", json={}, environ_base={"REMOTE_ADDR": "127.0.0.1"})

    assert response.status_code == 200
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_chatbot_helpers.py::test_kakao_webhook_rejects_disallowed_remote_addr tests/test_chatbot_helpers.py::test_kakao_webhook_accepts_allowed_remote_addr -q
```

Expected: fail because `allowed_webhook_cidrs` does not exist.

- [ ] **Step 3: Implement minimal code**

At top of `kakao_colab_bot.py`, import:

```python
import ipaddress
```

In `PipelineRuntimeConfig`, add:

```python
allowed_webhook_cidrs: tuple[str, ...] = ()
```

Add helper functions:

```python
def _parse_csv_tuple(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _is_remote_addr_allowed(remote_addr: str | None, cidrs: tuple[str, ...]) -> bool:
    if not cidrs:
        return True
    try:
        ip = ipaddress.ip_address(str(remote_addr or ""))
    except ValueError:
        return False
    for cidr in cidrs:
        try:
            network = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            continue
        if ip in network:
            return True
    return False
```

In webhook before secret check:

```python
if not _is_remote_addr_allowed(request.remote_addr, effective_config.allowed_webhook_cidrs):
    return jsonify({"error": "forbidden"}), 403
```

- [ ] **Step 4: Run tests and verify pass**

Run same pytest command. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/chatbot/kakao_colab_bot.py tests/test_chatbot_helpers.py
git commit -m "Restrict chatbot webhook by CIDR allowlist"
```

---

### Task 4: CLI/Environment Wiring and Documentation

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`
- Modify: `docs/09_chatbot.md`
- Test: `tests/test_chatbot_helpers.py`

- [ ] **Step 1: Write focused parsing test**

If a public parser helper exists, test it. If not, skip parser internals and verify config fields through direct dataclass construction already covered.

- [ ] **Step 2: Implement CLI/env wiring**

In argparse setup, add:

```python
parser.add_argument("--runtime-dir", default=None)
parser.add_argument("--allowed-webhook-cidrs", default=None)
```

When building `PipelineRuntimeConfig`, add:

```python
runtime_dir=args.runtime_dir or os.getenv("CHATBOT_RUNTIME_DIR", "result/runtime"),
allowed_webhook_cidrs=_parse_csv_tuple(args.allowed_webhook_cidrs or os.getenv("KAKAO_ALLOWED_WEBHOOK_CIDRS")),
```

- [ ] **Step 3: Update docs**

In `docs/09_chatbot.md`:

- Add `--runtime-dir` / `CHATBOT_RUNTIME_DIR` to option table.
- Add `--allowed-webhook-cidrs` / `KAKAO_ALLOWED_WEBHOOK_CIDRS` to option table.
- Update Colab tips with Drive mount example path.
- Update remaining improvements list to mark implemented items and leave signature verification as future work.

- [ ] **Step 4: Run impacted tests**

```bash
pytest tests/test_chatbot_helpers.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/chatbot/kakao_colab_bot.py docs/09_chatbot.md tests/test_chatbot_helpers.py
git commit -m "Document chatbot operations options"
```

---

### Task 5: Final Verification and PR

**Files:**
- No code changes unless verification exposes a bug.

- [ ] **Step 1: Run required tests**

```bash
pytest tests/test_chatbot_helpers.py
pytest tests/test_pipeline_smoke.py
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: all pass and sample pipeline exits 0.

- [ ] **Step 2: Check git diff**

```bash
git status --short
git log --oneline -5
```

Expected: only intended files changed/committed; note unrelated untracked files if still present.

- [ ] **Step 3: Push and create PR**

```bash
git push
```

Then create PR with summary, test results, and docs path.
