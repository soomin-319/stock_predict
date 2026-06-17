# Chatbot Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instruction forbids subagents, so use inline execution only.

**Goal:** Harden the Kakao chatbot webhook and request handling while preserving existing Colab/local defaults.

**Architecture:** Add opt-in secret validation at Flask boundary, lightweight in-process throttles in `KakaoColabPredictionBot`, strict symbol parsing, tolerant intent matching, and centralized Kakao response truncation.

**Tech Stack:** Python 3.10+, Flask test client, pytest, pandas-free helper tests where possible.

---

## File map
- Modify: `src/chatbot/intent.py` — normalize user text and match help/status by contained keywords.
- Modify: `src/chatbot/responses.py` — enforce Kakao simpleText length and type quick replies as tuples.
- Modify: `src/chatbot/kakao_colab_bot.py` — add webhook secret config, strict stock-code extraction, job concurrency/cooldown/dedup guards, CLI/env wiring.
- Modify: `tests/test_chatbot_helpers.py` — add failing tests before implementation.
- Modify: `docs/09_chatbot.md` — record implemented P0/P1 behavior and remaining P2 follow-up.

### Task 1: Intent and response helper tests

**Files:**
- Modify: `tests/test_chatbot_helpers.py`

- [ ] **Step 1: Write failing tests**

```python
def test_intents_match_phrases_and_punctuation():
    assert is_help_utterance("도움말 좀 알려줘!")
    assert is_status_utterance("결과 확인 부탁")


def test_simple_text_response_truncates_long_text():
    response = simple_text_response("가" * 1200, max_text_length=20)
    text = response["template"]["outputs"][0]["simpleText"]["text"]
    assert len(text) <= 20
    assert text.endswith("...(생략)")
```

- [ ] **Step 2: Verify red**
Run: `pytest tests/test_chatbot_helpers.py::test_intents_match_phrases_and_punctuation tests/test_chatbot_helpers.py::test_simple_text_response_truncates_long_text -q`
Expected: FAIL because phrase matching/truncation args are missing.

- [ ] **Step 3: Implement minimal code**
Update `intent.py` normalization and `responses.py` truncation.

- [ ] **Step 4: Verify green**
Run same pytest command. Expected: PASS.

### Task 2: Stock-code extraction tests

**Files:**
- Modify: `tests/test_chatbot_helpers.py`
- Modify: `src/chatbot/kakao_colab_bot.py`

- [ ] **Step 1: Write failing tests**

```python
from src.chatbot.kakao_colab_bot import KakaoColabPredictionBot


def test_extract_stock_code_rejects_noisy_numeric_tokens(tmp_path):
    bot = KakaoColabPredictionBot(result_simple_path=tmp_path / "missing.csv")
    assert bot._extract_stock_code("005930") == "005930"
    assert bot._extract_stock_code("005930.KS") == "005930.KS"
    assert bot._extract_stock_code("005930 보여줘") == "005930"
    assert bot._extract_stock_code("abc123") is None
    assert bot._extract_stock_code("12345") is None
```

- [ ] **Step 2: Verify red**
Run: `pytest tests/test_chatbot_helpers.py::test_extract_stock_code_rejects_noisy_numeric_tokens -q`
Expected: FAIL because `abc123` currently returns as stock code.

- [ ] **Step 3: Implement minimal code**
Remove first-token numeric fallback; keep regex search.

- [ ] **Step 4: Verify green**
Run same pytest command. Expected: PASS.

### Task 3: Webhook secret tests

**Files:**
- Modify: `tests/test_chatbot_helpers.py`
- Modify: `src/chatbot/kakao_colab_bot.py`

- [ ] **Step 1: Write failing tests**

```python
from src.chatbot.kakao_colab_bot import KakaoColabPredictionBot, PipelineRuntimeConfig, create_app


def test_kakao_webhook_rejects_missing_or_bad_secret(tmp_path):
    bot = KakaoColabPredictionBot(result_simple_path=tmp_path / "missing.csv")
    app = create_app(bot=bot, runtime_config=PipelineRuntimeConfig(kakao_webhook_secret="secret"))
    client = app.test_client()
    assert client.post("/kakao/webhook", json={}).status_code == 401
    assert client.post("/kakao/webhook", json={}, headers={"X-Webhook-Secret": "bad"}).status_code == 401


def test_kakao_webhook_accepts_matching_secret(tmp_path):
    bot = KakaoColabPredictionBot(result_simple_path=tmp_path / "missing.csv")
    app = create_app(bot=bot, runtime_config=PipelineRuntimeConfig(kakao_webhook_secret="secret"))
    client = app.test_client()
    response = client.post("/kakao/webhook", json={}, headers={"X-Webhook-Secret": "secret"})
    assert response.status_code == 200
```

- [ ] **Step 2: Verify red**
Run: `pytest tests/test_chatbot_helpers.py::test_kakao_webhook_rejects_missing_or_bad_secret tests/test_chatbot_helpers.py::test_kakao_webhook_accepts_matching_secret -q`
Expected: FAIL because config field/auth check does not exist.

- [ ] **Step 3: Implement minimal code**
Add `kakao_webhook_secret` to config, constant-time header check in route, CLI/env wiring.

- [ ] **Step 4: Verify green**
Run same pytest command. Expected: PASS.

### Task 4: Job guard tests

**Files:**
- Modify: `tests/test_chatbot_helpers.py`
- Modify: `src/chatbot/kakao_colab_bot.py`

- [ ] **Step 1: Write failing tests**

```python
class _DummyProcess:
    pid = 123
    stdout = None
    def wait(self):
        return 0


def test_start_prediction_job_deduplicates_running_symbol(tmp_path):
    calls = []
    def runner(*args, **kwargs):
        calls.append(args)
        return _DummyProcess()
    bot = KakaoColabPredictionBot(result_simple_path=tmp_path / "missing.csv", process_runner=runner)
    assert bot._start_prediction_job("005930.KS") is True
    assert bot._start_prediction_job("005930.KS") is True
    assert len(calls) == 1


def test_start_prediction_job_respects_concurrency_limit(tmp_path):
    def runner(*args, **kwargs):
        return _DummyProcess()
    cfg = PipelineRuntimeConfig(max_concurrent_prediction_jobs=1)
    bot = KakaoColabPredictionBot(runtime_config=cfg, result_simple_path=tmp_path / "missing.csv", process_runner=runner)
    assert bot._start_prediction_job("005930.KS") is True
    assert bot._start_prediction_job("000660.KS") is False
```

- [ ] **Step 2: Verify red**
Run: `pytest tests/test_chatbot_helpers.py::test_start_prediction_job_deduplicates_running_symbol tests/test_chatbot_helpers.py::test_start_prediction_job_respects_concurrency_limit -q`
Expected: FAIL because dedup/concurrency fields are missing.

- [ ] **Step 3: Implement minimal code**
Add config fields, helper to count active/running jobs, and early returns before subprocess spawn.

- [ ] **Step 4: Verify green**
Run same pytest command. Expected: PASS.

### Task 5: Docs and full verification

**Files:**
- Modify: `docs/09_chatbot.md`

- [ ] **Step 1: Update docs**
Document new env/CLI options, webhook header, strict code parsing, job guard behavior, and remaining P2 Drive persistence follow-up.

- [ ] **Step 2: Run targeted tests**
Run: `pytest tests/test_chatbot_helpers.py -q`
Expected: PASS.

- [ ] **Step 3: Run smoke test**
Run: `pytest tests/test_pipeline_smoke.py -q`
Expected: PASS.

- [ ] **Step 4: Run sample pipeline**
Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`
Expected: exit code 0 and report under `result/` or documented generated path.

- [ ] **Step 5: Commit, push, PR**
Run `git status`, commit focused changes, push branch, create PR with summary/test results.
