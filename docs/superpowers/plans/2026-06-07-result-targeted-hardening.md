# Result Targeted Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** result 소비·승격·정리 경로의 확인된 안전 공백을 최소 변경으로 보완한다.

**Architecture:** 기존 manifest와 실행별 산출물 구조를 유지한다. 소비자는 검증된 운영 manifest 또는 운영 메타데이터가 있는 legacy CSV만 사용하고, 생산자와 cleanup은 허용 경로·상태를 명시적으로 검증한다.

**Tech Stack:** Python 3.10+, pathlib, pandas, pytest

---

### Task 1: Artifact Path and Promotion Safety

**Files:**
- Modify: `src/reports/run_artifacts.py`
- Test: `tests/test_run_artifacts.py`

- [ ] **Step 1: Write failing tests**

Add tests proving `manager.path("../outside.json")`, absolute paths, and unknown statuses are rejected or not promoted.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_run_artifacts.py -q`

- [ ] **Step 3: Implement minimal safety checks**

Resolve requested paths under `run_dir`, reject paths outside it, and promote only `pass` or `warning`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_run_artifacts.py -q`

### Task 2: Safe Chatbot and Colab Result Consumption

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`
- Modify: `colab/stock_predict_colab.py`
- Test: `tests/test_kakao_colab_bot.py`
- Test: `tests/test_colab_runner.py`

- [ ] **Step 1: Write failing chatbot and Colab tests**

Add tests proving failed/unpromoted latest manifests and unsafe legacy CSVs are blocked, while Colab returns current run artifact paths.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_kakao_colab_bot.py tests/test_colab_runner.py -q`

- [ ] **Step 3: Implement validated consumers**

Validate latest manifest status/promotion and legacy CSV metadata. Use the returned pipeline manifest to resolve Colab outputs.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_kakao_colab_bot.py tests/test_colab_runner.py -q`

### Task 3: Test Artifact Cleanup

**Files:**
- Modify: `src/utils/result_cleanup.py`
- Test: `tests/test_result_cleanup.py`

- [ ] **Step 1: Write failing cleanup tests**

Add tests proving expired `result/test/` children are removed while `latest/`, runtime JSON, and outside paths remain.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_result_cleanup.py -q`

- [ ] **Step 3: Implement test-root cleanup**

Delete only children directly validated under `result/test/`, using the failed-run retention period.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_result_cleanup.py -q`

### Task 4: Documentation and Final Verification

**Files:**
- Modify: `docs/RESULT_IMPROVEMENT_PROGRESS_2026-06-07.md`
- Modify: `docs/RESULT_CLEANUP.md`

- [ ] **Step 1: Update documentation**

Remove contradictory chatbot notes, document strict legacy fallback and test cleanup, and record fresh verification counts.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_run_artifacts.py tests/test_kakao_colab_bot.py tests/test_colab_runner.py tests/test_result_cleanup.py -q`

- [ ] **Step 3: Run full suite**

Run: `pytest -q`

- [ ] **Step 4: Run sample smoke**

Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`

- [ ] **Step 5: Review and publish**

Review the scoped diff, commit only targeted-hardening files, push, and update/create the draft PR.
