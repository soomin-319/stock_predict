# Operational Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix operational isolation, timeout duplication, context observability, and fail-fast validation issues.

**Architecture:** Keep fixes at existing boundaries. Add validation helpers in config/model modules, structured coverage metadata in investor context and pipeline preparation, and bot-owned keyed futures for timeout single-flight behavior.

**Tech Stack:** Python 3.10+, pandas, concurrent.futures, pytest

---

### Task 1: Restore pytest temp isolation

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_operational_hardening.py`

- [ ] Add regression test proving global tempfile directory is not repository shared temp.
- [ ] Run focused test and confirm failure.
- [ ] Remove global temp environment and `tempfile.tempdir` mutation.
- [ ] Run focused test and confirm pass.

### Task 2: Add config and quantile fail-fast validation

**Files:**
- Modify: `src/config/settings.py`
- Modify: `src/models/lgbm_heads.py`
- Test: `tests/test_operational_hardening.py`

- [ ] Add tests for unknown config keys, invalid ranges, and invalid quantiles.
- [ ] Run focused tests and confirm failures.
- [ ] Add path-aware config merge validation and model quantile validation.
- [ ] Run focused tests and confirm passes.

### Task 3: Add context failure observability

**Files:**
- Modify: `src/data/investor_context.py`
- Modify: `src/pipeline.py`
- Test: `tests/test_operational_hardening.py`

- [ ] Add tests distinguishing no events from collection failure.
- [ ] Run focused tests and confirm failures.
- [ ] Add structured failure metadata without affecting prediction inputs.
- [ ] Run focused tests and display-only guard tests.

### Task 4: Prevent duplicate timed chatbot work

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py`
- Test: `tests/test_kakao_colab_bot.py`

- [ ] Add tests proving same keyed timed operation runs once and timeout does not spawn duplicate summary.
- [ ] Run focused tests and confirm failures.
- [ ] Add bot-owned keyed future tracking and reuse.
- [ ] Run focused tests and confirm passes.

### Task 5: Verify

- [ ] Run impacted tests.
- [ ] Run full `pytest`.
- [ ] Run `python -m compileall src`.
- [ ] Run sample pipeline smoke command.
