# Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instructions forbid subagents, so use inline execution only.

**Goal:** Implement all `docs/01_pipeline.md` pipeline hardening improvements with tests and docs.

**Architecture:** Keep changes localized to `src/pipeline.py`, `tests/test_pipeline_smoke.py`, and `docs/01_pipeline.md`. Preserve existing public report fields while adding clearer metadata. Contain only optional-stage failures; core failures still raise.

**Tech Stack:** Python 3.10+, pandas, pytest, existing config/report helpers.

---

## File Structure

- Modify: `src/pipeline.py`
  - Diagnostics stage status/warnings.
  - Tuned signal copy.
  - Adaptive walk-forward retry threshold.
  - Optional failure guards.
  - Report metadata additions.
- Modify: `tests/test_pipeline_smoke.py`
  - Unit tests for each behavior change.
- Modify: `docs/01_pipeline.md`
  - Implementation status and CLI exit codes.

---

### Task 1: Diagnostics stage status and coverage

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing diagnostics test**

Add a test importing `PipelineDiagnostics` and `PIPELINE_STAGE_KEYS`; assert `mark_stage`, `warn`, and `validate_stage_coverage` populate report fields.

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_pipeline_smoke.py::test_pipeline_diagnostics_records_stage_status_and_warnings -q`
Expected: FAIL because helpers/constant are missing.

- [ ] **Step 3: Implement diagnostics helpers**

Add `PIPELINE_STAGE_KEYS`, `stage_status`, `warnings`, `mark_stage`, `warn`, `validate_stage_coverage`, and include them in `to_report()`.

- [ ] **Step 4: Run passing test**

Run: `pytest tests/test_pipeline_smoke.py::test_pipeline_diagnostics_records_stage_status_and_warnings -q`
Expected: PASS.

### Task 2: Preserve input signal config

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing immutability test**

Patch validation dependencies to return deterministic OOF/tuning data, call `_run_pipeline_validation`, and assert `cfg.signal` equals its original values while returned `tuned_signal_config` has tuned values.

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_pipeline_smoke.py::test_pipeline_validation_does_not_mutate_signal_config -q`
Expected: FAIL because current code mutates `cfg.signal`.

- [ ] **Step 3: Implement tuned signal copy**

Import `replace` from `dataclasses`. Replace in-place assignments with `tuned_signal_cfg = replace(cfg.signal, **tuned)`. Use this copy in OOF rescoring and latest prediction.

- [ ] **Step 4: Run passing test**

Run: `pytest tests/test_pipeline_smoke.py::test_pipeline_validation_does_not_mutate_signal_config -q`
Expected: PASS.

### Task 3: Adaptive retry when fold count below threshold

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing retry test**

Patch `walk_forward_validate_result` to return one fold first and three folds second. Assert it is called twice and validation metadata reports adaptive retry.

- [ ] **Step 2: Run failing test**

Run: `pytest tests/test_pipeline_smoke.py::test_pipeline_validation_retries_when_fold_count_is_too_low -q`
Expected: FAIL because retry only happens at zero folds.

- [ ] **Step 3: Implement threshold retry**

Use `min_required_folds = int(getattr(cfg.training, "min_required_folds", 3) or 3)`. Retry when `len(folds) < min_required_folds`. Return `walk_forward_diagnostics` with initial/final fold counts and retry flag.

- [ ] **Step 4: Run passing test**

Run: `pytest tests/test_pipeline_smoke.py::test_pipeline_validation_retries_when_fold_count_is_too_low -q`
Expected: PASS.

### Task 4: Optional failure containment

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing optional failure tests**

Patch `add_external_market_features_with_coverage` to raise and assert `_build_pipeline_feature_matrix` returns price features plus failed coverage. Patch `append_issue_summary_columns` to raise and assert `_predict_pipeline_latest` keeps predictions and emits warning metadata where practical.

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_pipeline_smoke.py::test_external_feature_failure_degrades_to_price_features -q`
Expected: FAIL because exception propagates.

- [ ] **Step 3: Implement guards**

Wrap external feature add and display-only context append calls. Record warnings in coverage or DataFrame attrs; `run_pipeline` copies attrs to diagnostics warnings.

- [ ] **Step 4: Run passing tests**

Run: `pytest tests/test_pipeline_smoke.py::test_external_feature_failure_degrades_to_price_features -q`
Expected: PASS.

### Task 5: Report metadata and docs

**Files:**
- Modify: `src/pipeline.py`
- Modify: `docs/01_pipeline.md`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write/extend smoke assertions**

Assert pipeline report contains `config_input`, `signal_weights_tuned`, `walk_forward_diagnostics`, `diagnostics.stage_status`, and `diagnostics.warnings`.

- [ ] **Step 2: Implement report fields**

Add fields in `_write_pipeline_artifacts`; preserve existing `config` and `tuned_signal`.

- [ ] **Step 3: Update docs**

Mark improvement items as implemented and add CLI exit-code table: `0` success/warning report written; non-zero unexpected fatal error.

- [ ] **Step 4: Run full impacted tests**

Run: `pytest tests/test_pipeline_smoke.py -q`
Expected: PASS.

### Task 6: Smoke pipeline and git handoff

**Files:**
- Generated under `result/` only.

- [ ] **Step 1: Run sample pipeline**

Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`
Expected: Exit code 0 and report under `result/`.

- [ ] **Step 2: Run git status**

Run: `git status --short`
Expected: only intended source/docs/test changes plus generated result artifacts if not ignored.

- [ ] **Step 3: Commit, push, PR**

Commit focused changes, push branch, create PR per AGENTS.md if credentials/network allow.
