# Reports Contracts and Issue Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instructions prohibit subagents.

**Goal:** Harden report artifact contracts and cap/cache issue-summary LLM calls without allowing news/disclosures to change predictions.

**Architecture:** Keep report schema metadata near the existing report builders. Extend `RunArtifactManager` manifests with CSV column contracts. Add optional issue-summary LLM budget/cache controls in `src/reports/issue_summary.py` while retaining rule-based summaries for non-LLM rows.

**Tech Stack:** Python 3.10+, pandas, pytest, existing atomic file helpers and `FileLLMResponseCache`.

---

### Task 1: Manifest CSV schema metadata

**Files:**
- Modify: `src/reports/run_artifacts.py`
- Test: `tests/test_run_artifacts.py`

- [x] **Step 1: Write failing test**

Add a test asserting manifest CSV artifact entries expose `columns`, `schema_kind`, and `schema_version`.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_artifacts.py::test_manifest_csv_artifacts_include_schema_contract -q`

- [x] **Step 3: Implement minimal code**

Read CSV headers with `pd.read_csv(..., nrows=0, encoding="utf-8-sig")`, add `columns`, `schema_kind` for known outputs, and `schema_version`.

- [x] **Step 4: Run test to verify it passes**

Run same pytest command.

### Task 2: PM report schema validator

**Files:**
- Modify: `src/reports/pm_report.py`
- Test: `tests/test_report_metadata.py`

- [x] **Step 1: Write failing test**

Add a test for `validate_pm_report_schema()` to require metadata and report payload keys.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_metadata.py::test_pm_report_schema_validator_requires_contract_fields -q`

- [x] **Step 3: Implement minimal code**

Add `PM_REPORT_REQUIRED_FIELDS`, `validate_pm_report_schema()`, and export both.

- [x] **Step 4: Run test to verify it passes**

Run same pytest command.

### Task 3: Issue-summary LLM cap and cache

**Files:**
- Modify: `src/reports/issue_summary.py`
- Test: `tests/test_issue_summary.py`

- [x] **Step 1: Write failing tests**

Add tests that `max_llm_symbols=1` limits LLM calls and that repeated calls with a cache dir reuse cached summaries.

- [x] **Step 2: Run tests to verify they fail**

Run targeted pytest for new issue-summary tests.

- [x] **Step 3: Implement minimal code**

Add `max_llm_symbols`, `llm_cache_dir`, stable cache keys, and dataclass dict conversion.

- [x] **Step 4: Run tests to verify they pass**

Run targeted pytest.

### Task 4: Documentation and verification

**Files:**
- Modify: `docs/07_reports.md`

- [x] **Step 1: Update docs**

Replace stale improvement text with current state and remaining notes.

- [x] **Step 2: Run impacted tests**

Run report/issue-summary tests and smoke pipeline.

- [x] **Step 3: Commit, push, PR**

Follow repository handoff instructions.
