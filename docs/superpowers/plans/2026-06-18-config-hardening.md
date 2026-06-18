# Config Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instructions prohibit subagents.

**Goal:** Harden AppConfig validation, improve config error messages, and restore the config documentation so settings behavior is accurate and readable.

**Architecture:** Keep all config schema and validation behavior in `src/config/settings.py`. Add deterministic pytest coverage in `tests/test_operational_hardening.py`. Replace corrupted `docs/10_config.md` with a UTF-8 Korean reference that reflects current code and guardrails.

**Tech Stack:** Python 3.10+, dataclasses, pytest, Markdown.

---

### Task 1: Strengthen unknown-key errors

**Files:**
- Modify: `src/config/settings.py`
- Test: `tests/test_operational_hardening.py`

- [ ] **Step 1: Write failing tests**

Add tests that unknown config keys include a close suggestion when available and omit a suggestion when none exists.

- [ ] **Step 2: Run targeted tests and verify RED**

Run: `pytest tests/test_operational_hardening.py::test_config_unknown_key_suggests_close_match tests/test_operational_hardening.py::test_config_unknown_key_without_close_match_has_plain_error -q`
Expected: FAIL because message has no suggestion support yet.

- [ ] **Step 3: Implement minimal suggestion helper**

Use `difflib.get_close_matches` in `_merge_dataclass_config` to append `; did you mean '<field>'?` for close matches.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run same targeted command. Expected: PASS.

### Task 2: Strengthen validation ranges and cross-field checks

**Files:**
- Modify: `src/config/settings.py`
- Test: `tests/test_operational_hardening.py`

- [ ] **Step 1: Write failing tests**

Add invalid config cases for:
- `training.min_train_size <= training.test_size`
- `training.step_size > training.test_size`
- non-positive/unsorted feature windows
- signal negative weights and zero primary weight sum
- investment criteria ordering/range mistakes
- non-negative bps checks and positive multipliers
- `backtest.max_positions_per_market_type < 1`

- [ ] **Step 2: Run targeted tests and verify RED**

Run: `pytest tests/test_operational_hardening.py::test_config_rejects_invalid_ranges -q`
Expected: FAIL for newly uncovered invalid cases.

- [ ] **Step 3: Implement minimal validation helpers**

Add helpers for sorted positive int lists, non-negative numbers, and bounded ratios. Extend `_validate_app_config` with training, feature, signal, investment criteria, and backtest constraints.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run same targeted command. Expected: PASS.

### Task 3: Add schema version to serialized config

**Files:**
- Modify: `src/config/settings.py`
- Test: `tests/test_operational_hardening.py`

- [ ] **Step 1: Write failing test**

Assert `app_config_to_dict(load_app_config())` contains `config_schema_version == 1`.

- [ ] **Step 2: Run targeted test and verify RED**

Run: `pytest tests/test_operational_hardening.py::test_app_config_to_dict_includes_schema_version -q`
Expected: FAIL because schema version does not exist yet.

- [ ] **Step 3: Implement minimal schema version**

Add `config_schema_version: int = 1` to `AppConfig`. Keep it in exported dict. Validate it is exactly `1` for now.

- [ ] **Step 4: Run targeted test and verify GREEN**

Run same targeted command. Expected: PASS.

### Task 4: Repair config documentation

**Files:**
- Modify: `docs/10_config.md`

- [ ] **Step 1: Replace mojibake document**

Write a clean UTF-8 Korean config guide covering AppConfig sections, loading precedence, env vars, CLI mappings, schema version, validation rules, and research-only/news-display-only guardrails.

- [ ] **Step 2: Inspect document encoding and key text**

Run: `python -c "from pathlib import Path; p=Path('docs/10_config.md'); s=p.read_text(encoding='utf-8'); assert '환경변수' in s and 'config_schema_version' in s and 'predicted_return' in s"`
Expected: exit 0.

### Task 5: Full verification and handoff artifacts

**Files:**
- Verify changed code and docs

- [ ] **Step 1: Run impacted tests**

Run: `pytest tests/test_operational_hardening.py -q`
Expected: PASS.

- [ ] **Step 2: Run smoke tests**

Run: `pytest tests/test_pipeline_smoke.py -q`
Expected: PASS.

- [ ] **Step 3: Run sample pipeline smoke command**

Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`
Expected: exit 0 and generated report under `result/` or configured output path per CLI defaults.

- [ ] **Step 4: Review diff**

Run: `git diff -- src/config/settings.py tests/test_operational_hardening.py docs/10_config.md docs/superpowers/plans/2026-06-18-config-hardening.md`
Expected: only intended changes.
