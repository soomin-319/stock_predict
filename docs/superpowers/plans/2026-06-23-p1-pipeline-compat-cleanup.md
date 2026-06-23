# P1 Pipeline Compatibility Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Repository `AGENTS.md` forbids subagents, so do not use subagent-driven execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove a safe slice of `src/pipeline.py` test-only compatibility wrappers after moving tests to canonical owner modules.

**Architecture:** `pipeline.py` remains the orchestration entrypoint. Feature-column policy is tested through `src.features.price_features.select_feature_columns`; probability calibration is tested through `src.validation.support.calibrate_up_probability`. Runtime pipeline behavior and output contracts do not change.

**Tech Stack:** Python 3.10+, pandas, pytest, existing `src.pipeline`, `src.features.price_features`, and `src.validation.support` modules.

---

## File Map

- Modify: `tests/test_investor_features.py`
  - Replace `from src.pipeline import _feature_columns` with direct import of `select_feature_columns` from `src.features.price_features`.
  - Replace `_feature_columns(out)` call with `select_feature_columns(out)`.
- Modify: `tests/test_probability_calibration_guard.py`
  - Replace `from src.pipeline import _calibrate_up_probability` with direct import of `calibrate_up_probability` from `src.validation.support`.
  - Replace `_calibrate_up_probability(...)` calls with `calibrate_up_probability(...)`.
- Modify: `src/pipeline.py`
  - Remove `_feature_columns` wrapper.
  - Remove `_calibrate_up_probability` wrapper.
  - Keep canonical imports already used elsewhere if needed; remove only unused imports if lint/test discovery shows them unnecessary.
- Verify: targeted tests, impacted pipeline smoke, required sample pipeline, full pytest.

---

### Task 1: Move Feature Column Test to Canonical Module

**Files:**
- Modify: `tests/test_investor_features.py`
- Test: `tests/test_investor_features.py`

- [ ] **Step 1: Change the import and call site**

In `tests/test_investor_features.py`, replace:

```python
from src.features.price_features import DISPLAY_ONLY_CONTEXT_COLUMNS, build_features
from src.pipeline import _feature_columns
```

with:

```python
from src.features.price_features import DISPLAY_ONLY_CONTEXT_COLUMNS, build_features, select_feature_columns
```

Replace:

```python
feature_cols = _feature_columns(out)
```

with:

```python
feature_cols = select_feature_columns(out)
```

- [ ] **Step 2: Run the targeted investor feature tests**

Run:

```bash
pytest tests/test_investor_features.py -q
```

Expected: all tests pass. If a failure says `select_feature_columns` cannot be imported, stop and inspect `src/features/price_features.py` before changing production behavior.

- [ ] **Step 3: Commit the test import migration**

Run:

```bash
git add tests/test_investor_features.py
git commit -m "Point investor feature tests at feature module"
```

Expected: one focused commit containing only the test import/call migration.

---

### Task 2: Move Calibration Guard Test to Canonical Module

**Files:**
- Modify: `tests/test_probability_calibration_guard.py`
- Test: `tests/test_probability_calibration_guard.py`

- [ ] **Step 1: Change the import and call sites**

In `tests/test_probability_calibration_guard.py`, replace:

```python
from src.pipeline import _calibrate_up_probability
from src.validation.support import calibration_split_metrics, fit_up_probability_calibrator
```

with:

```python
from src.validation.support import (
    calibrate_up_probability,
    calibration_split_metrics,
    fit_up_probability_calibrator,
)
```

Replace both calls:

```python
_calibrate_up_probability(oof, raw)
_calibrate_up_probability(pd.DataFrame(), raw)
```

with:

```python
calibrate_up_probability(oof, raw)
calibrate_up_probability(pd.DataFrame(), raw)
```

- [ ] **Step 2: Run the targeted calibration tests**

Run:

```bash
pytest tests/test_probability_calibration_guard.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit the test import migration**

Run:

```bash
git add tests/test_probability_calibration_guard.py
git commit -m "Point calibration tests at validation module"
```

Expected: one focused commit containing only the calibration test import/call migration.

---

### Task 3: Remove Now-Unused Pipeline Wrappers

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_investor_features.py`, `tests/test_probability_calibration_guard.py`, `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Confirm no remaining references to wrappers**

Run:

```bash
Select-String -Path src/*.py,src/**/*.py,tests/*.py -Pattern '_feature_columns|_calibrate_up_probability'
```

Expected: references only in `src/pipeline.py` before deletion. If tests or runtime code still reference either wrapper, migrate those references to the canonical modules before deleting.

- [ ] **Step 2: Remove `_feature_columns` from `src/pipeline.py`**

Delete this function:

```python
def _feature_columns(df: pd.DataFrame) -> list[str]:
    return select_feature_columns(df)
```

Keep the `select_feature_columns` import because `run_pipeline()` still calls it directly or via existing feature-matrix logic. If it becomes unused, remove the import only after confirming no runtime reference remains.

- [ ] **Step 3: Remove `_calibrate_up_probability` from `src/pipeline.py`**

Delete this function:

```python
def _calibrate_up_probability(oof_df: pd.DataFrame, up_probs: pd.Series | pd.Index | list | tuple | pd.Series) -> pd.Series:
    return validation_calibrate_up_probability(oof_df, up_probs)
```

Keep `validation_calibrate_up_probability` import if another pipeline function calls it. If no reference remains, remove only that imported alias from the `src.validation.support` import block.

- [ ] **Step 4: Run impacted tests**

Run:

```bash
pytest tests/test_investor_features.py tests/test_probability_calibration_guard.py tests/test_pipeline_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit wrapper removal**

Run:

```bash
git add src/pipeline.py
git commit -m "Remove pipeline compatibility wrappers"
```

Expected: one focused production cleanup commit.

---

### Task 4: Required Verification and PR Prep

**Files:**
- Generated ignored artifact: `pipeline_report_smoke.json`

- [ ] **Step 1: Run broader impacted tests**

Run:

```bash
pytest tests/test_investor_features.py tests/test_probability_calibration_guard.py tests/test_fetch_real_fallback.py tests/test_news_impact_context.py tests/test_operational_hardening.py tests/test_pipeline_smoke.py -q
```

Expected: all tests pass. Warnings are acceptable if they match existing pandas FutureWarnings.

- [ ] **Step 2: Run required sample pipeline smoke command**

Run:

```bash
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit code 0 and console prints sample recommendations. Do not commit `pipeline_report_smoke.json`.

- [ ] **Step 3: Run full test suite**

Run:

```bash
pytest -q
```

Expected: full suite passes.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git status --short
git diff --stat origin/p1-kakao-job-store...HEAD
```

Expected: only design doc, plan doc, two test files, and `src/pipeline.py` changed in tracked commits.

- [ ] **Step 5: Push and create draft PR**

Run:

```bash
git push -u origin p1-pipeline-compat-cleanup
```

Then create a draft PR with:

- base: `p1-kakao-job-store`
- head: `p1-pipeline-compat-cleanup`
- title: `Clean up pipeline compatibility wrappers`
- summary: tests now import canonical feature/calibration helpers; removed two pipeline wrapper functions; no runtime behavior changes.
- tests: list targeted, impacted, sample pipeline, and full pytest results.

If push is rejected by approval policy because it exports branch contents to GitHub, ask the user for explicit approval before retrying.
