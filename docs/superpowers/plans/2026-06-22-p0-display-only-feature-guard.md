# P0 Display-only Feature Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not use subagents in this repository.

**Goal:** Add pattern-based safeguards so future news/disclosure/impact context columns cannot enter model feature selection.

**Architecture:** `src.features.feature_selection` remains the single source of truth for model feature selection. Add one small predicate for display-only context names, use it in feature selection and display context discovery, and cover it with deterministic pytest tests.

**Tech Stack:** Python 3.10+, pandas, pytest.

---

## File Structure

- Modify: `src/features/feature_selection.py`
  - Add `is_display_only_context_column(column: str) -> bool`.
  - Expand display-only pattern exclusion beyond the current `news_impact_` prefix.
  - Route `select_feature_columns()` and `display_context_columns()` through the predicate.
- Modify: `tests/test_display_only_feature_guard.py`
  - Add a failing regression test proving pattern-like news/disclosure/impact columns are excluded even when not explicitly listed.
- Verification only:
  - `tests/test_feature_module_boundaries.py`
  - `tests/test_pipeline_smoke.py`

---

### Task 1: Add failing pattern regression test

**Files:**
- Modify: `tests/test_display_only_feature_guard.py`

- [ ] **Step 1: Add the failing test**

Append this test after `test_select_feature_columns_excludes_any_news_impact_prefixed_column_even_missing_flags`:

```python
def test_select_feature_columns_excludes_display_context_name_patterns():
    df = pd.DataFrame(
        {
            "daily_return": [0.01],
            "ret_1": [0.02],
            "news_sentiment_raw": [0.9],
            "news_sentiment_raw_missing": [0.0],
            "latest_news_headline": ["수주 확대"],
            "latest_news_headline_missing": [0.0],
            "foo_impact_score": [95.0],
            "foo_impact_score_missing": [0.0],
            "disclosure_impact_label": ["positive"],
            "disclosure_impact_label_missing": [0.0],
            "disclosure_event_summary": ["공급계약"],
            "disclosure_event_summary_missing": [0.0],
        }
    )

    selected = select_feature_columns(df)

    assert selected == ["daily_return", "ret_1"]
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
pytest tests/test_display_only_feature_guard.py::test_select_feature_columns_excludes_display_context_name_patterns -q
```

Expected before implementation: FAIL because one or more `*_missing` context columns are selected.

---

### Task 2: Implement display-only context predicate

**Files:**
- Modify: `src/features/feature_selection.py`

- [ ] **Step 1: Replace the narrow prefix constant with broader patterns**

Change:

```python
DISPLAY_ONLY_CONTEXT_PREFIXES = ("news_impact_",)
```

to:

```python
DISPLAY_ONLY_CONTEXT_PREFIXES = ("news_", "disclosure_",)
DISPLAY_ONLY_CONTEXT_SUBSTRINGS = ("_news_", "_impact_",)
```

- [ ] **Step 2: Add the predicate**

Insert after `MODEL_FEATURE_COLUMN_BASE = FEATURE_COLUMN_BASE - DISPLAY_ONLY_CONTEXT_COLUMNS`:

```python
def is_display_only_context_column(column: str) -> bool:
    return (
        column in DISPLAY_ONLY_CONTEXT_COLUMNS
        or column.startswith(DISPLAY_ONLY_CONTEXT_PREFIXES)
        or any(pattern in column for pattern in DISPLAY_ONLY_CONTEXT_SUBSTRINGS)
    )
```

- [ ] **Step 3: Route `select_feature_columns()` through the predicate**

Replace the function body with:

```python
def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if not is_display_only_context_column(c)
        and (c.startswith(FEATURE_COLUMN_PREFIXES) or c.endswith("_missing") or c in MODEL_FEATURE_COLUMN_BASE)
    ]
```

- [ ] **Step 4: Route `display_context_columns()` through the predicate**

Replace the function body with:

```python
def display_context_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_display_only_context_column(c)]
```

---

### Task 3: Verify guard tests

**Files:**
- Test: `tests/test_display_only_feature_guard.py`
- Test: `tests/test_feature_module_boundaries.py`

- [ ] **Step 1: Run display-only guard tests**

Run:

```bash
pytest tests/test_display_only_feature_guard.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run module-boundary tests**

Run:

```bash
pytest tests/test_feature_module_boundaries.py -q
```

Expected: all tests pass.

---

### Task 4: Run smoke verification

**Files:**
- Test: `tests/test_pipeline_smoke.py`
- Generated output: `result/pipeline_report_smoke.json`

- [ ] **Step 1: Run pipeline smoke tests**

Run:

```bash
pytest tests/test_pipeline_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run sample pipeline smoke command**

Run:

```bash
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: command exits with code 0 and writes report output under `result/`.

---

### Task 5: Commit P0 implementation

**Files:**
- `src/features/feature_selection.py`
- `tests/test_display_only_feature_guard.py`
- `docs/superpowers/plans/2026-06-22-p0-display-only-feature-guard.md`

- [ ] **Step 1: Inspect changed files**

Run:

```bash
git status --short
git diff -- src/features/feature_selection.py tests/test_display_only_feature_guard.py docs/superpowers/plans/2026-06-22-p0-display-only-feature-guard.md
```

Expected: only the P0 predicate, test, and plan are changed.

- [ ] **Step 2: Commit**

Run:

```bash
git add src/features/feature_selection.py tests/test_display_only_feature_guard.py docs/superpowers/plans/2026-06-22-p0-display-only-feature-guard.md
git commit -m "Strengthen display-only feature guard"
```

Expected: commit succeeds.
