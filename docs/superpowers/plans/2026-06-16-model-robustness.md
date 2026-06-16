# Model Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Subagents are forbidden by repository instructions.

**Goal:** Make model prediction, validation, persistence, and documentation robust against quantile crossing, unsafe missing-value handling, training-window mismatch, and guardrail regressions.

**Architecture:** Keep changes localized to model, inference, validation, pipeline artifact, config, and docs layers. Add tests first for each behavior, then implement minimal production changes. Preserve `predicted_return`-only recommendation policy and display-only news/disclosure context.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, LightGBM/sklearn-compatible model heads, joblib persistence.

---

### Task 1: Quantile monotonicity and uncertainty clipping

**Files:**
- Modify: `tests/test_lgbm_heads.py`
- Modify: `tests/test_inference_predict.py`
- Modify: `src/models/lgbm_heads.py`
- Modify: `src/inference/predict.py`

- [ ] Write failing tests proving crossed quantile head outputs are sorted row-wise and `uncertainty_width` is never negative.
- [ ] Run targeted tests and confirm failure from current crossing behavior.
- [ ] Sort selected quantile predictions row-wise in `MultiHeadStockModel.predict()` and clip uncertainty width in `build_prediction_frame()`.
- [ ] Re-run targeted tests and confirm pass.

### Task 2: Training imputer and model artifact compatibility

**Files:**
- Modify: `tests/test_lgbm_heads.py`
- Modify: `src/models/lgbm_heads.py`

- [ ] Write failing tests proving `predict()` uses stored medians/neutral values instead of zero and save/load preserves imputer values.
- [ ] Run targeted tests and confirm failure.
- [ ] Add `_feature_imputer_values`, training-time computation, prediction-time imputation, metadata/payload persistence, and artifact version bump.
- [ ] Re-run targeted tests and confirm pass.

### Task 3: Fit validation guardrails

**Files:**
- Modify: `tests/test_lgbm_heads.py`
- Modify: `src/models/lgbm_heads.py`

- [ ] Write failing tests for empty feature columns, no usable rows, and single-class classifier target.
- [ ] Run targeted tests and confirm failure.
- [ ] Add clear `ValueError` checks in `fit()`.
- [ ] Re-run targeted tests and confirm pass.

### Task 4: Walk-forward rolling lookback option

**Files:**
- Modify: `tests/test_walk_forward.py` or create nearest validation test file
- Modify: `src/config/settings.py`
- Modify: `src/validation/walk_forward.py`
- Modify: `docs/04_model.md`

- [ ] Write failing test proving `walk_forward_lookback_days=60` limits fold `train_start` to recent training dates.
- [ ] Run targeted test and confirm failure.
- [ ] Add config field/validation and filter train folds when option is positive.
- [ ] Re-run targeted test and confirm pass.

### Task 5: Final model artifact and feature importance export

**Files:**
- Modify: relevant pipeline tests
- Modify: `src/models/lgbm_heads.py`
- Modify: `src/pipeline.py`

- [ ] Write failing tests for feature-importance frame shape and pipeline report artifact fields.
- [ ] Run targeted tests and confirm failure.
- [ ] Add `feature_importance_frame()` to model and save model/importances in `_predict_pipeline_latest()` through `ArtifactManager`.
- [ ] Re-run targeted tests and confirm pass.

### Task 6: Policy/news-disclosure guardrails and docs

**Files:**
- Modify: existing policy/feature tests
- Modify: `docs/04_model.md`

- [ ] Write or extend tests proving display-only columns are excluded from model features and recommendation ignores non-return inputs.
- [ ] Run targeted tests and confirm pass/fail as appropriate.
- [ ] Update `docs/04_model.md` with implemented behavior and future-work notes.
- [ ] Run docs-related tests if present.

### Task 7: Full verification and publish

**Files:**
- No production files beyond prior tasks.

- [ ] Run impacted tests.
- [ ] Run full `pytest`.
- [ ] Run sample pipeline: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`.
- [ ] Inspect `git diff` and avoid unrelated user changes.
- [ ] Commit only intended files.
- [ ] Push branch and open draft PR if GitHub authentication/remotes allow.
