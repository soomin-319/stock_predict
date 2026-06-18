# Model Training Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution because repository instructions forbid subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable LightGBM early stopping/regularization and stronger sklearn fallback reporting.

**Architecture:** `TrainingConfig` owns knobs; `MultiHeadStockModel` owns backend params, optional eval-set fitting, metadata, and persistence. Pipeline copies model metadata warnings into diagnostics/report.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, LightGBM/sklearn-compatible model wrappers.

---

### Task 1: Model config and metadata tests

**Files:**
- Modify: `tests/test_lgbm_heads_persistence.py`

- [ ] Add tests for constructor fields, metadata fields, sklearn fallback warning, and fake-LightGBM eval kwargs.
- [ ] Run `pytest tests/test_lgbm_heads_persistence.py -q` and confirm new tests fail before implementation.

### Task 2: Implement model API

**Files:**
- Modify: `src/models/lgbm_heads.py`

- [ ] Add additive constructor fields: `early_stopping_rounds`, `reg_alpha`, `reg_lambda`, `min_child_samples`.
- [ ] Include regularization params in LightGBM params.
- [ ] Support `fit(..., eval_df=None)` with LightGBM early-stopping callbacks when enabled and eval rows exist.
- [ ] Persist/load metadata fields without bumping artifact version.
- [ ] Run `pytest tests/test_lgbm_heads_persistence.py -q`.

### Task 3: Config and pipeline warning

**Files:**
- Modify: `src/config/settings.py`
- Modify: `src/pipeline.py`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] Add `TrainingConfig` fields and validation.
- [ ] Add model metadata warnings to diagnostics before report creation.
- [ ] Add tests asserting report diagnostics include sklearn fallback warning.
- [ ] Run `pytest tests/test_pipeline_smoke.py::test_run_pipeline_generates_report_without_graph_artifacts -q`.

### Task 4: Docs and final verification

**Files:**
- Modify: `docs/04_model.md`

- [ ] Replace P1/P2 candidate text with implemented behavior and remaining caveat.
- [ ] Run `pytest tests/test_lgbm_heads_persistence.py tests/test_pipeline_smoke.py -q`.
- [ ] Run `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`.
- [ ] Commit, push, and open PR.
