# Remove Multi-Horizon Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove 5-day and 20-day prediction targets, model heads, and prediction/report outputs while retaining 5-day and 20-day input features.

**Architecture:** Keep the existing next-day regression, classification, and quantile heads as the sole prediction contract. Remove multi-horizon fields through feature generation, model inference, walk-forward validation, policy context, and reports. Set the default purge gap to one trading day because only the next-day target remains.

**Tech Stack:** Python, pandas, NumPy, pytest, LightGBM/sklearn fallback

---

### Task 1: Define the next-day-only contract

**Files:**
- Modify: `tests/test_pipeline_smoke.py`
- Modify: `tests/test_lgbm_heads_persistence.py`

- [ ] Change tests to require only next-day targets and predictions.
- [ ] Run focused tests and confirm they fail because multi-horizon behavior still exists.

### Task 2: Remove multi-horizon training and inference

**Files:**
- Modify: `src/features/price_features.py`
- Modify: `src/models/lgbm_heads.py`
- Modify: `src/inference/predict.py`
- Modify: `src/validation/support.py`
- Modify: `src/validation/walk_forward.py`
- Modify: `src/config/settings.py`

- [ ] Remove 5-day and 20-day target generation and model heads.
- [ ] Remove multi-horizon prediction propagation.
- [ ] Change the default purge gap to one trading day.

### Task 3: Remove multi-horizon policy and report fields

**Files:**
- Modify: `src/domain/signal_policy.py`
- Modify: `src/pipeline_support.py`
- Modify: `src/reports/output.py`
- Modify: `src/reports/pm_report.py`
- Modify: `src/reports/result_formatter.py`
- Modify: `src/pipeline.py`

- [ ] Remove horizon alignment, horizon-specific risk flags, and report columns.
- [ ] Run focused tests and update stale expectations.

### Task 4: Verify and publish

- [ ] Run the full pytest suite.
- [ ] Inspect the final diff for unrelated changes.
- [ ] Commit, push the current branch, and open a draft PR targeting `main`.
