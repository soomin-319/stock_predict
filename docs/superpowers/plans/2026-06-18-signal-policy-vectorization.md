# Signal Policy Vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are disabled by repository instruction; execute inline only.

**Goal:** Remove row-wise `apply` from `build_prediction_policy_frame` while preserving signal policy outputs and display-only context guardrails.

**Architecture:** Add vectorized dataframe helpers in `src/domain/signal_policy.py` for recommendations, confidence labels, risk/PM fields, jongbae score, and prediction reasons. Keep existing scalar helpers for compatibility, but route frame construction through vectorized helpers. Update `docs/06_signal_policy.md` to mark P2 complete and document the vectorized path.

**Tech Stack:** Python 3.10+, pandas, pytest.

---

### Task 1: Add equivalence tests before implementation

**Files:**
- Modify: `tests/test_signal_policy.py`

- [ ] Add a test comparing `build_prediction_policy_frame` outputs with scalar helpers on mixed rows.
- [ ] Add a test that missing/zero/NaN liquidity thresholds use `BacktestConfig().min_value_traded`.
- [ ] Run: `pytest tests/test_signal_policy.py -q`; expected first new vectorization coverage may pass for current behavior except NaN liquidity threshold should fail.

### Task 2: Vectorize policy frame internals

**Files:**
- Modify: `src/domain/signal_policy.py`

- [ ] Add `_recommendation_series`, `_confidence_label_series`, `_risk_flag_series`, `_position_size_hint_series`, `_pm_summary_frame`, `_jongbae_score_series`, `_prediction_reason_series`.
- [ ] Change `build_prediction_policy_frame` to use those helpers and remove row-wise `apply` there.
- [ ] Preserve recommendation guardrail: `recommendation` depends only on `predicted_return`.
- [ ] Preserve `news_impact_*` display-only behavior by not reading those columns.

### Task 3: Update documentation

**Files:**
- Modify: `docs/06_signal_policy.md`

- [ ] Replace the open P2 note with completed status.
- [ ] Document that `build_prediction_policy_frame` uses vectorized helpers and scalar functions remain compatibility wrappers.

### Task 4: Verify

**Commands:**
- `pytest tests/test_signal_policy.py tests/test_console_summary.py tests/test_news_impact_context.py tests/test_pipeline_smoke.py -q`
- `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`

### Task 5: Commit, push, PR

**Commands:**
- `git status --short`
- `git add src/domain/signal_policy.py tests/test_signal_policy.py docs/06_signal_policy.md docs/superpowers/plans/2026-06-18-signal-policy-vectorization.md`
- `git commit -m "Vectorize signal policy frame construction"`
- `git push`
- Open PR with summary and test results.
