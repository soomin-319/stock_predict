# Signal Tuning Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution because repository instructions forbid subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `tune_signal_weights` so it rewards stable validation performance instead of only top-decile log return.

**Architecture:** Keep public API and signal-score formula unchanged. Add internal objective helpers in `src/validation/signal_tuning.py` for rank IC, downside penalty, composite score, and default fallback when train/validation gap is excessive. Update docs to mark P2 addressed.

**Tech Stack:** Python 3.10+, pandas, pytest.

---

### Task 1: Add objective tests

**Files:**
- Modify: `tests/test_signal_tuning.py`

- [ ] Add a test that a candidate with better validation return but negative rank IC loses to a stable candidate.
- [ ] Add a test that excessive train/validation gap falls back to `DEFAULT_WEIGHTS`.
- [ ] Run `pytest tests/test_signal_tuning.py -q` and confirm new tests fail before production changes.

### Task 2: Implement composite objective

**Files:**
- Modify: `src/validation/signal_tuning.py`

- [ ] Add `_score_series`, `_top_decile_return`, `_rank_ic`, and `_objective_score` helpers.
- [ ] Rank candidates by composite validation objective, then simplicity tie-breaker.
- [ ] Add overfit guard: if selected gap exceeds threshold and default has acceptable objective, return default weights with diagnostics.
- [ ] Run `pytest tests/test_signal_tuning.py -q` and confirm pass.

### Task 3: Update validation docs

**Files:**
- Modify: `docs/05_validation.md`

- [ ] Replace remaining P2 proposal with implemented behavior and diagnostics.
- [ ] Run impacted tests and smoke pipeline.
