# Signal Policy Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are explicitly disabled by AGENTS.md. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix signal policy inconsistencies from `docs/06_signal_policy.md`: prevent event boost double-application, align docs with code, and harden liquidity risk threshold behavior.

**Architecture:** Keep event boost calculation in `vectorized_event_signal_boost`, but make it idempotent for frames where `event_boost_score` already exists. Keep recommendation decisions based only on `predicted_return`. Documentation is updated to match current labels, thresholds, cumulative boost semantics, and implementation state.

**Tech Stack:** Python 3.10+, pandas, pytest, existing pipeline/domain modules.

---

### Task 1: Prevent event boost double-application

**Files:**
- Modify: `src/domain/signal_policy.py`
- Test: `tests/test_signal_policy_event_boost.py`

- [ ] **Step 1: Write failing test**

Create/modify `tests/test_signal_policy_event_boost.py` with:

```python
import pandas as pd
import pytest

from src.domain.signal_policy import build_prediction_policy_frame, vectorized_event_signal_boost


def _boostable_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": "AAA",
                "predicted_return": 1.0,
                "up_probability": 0.55,
                "uncertainty_score": 0.2,
                "confidence_score": 0.8,
                "signal_score": 0.40,
                "value_traded": 5_000_000_000.0,
                "min_liquidity_threshold": 3_000_000_000.0,
                "turnover_rank_daily": 1,
                "foreign_net_buy": 120_000_000_000.0,
                "institution_net_buy": 120_000_000_000.0,
                "nq_f_ret_1d": 0.012,
                "rsi_14": 45.0,
                "near_52w_high_flag": 1,
                "breakout_52w_flag": 0,
                "leader_confirmation_flag": 1,
            }
        ]
    )


def test_build_prediction_policy_frame_does_not_add_existing_event_boost_twice():
    boosted = vectorized_event_signal_boost(_boostable_frame())
    once_score = boosted.loc[0, "signal_score"]
    boost = boosted.loc[0, "event_boost_score"]

    finalized = build_prediction_policy_frame(boosted)

    assert boost > 0
    assert finalized.loc[0, "event_boost_score"] == pytest.approx(boost)
    assert finalized.loc[0, "signal_score"] == pytest.approx(once_score)
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_signal_policy_event_boost.py::test_build_prediction_policy_frame_does_not_add_existing_event_boost_twice -q`

Expected: FAIL because `signal_score` is incremented a second time.

- [ ] **Step 3: Implement minimal fix**

In `src/domain/signal_policy.py`, update `vectorized_event_signal_boost` so it recomputes `event_boost_score` but only adds the delta to `signal_score` when `event_boost_score` already exists:

```python
existing_boost = _to_numeric_series(out, "event_boost_score") if "event_boost_score" in out.columns else None
out["event_boost_score"] = event_boost.round(6)
if "signal_score" in out.columns:
    base_score = pd.to_numeric(out["signal_score"], errors="coerce").fillna(0.0)
    if existing_boost is not None:
        base_score = base_score - existing_boost.fillna(0.0)
    out["signal_score"] = base_score + out["event_boost_score"]
```

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_signal_policy_event_boost.py::test_build_prediction_policy_frame_does_not_add_existing_event_boost_twice -q`

Expected: PASS.

### Task 2: Harden LOW_LIQUIDITY threshold default at policy layer

**Files:**
- Modify: `src/domain/signal_policy.py`
- Test: `tests/test_signal_policy_event_boost.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_build_prediction_policy_frame_uses_default_liquidity_threshold_when_missing():
    frame = pd.DataFrame(
        [
            {
                "Symbol": "LOW",
                "predicted_return": 0.5,
                "up_probability": 0.55,
                "uncertainty_score": 0.2,
                "confidence_score": 0.8,
                "signal_score": 0.40,
                "value_traded": 1_000_000_000.0,
            }
        ]
    )

    out = build_prediction_policy_frame(frame)

    assert "LOW_LIQUIDITY" in out.loc[0, "risk_flag"]
```

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_signal_policy_event_boost.py::test_build_prediction_policy_frame_uses_default_liquidity_threshold_when_missing -q`

Expected: FAIL because missing `min_liquidity_threshold` defaults to 0.

- [ ] **Step 3: Implement minimal fix**

In `src/domain/signal_policy.py`, import `BacktestConfig` and use `BacktestConfig().min_value_traded` when `min_liquidity_threshold` is absent or non-positive:

```python
from src.config.settings import BacktestConfig, InvestmentCriteriaConfig

DEFAULT_MIN_LIQUIDITY_THRESHOLD = BacktestConfig().min_value_traded
```

Then update `risk_flag` threshold handling:

```python
min_liquidity = float(row.get("min_liquidity_threshold", 0) or 0)
if min_liquidity <= 0:
    min_liquidity = DEFAULT_MIN_LIQUIDITY_THRESHOLD
if float(row.get("value_traded", 0) or 0) < min_liquidity:
    flags.append("LOW_LIQUIDITY")
```

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_signal_policy_event_boost.py::test_build_prediction_policy_frame_uses_default_liquidity_threshold_when_missing -q`

Expected: PASS.

### Task 3: Update signal policy documentation

**Files:**
- Modify: `docs/06_signal_policy.md`

- [ ] **Step 1: Update confidence label docs**

Document four levels: `>=0.80 매우 높음`, `>=0.67 높음`, `>=0.34 보통`, else `낮음`.

- [ ] **Step 2: Update event boost docs**

Document that event boosts are cumulative/additive; top-3 turnover also receives top-turnover boost; high-conviction dual buy can stack with dual-buy and combined boosts. Document that `event_boost_score` is recalculated idempotently to avoid double application to `signal_score`.

- [ ] **Step 3: Update 52-week docs**

Document config-based threshold: `distance_to_52w_high <= near_52w_distance_threshold` with default `0.03` equivalent to close at least 97% of 52-week high.

- [ ] **Step 4: Update signal label docs**

Use labels: `strong_negative`, `weak_negative`, `neutral`, `weak_positive`, `strong_positive` and boundaries `0.25/0.45/0.55/0.75`.

- [ ] **Step 5: Update improvement section**

Mark fixed items as resolved and leave performance vectorization as future P2.

### Task 4: Verification

**Files:**
- No code changes unless tests reveal failures.

- [ ] **Step 1: Run impacted tests**

Run: `pytest tests/test_signal_policy_event_boost.py tests/test_console_summary.py tests/test_pipeline_smoke.py::test_build_scored_prediction_frame_keeps_signal_label_separate_from_confidence_context -q`

Expected: PASS.

- [ ] **Step 2: Run smoke test requested by AGENTS.md if feasible**

Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`

Expected: exits 0 and writes `result/pipeline_report_smoke.json` or the configured report path.

- [ ] **Step 3: Check git diff**

Run: `git diff -- src/domain/signal_policy.py tests/test_signal_policy_event_boost.py docs/06_signal_policy.md docs/superpowers/plans/2026-06-16-signal-policy-fixes.md`

Expected: only intended changes.
