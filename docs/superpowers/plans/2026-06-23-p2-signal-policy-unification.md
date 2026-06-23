# P2 Signal Policy Row/Vectorized Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository `AGENTS.md` forbids subagents, so execute inline only.

**Goal:** Make vectorized signal policy helpers the single source of truth and convert scalar row helpers into 1-row adapters without changing policy outputs.

**Architecture:** Keep the pipeline on the existing vectorized path. Add tiny adapter helpers in `src/domain/signal_policy.py`, then rewrite scalar compatibility helpers to call vectorized helpers. Tests first lock exact scalar/vectorized equality and recommendation guardrails.

**Tech Stack:** Python 3.10+, pandas, pytest, existing `SignalConfig` and `InvestmentCriteriaConfig` dataclasses.

---

## Files

- Modify: `src/domain/signal_policy.py`
  - Add `_row_frame`, `_first_scalar`, `_first_record`.
  - Rewrite `risk_flag`, `prediction_reason`, `_jongbae_score`, `build_pm_summary_fields` as adapters.
  - Keep public APIs and labels unchanged.
- Modify: `tests/test_signal_policy.py`
  - Add RED tests proving scalar helpers delegate/equal vectorized helpers.
  - Add guardrail test proving recommendation unaffected by non-return fields and news-like columns.
- Run existing impacted tests:
  - `tests/test_signal_policy.py`
  - `tests/test_signal_policy_contract.py`
  - `tests/test_signal_policy_recommendation.py`
  - `tests/test_pipeline_smoke.py`

---

### Task 1: Add equality tests for scalar policy helpers

**Files:**
- Modify: `tests/test_signal_policy.py`

- [ ] **Step 1: Write failing tests**

Append these tests after `test_build_prediction_policy_frame_matches_scalar_policy_helpers`:

```python
def test_scalar_policy_helpers_match_vectorized_single_row_outputs():
    frame = _policy_input_frame()

    for idx, row in frame.iterrows():
        one_row = frame.loc[[idx]]
        assert risk_flag(row) == signal_policy._risk_flag_series(one_row).iloc[0]
        assert prediction_reason(row) == signal_policy._prediction_reason_series(one_row).iloc[0]
        assert signal_policy._jongbae_score(row) == signal_policy._jongbae_score_series(one_row).iloc[0]
        assert build_pm_summary_fields(row) == signal_policy._pm_summary_frame(one_row).iloc[0].to_dict()
```

Why this should fail before implementation: code may already produce equal values, but it does not prove adapter structure. If this passes, Task 2 adds structural RED tests.

- [ ] **Step 2: Run tests**

Run:

```powershell
pytest tests/test_signal_policy.py::test_scalar_policy_helpers_match_vectorized_single_row_outputs -q
```

Expected before implementation: may PASS because current duplicated logic is still equivalent. If PASS, continue to Task 2 for structural RED.

- [ ] **Step 3: Commit only if tests added and accepted as coverage**

Run:

```powershell
git add tests/test_signal_policy.py
git commit -m "Lock scalar and vectorized signal policy equivalence"
```

---

### Task 2: Add structural tests proving scalar helpers use vectorized source

**Files:**
- Modify: `tests/test_signal_policy.py`

- [ ] **Step 1: Write failing structural tests**

Append:

```python
def test_scalar_policy_helpers_are_vectorized_adapters():
    assert "_risk_flag_series" in inspect.getsource(risk_flag)
    assert "_prediction_reason_series" in inspect.getsource(prediction_reason)
    assert "_jongbae_score_series" in inspect.getsource(signal_policy._jongbae_score)
    assert "_pm_summary_frame" in inspect.getsource(build_pm_summary_fields)
```

Why this fails before implementation: scalar helpers currently contain independent row-wise logic, not vectorized adapter calls.

- [ ] **Step 2: Verify RED**

Run:

```powershell
pytest tests/test_signal_policy.py::test_scalar_policy_helpers_are_vectorized_adapters -q
```

Expected: FAIL, assertion missing vectorized helper names in at least `risk_flag`, `prediction_reason`, `_jongbae_score`, `build_pm_summary_fields`.

- [ ] **Step 3: Implement adapters minimally**

In `src/domain/signal_policy.py`, add after `_signal_cfg`:

```python
def _row_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([row.to_dict()], index=[row.name])


def _first_scalar(series: pd.Series):
    return series.iloc[0]


def _first_record(frame: pd.DataFrame) -> dict[str, str]:
    return frame.iloc[0].to_dict()
```

Replace `risk_flag(row)` body with:

```python
def risk_flag(row: pd.Series) -> str:
    return str(_first_scalar(_risk_flag_series(_row_frame(row))))
```

Replace `prediction_reason(row, cfg)` body with:

```python
def prediction_reason(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> str:
    return str(_first_scalar(_prediction_reason_series(_row_frame(row), cfg=cfg)))
```

Replace `_jongbae_score(row, cfg)` body with:

```python
def _jongbae_score(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> float:
    return float(_first_scalar(_jongbae_score_series(_row_frame(row), cfg=cfg)))
```

Replace `build_pm_summary_fields(row, cfg, signal_cfg)` body with:

```python
def build_pm_summary_fields(
    row: pd.Series,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> dict[str, str]:
    return _first_record(_pm_summary_frame(_row_frame(row), cfg=cfg, signal_cfg=signal_cfg))
```

Do not change `recommendation_from_signal`, `_recommendation_series`, or event boost math.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
pytest tests/test_signal_policy.py::test_scalar_policy_helpers_are_vectorized_adapters tests/test_signal_policy.py::test_scalar_policy_helpers_match_vectorized_single_row_outputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/domain/signal_policy.py tests/test_signal_policy.py
git commit -m "Unify scalar signal policy helpers"
```

---

### Task 3: Add missing-value adapter regression tests

**Files:**
- Modify: `tests/test_signal_policy.py`

- [ ] **Step 1: Write failing/guard tests**

Append:

```python
def test_scalar_adapters_preserve_missing_value_policy_defaults():
    row = pd.Series(
        {
            "predicted_return": np.nan,
            "confidence_score": np.nan,
            "coverage_gate_status": "",
            "uncertainty_score": np.nan,
            "up_probability": np.nan,
            "history_direction_accuracy": np.nan,
            "value_traded": 0,
            "min_liquidity_threshold": np.nan,
            "external_coverage_ratio": np.nan,
            "investor_coverage_ratio": np.nan,
            "market_headwind_score": np.nan,
            "turnover_rank_daily": np.nan,
            "foreign_net_buy": np.nan,
            "institution_net_buy": np.nan,
            "breakout_52w_flag": np.nan,
            "near_52w_high_flag": np.nan,
            "leader_confirmation_flag": np.nan,
            "nq_f_ret_1d": np.nan,
            "rsi_14": np.nan,
        },
        name="missing",
    )
    frame = pd.DataFrame([row.to_dict()], index=[row.name])

    assert risk_flag(row) == signal_policy._risk_flag_series(frame).iloc[0]
    assert prediction_reason(row) == signal_policy._prediction_reason_series(frame).iloc[0]
    assert signal_policy._jongbae_score(row) == signal_policy._jongbae_score_series(frame).iloc[0]
    assert build_pm_summary_fields(row) == signal_policy._pm_summary_frame(frame).iloc[0].to_dict()
```

Expected: PASS after Task 2. If FAIL, adapter changed NaN/default handling and production code must be fixed, not test weakened.

- [ ] **Step 2: Run test**

```powershell
pytest tests/test_signal_policy.py::test_scalar_adapters_preserve_missing_value_policy_defaults -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_signal_policy.py
git commit -m "Cover missing signal policy adapter defaults"
```

---

### Task 4: Add recommendation guardrail test around adapters

**Files:**
- Modify: `tests/test_signal_policy.py`

- [ ] **Step 1: Write test**

Append:

```python
def test_policy_adapter_recommendation_ignores_non_return_context_columns():
    base = pd.Series(
        {
            "predicted_return": 2.5,
            "signal_score": -999.0,
            "up_probability": 0.0,
            "uncertainty_score": 1.0,
            "news_impact_score": -1.0,
            "disclosure_impact_score": -1.0,
            "confidence_score": 0.9,
        }
    )
    changed = base.copy()
    changed["signal_score"] = 999.0
    changed["up_probability"] = 1.0
    changed["uncertainty_score"] = 0.0
    changed["news_impact_score"] = 1.0
    changed["disclosure_impact_score"] = 1.0

    assert build_pm_summary_fields(base)["recommendation"] == "매수"
    assert build_pm_summary_fields(changed)["recommendation"] == "매수"

    base_frame = pd.DataFrame([base.to_dict(), changed.to_dict()])
    out = build_prediction_policy_frame(base_frame)
    assert out["recommendation"].tolist() == ["매수", "매수"]
```

Expected: PASS. This guards safety after adapter rewrite.

- [ ] **Step 2: Run test**

```powershell
pytest tests/test_signal_policy.py::test_policy_adapter_recommendation_ignores_non_return_context_columns -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_signal_policy.py
git commit -m "Guard signal policy adapter recommendation inputs"
```

---

### Task 5: Impacted and full verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run impacted tests**

```powershell
pytest tests/test_signal_policy.py tests/test_signal_policy_contract.py tests/test_signal_policy_recommendation.py tests/test_pipeline_smoke.py -q
```

Expected: all pass.

- [ ] **Step 2: Run sample pipeline**

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit 0, report JSON written locally.

- [ ] **Step 3: Run full tests**

```powershell
pytest -q
```

Expected: all pass.

- [ ] **Step 4: Inspect git status and safety diff**

```powershell
git status --short
git diff p2-signal-config-thresholds..HEAD -- src/domain/signal_policy.py tests/test_signal_policy.py
```

Confirm:

- No generated result files staged.
- `recommendation_from_signal` unchanged except surrounding context if any.
- No news/disclosure columns used in recommendation logic.

---

### Task 6: Push and open PR

**Files:**
- No local source changes except committed work.

- [ ] **Step 1: Push branch**

```powershell
git push -u origin p2-signal-policy-unification
```

If approval is required for network/data transfer, ask user for explicit approval.

- [ ] **Step 2: Create draft PR**

Use GitHub connector. Base branch: `p2-signal-config-thresholds`. Head branch: `p2-signal-policy-unification`.

PR body must include:

- Summary of adapter unification.
- Guardrail note: recommendation remains `predicted_return` + `SignalConfig` thresholds only.
- Tests run and results.
- Artifact path: `pipeline_report_smoke.json` local generated smoke report, not committed.

---
