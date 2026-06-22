# P2 Signal Config Thresholds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not use subagents; AGENTS.md forbids subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make buy/sell recommendation thresholds configurable through `SignalConfig` and keep scalar/vectorized recommendation behavior equivalent.

**Architecture:** Add threshold fields and validation in `src/config/settings.py`, thread `SignalConfig` into policy-frame creation, and make scalar/vectorized recommendation helpers use the same thresholds. Keep recommendation guardrail unchanged: only `predicted_return` plus config thresholds decide `"매수"`, `"매도"`, `"관망"`.

**Tech Stack:** Python 3.10+, dataclasses, pandas, pytest.

---

## File Structure

- Modify `src/config/settings.py`
  - Add `SignalConfig.recommendation_buy_threshold_pct`.
  - Add `SignalConfig.recommendation_sell_threshold_pct`.
  - Validate type, sign, and order in `_validate_app_config`.
- Modify `src/domain/signal_policy.py`
  - Import `SignalConfig`.
  - Add default signal config resolver.
  - Add optional keyword-only `signal_cfg` to `recommendation_from_signal`.
  - Add optional `signal_cfg` to `_recommendation_series`, `_policy_recommendation`, `_pm_summary_frame`, `build_pm_summary_fields`, `build_prediction_policy_frame`.
- Modify `src/pipeline_support.py`
  - Add optional `signal_cfg` to `finalize_latest_prediction_frame`.
  - Pass it into `build_prediction_policy_frame`.
- Modify `src/pipeline.py`
  - Pass `effective_signal_config` to `finalize_latest_prediction_frame`.
- Modify tests:
  - `tests/test_operational_hardening.py`
  - `tests/test_signal_policy_recommendation.py`
  - `tests/test_signal_policy.py`
  - `tests/test_pipeline_smoke.py`

---

### Task 1: Config fields and validation

**Files:**
- Modify: `tests/test_operational_hardening.py`
- Modify: `src/config/settings.py`

- [ ] **Step 1: Write failing config tests**

Append to `tests/test_operational_hardening.py` after `test_app_config_to_dict_includes_schema_version`:

```python
def test_signal_config_exposes_recommendation_thresholds():
    cfg = load_app_config(
        overrides={
            "signal": {
                "recommendation_buy_threshold_pct": 3.5,
                "recommendation_sell_threshold_pct": -1.5,
            }
        }
    )

    signal_dict = app_config_to_dict(cfg)["signal"]
    assert signal_dict["recommendation_buy_threshold_pct"] == 3.5
    assert signal_dict["recommendation_sell_threshold_pct"] == -1.5


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"signal": {"recommendation_buy_threshold_pct": 0.0}}, "signal.recommendation_buy_threshold_pct"),
        ({"signal": {"recommendation_buy_threshold_pct": -0.1}}, "signal.recommendation_buy_threshold_pct"),
        ({"signal": {"recommendation_buy_threshold_pct": True}}, "signal.recommendation_buy_threshold_pct"),
        ({"signal": {"recommendation_sell_threshold_pct": 0.0}}, "signal.recommendation_sell_threshold_pct"),
        ({"signal": {"recommendation_sell_threshold_pct": 0.1}}, "signal.recommendation_sell_threshold_pct"),
        ({"signal": {"recommendation_sell_threshold_pct": False}}, "signal.recommendation_sell_threshold_pct"),
        (
            {
                "signal": {
                    "recommendation_buy_threshold_pct": 1.0,
                    "recommendation_sell_threshold_pct": 1.0,
                }
            },
            "signal.recommendation_sell_threshold_pct must be less than signal.recommendation_buy_threshold_pct",
        ),
    ],
)
def test_signal_config_rejects_invalid_recommendation_thresholds(overrides, expected):
    with pytest.raises(ValueError, match=expected):
        load_app_config(overrides=overrides)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_operational_hardening.py::test_signal_config_exposes_recommendation_thresholds tests/test_operational_hardening.py::test_signal_config_rejects_invalid_recommendation_thresholds -q
```

Expected: fail with `Unknown configuration key: signal.recommendation_buy_threshold_pct`.

- [ ] **Step 3: Add config fields and validation**

In `src/config/settings.py`, change `SignalConfig` to:

```python
@dataclass
class SignalConfig:
    return_weight: float = 0.65
    up_prob_weight: float = 0.35
    uncertainty_penalty: float = 0.25
    recommendation_buy_threshold_pct: float = 2.0
    recommendation_sell_threshold_pct: float = -2.0
```

Add helper near `_validate_number`:

```python
def _validate_negative_number(value, path: str) -> None:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0
    if not valid:
        raise ValueError(f"{path} must be a negative number, got {value!r}")
```

In `_validate_app_config`, after current signal weight validation and primary weight sum check, add:

```python
    _validate_number(cfg.signal.recommendation_buy_threshold_pct, "signal.recommendation_buy_threshold_pct")
    _validate_negative_number(
        cfg.signal.recommendation_sell_threshold_pct,
        "signal.recommendation_sell_threshold_pct",
    )
    if cfg.signal.recommendation_sell_threshold_pct >= cfg.signal.recommendation_buy_threshold_pct:
        raise ValueError(
            "signal.recommendation_sell_threshold_pct must be less than "
            "signal.recommendation_buy_threshold_pct"
        )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
pytest tests/test_operational_hardening.py::test_signal_config_exposes_recommendation_thresholds tests/test_operational_hardening.py::test_signal_config_rejects_invalid_recommendation_thresholds -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/config/settings.py tests/test_operational_hardening.py
git commit -m "Add configurable recommendation thresholds"
```

---

### Task 2: Scalar recommendation uses SignalConfig

**Files:**
- Modify: `tests/test_signal_policy_recommendation.py`
- Modify: `src/domain/signal_policy.py`

- [ ] **Step 1: Write failing scalar tests**

Append to `tests/test_signal_policy_recommendation.py`:

```python
from src.config.settings import SignalConfig
```

If import placement creates duplicate imports, keep one import at top.

Append tests:

```python
def test_recommendation_uses_custom_signal_config_thresholds():
    cfg = SignalConfig(
        recommendation_buy_threshold_pct=3.0,
        recommendation_sell_threshold_pct=-1.0,
    )

    assert recommendation_from_signal(None, 2.5, signal_cfg=cfg) == "관망"
    assert recommendation_from_signal(None, 3.1, signal_cfg=cfg) == "매수"
    assert recommendation_from_signal(None, -1.0, signal_cfg=cfg) == "매도"
    assert recommendation_from_signal(None, -0.9, signal_cfg=cfg) == "관망"


def test_recommendation_custom_thresholds_still_ignore_non_return_inputs():
    cfg = SignalConfig(
        recommendation_buy_threshold_pct=3.0,
        recommendation_sell_threshold_pct=-1.0,
    )

    assert recommendation_from_signal(1.0, 2.5, up_probability=1.0, uncertainty_score=0.0, signal_cfg=cfg) == "관망"
    assert recommendation_from_signal(0.0, 3.1, up_probability=0.0, uncertainty_score=1.0, signal_cfg=cfg) == "매수"
    assert recommendation_from_signal(1.0, -1.0, up_probability=1.0, uncertainty_score=0.0, signal_cfg=cfg) == "매도"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
pytest tests/test_signal_policy_recommendation.py::test_recommendation_uses_custom_signal_config_thresholds tests/test_signal_policy_recommendation.py::test_recommendation_custom_thresholds_still_ignore_non_return_inputs -q
```

Expected: fail with `TypeError: recommendation_from_signal() got an unexpected keyword argument 'signal_cfg'`.

- [ ] **Step 3: Implement scalar config support**

In `src/domain/signal_policy.py`, change import:

```python
from src.config.settings import BacktestConfig, InvestmentCriteriaConfig, SignalConfig
```

Add near `DEFAULT_CRITERIA`:

```python
DEFAULT_SIGNAL_CONFIG = SignalConfig()
```

Add helper after `_criteria`:

```python
def _signal_cfg(cfg: SignalConfig | None) -> SignalConfig:
    return cfg or DEFAULT_SIGNAL_CONFIG
```

Change `recommendation_from_signal` signature and threshold logic:

```python
def recommendation_from_signal(
    signal_score: float | int | None,
    predicted_return: float | int | None,
    up_probability: float | int | None = None,
    uncertainty_score: float | int | None = None,
    *,
    signal_cfg: SignalConfig | None = None,
) -> str:
    """Return buy/sell/hold policy from next-day expected return only.

    Other inputs are accepted for backward compatibility, but they must not
    change the user-facing recommendation.
    """
    if pd.isna(predicted_return):
        return "관망"

    cfg = _signal_cfg(signal_cfg)
    ret = float(predicted_return)
    if ret > float(cfg.recommendation_buy_threshold_pct):
        return "매수"
    if ret <= float(cfg.recommendation_sell_threshold_pct):
        return "매도"
    return "관망"
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
pytest tests/test_signal_policy_recommendation.py -q
```

Expected: all tests in file pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/domain/signal_policy.py tests/test_signal_policy_recommendation.py
git commit -m "Use signal config for scalar recommendation thresholds"
```

---

### Task 3: Vectorized recommendation matches scalar thresholds

**Files:**
- Modify: `tests/test_signal_policy.py`
- Modify: `src/domain/signal_policy.py`

- [ ] **Step 1: Write failing equivalence test**

Add import to `tests/test_signal_policy.py`:

```python
from src.config.settings import BacktestConfig, SignalConfig
```

Replace existing `from src.config.settings import BacktestConfig` import with the combined import above.

Append test:

```python
def test_vectorized_recommendation_matches_scalar_with_custom_thresholds():
    cfg = SignalConfig(
        recommendation_buy_threshold_pct=3.0,
        recommendation_sell_threshold_pct=-1.0,
    )
    frame = pd.DataFrame(
        {
            "predicted_return": [3.1, 3.0, 2.5, -0.9, -1.0, -1.1, np.nan],
        }
    )

    vectorized = signal_policy._recommendation_series(frame, signal_cfg=cfg).tolist()
    scalar = [
        signal_policy.recommendation_from_signal(None, value, signal_cfg=cfg)
        for value in frame["predicted_return"].tolist()
    ]

    assert vectorized == scalar
    assert vectorized == ["매수", "관망", "관망", "관망", "매도", "매도", "관망"]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
pytest tests/test_signal_policy.py::test_vectorized_recommendation_matches_scalar_with_custom_thresholds -q
```

Expected: fail with `TypeError: _recommendation_series() got an unexpected keyword argument 'signal_cfg'`.

- [ ] **Step 3: Implement vectorized config support and pass-through**

In `src/domain/signal_policy.py`, change:

```python
def _policy_recommendation(row: pd.Series, cfg: InvestmentCriteriaConfig | None = None) -> str:
    predicted_return = row.get("predicted_return")
    return recommendation_from_signal(None, predicted_return)
```

to:

```python
def _policy_recommendation(
    row: pd.Series,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> str:
    predicted_return = row.get("predicted_return")
    return recommendation_from_signal(None, predicted_return, signal_cfg=signal_cfg)
```

Change `_recommendation_series`:

```python
def _recommendation_series(df: pd.DataFrame, signal_cfg: SignalConfig | None = None) -> pd.Series:
    cfg = _signal_cfg(signal_cfg)
    predicted_return = _to_numeric_series_preserve_na(df, "predicted_return", default=float("nan"))
    recommendation = pd.Series("관망", index=df.index, dtype=object)
    recommendation.loc[predicted_return > float(cfg.recommendation_buy_threshold_pct)] = "매수"
    recommendation.loc[predicted_return <= float(cfg.recommendation_sell_threshold_pct)] = "매도"
    return recommendation
```

Change `_pm_summary_frame` signature and action:

```python
def _pm_summary_frame(
    df: pd.DataFrame,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> pd.DataFrame:
    action = _recommendation_series(df, signal_cfg=signal_cfg)
```

Change `build_pm_summary_fields` signature and action:

```python
def build_pm_summary_fields(
    row: pd.Series,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> dict[str, str]:
    action = _policy_recommendation(row, cfg=cfg, signal_cfg=signal_cfg)
```

Change `build_prediction_policy_frame` signature and pm call:

```python
def build_prediction_policy_frame(
    pred_df: pd.DataFrame,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    out = vectorized_event_signal_boost(pred_df, cfg=cfg)
    pm = _pm_summary_frame(out, cfg=cfg, signal_cfg=signal_cfg)
```

- [ ] **Step 4: Run signal policy tests to verify GREEN**

Run:

```powershell
pytest tests/test_signal_policy.py tests/test_signal_policy_recommendation.py tests/test_signal_policy_contract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/domain/signal_policy.py tests/test_signal_policy.py
git commit -m "Align vectorized recommendation thresholds with signal config"
```

---

### Task 4: Thread SignalConfig through pipeline finalization

**Files:**
- Modify: `tests/test_pipeline_smoke.py`
- Modify: `src/pipeline_support.py`
- Modify: `src/pipeline.py`

- [ ] **Step 1: Write failing pipeline pass-through test**

Append to `tests/test_pipeline_smoke.py` near other `_predict_pipeline_latest` tests:

```python
def test_latest_prediction_uses_custom_recommendation_thresholds(monkeypatch):
    from src.models.lgbm_heads import MultiHeadPrediction

    class IdentityCalibrator:
        def transform(self, values):
            return pd.Series(values)

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            return self

        def predict(self, latest):
            return MultiHeadPrediction(
                predicted_return=np.array([np.log1p(0.025), np.log1p(-0.015)]),
                up_probability=np.array([0.8, 0.2]),
                quantile_low=np.array([0.0, -0.02]),
                quantile_mid=np.array([0.02, -0.01]),
                quantile_high=np.array([0.03, 0.0]),
            )

    feat = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Symbol": ["A", "B"],
            "Close": [100.0, 100.0],
            "market_regime": ["normal", "normal"],
            "target_log_return": [0.0, 0.0],
            "target_up": [1, 0],
            "value_traded": [5_000_000_000.0, 5_000_000_000.0],
        }
    )
    scored_oof = pd.DataFrame(
        {
            "Symbol": ["A", "B"],
            "target_up": [1, 0],
            "predicted_return": [0.01, -0.01],
        }
    )
    monkeypatch.setattr("src.pipeline.MultiHeadStockModel", FakeModel)
    monkeypatch.setattr("src.pipeline.get_symbol_name_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("src.pipeline.append_issue_summary_columns", lambda pred_df, **_kwargs: pred_df)
    monkeypatch.setattr(
        "src.pipeline.append_generated_news_impact_context_with_runtime",
        lambda pred_df, *_args, **_kwargs: types.SimpleNamespace(
            frame=pred_df,
            to_metadata=lambda: {
                "requested_mode": "rule",
                "actual_mode": "none",
                "fallback_used": False,
                "fallback_reason": "no_context_rows",
            },
        ),
    )

    pred_df, *_ = _predict_pipeline_latest(
        feat=feat,
        feature_columns=[],
        cfg=AppConfig(),
        scored_oof=scored_oof,
        probability_calibrator=IdentityCalibrator(),
        prediction_context=PredictionFrameContext(
            external_coverage_ratio=1.0,
            investor_coverage_ratio=1.0,
            min_liquidity_threshold=0.0,
        ),
        coverage_gate_status="ok",
        context_raw_df=pd.DataFrame(),
        effective_openai_api_key=None,
        effective_openai_model=None,
        issue_summary_symbols=None,
        issue_summary_n_jobs=1,
        news_impact_report=None,
        signal_config=SignalConfig(
            return_weight=1.0,
            up_prob_weight=0.0,
            uncertainty_penalty=0.0,
            recommendation_buy_threshold_pct=3.0,
            recommendation_sell_threshold_pct=-1.0,
        ),
    )

    ordered = pred_df.sort_values("Symbol")
    assert ordered["recommendation"].tolist() == ["관망", "매도"]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
pytest tests/test_pipeline_smoke.py::test_latest_prediction_uses_custom_recommendation_thresholds -q
```

Expected: fail because `"A"` is still `"매수"` under hard-coded/default policy at `2.5%`.

- [ ] **Step 3: Thread signal config through finalization**

In `src/pipeline_support.py`, change `finalize_latest_prediction_frame`:

```python
def finalize_latest_prediction_frame(
    pred_df: pd.DataFrame,
    symbol_name_map: dict[str, str],
    investment_criteria: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> pd.DataFrame:
    out = pred_df.copy()
    out["symbol_name"] = out["Symbol"].astype(str).map(symbol_name_map).fillna(out["Symbol"].astype(str))
    out["confidence_score"] = (1 - out["uncertainty_score"].fillna(1)).clip(lower=0, upper=1)
    out["confidence_label"] = out["confidence_score"].map(confidence_label)
    out = build_prediction_policy_frame(out, cfg=investment_criteria, signal_cfg=signal_cfg)
    return out
```

In `src/pipeline.py`, change the call:

```python
    pred_df = finalize_latest_prediction_frame(
        pred_df,
        symbol_name_map,
        investment_criteria=cfg.investment_criteria,
        signal_cfg=effective_signal_config,
    )
```

- [ ] **Step 4: Run pipeline tests to verify GREEN**

Run:

```powershell
pytest tests/test_pipeline_smoke.py::test_latest_prediction_uses_custom_recommendation_thresholds tests/test_pipeline_smoke.py::test_latest_prediction_uses_tuned_signal_config -q
```

Expected: selected tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/pipeline.py src/pipeline_support.py tests/test_pipeline_smoke.py
git commit -m "Thread signal thresholds into pipeline recommendations"
```

---

### Task 5: Final verification and PR preparation

**Files:**
- Review only unless tests require small fixes.

- [ ] **Step 1: Run impacted tests**

Run:

```powershell
pytest tests/test_operational_hardening.py tests/test_signal_policy_recommendation.py tests/test_signal_policy.py tests/test_signal_policy_contract.py tests/test_pipeline_smoke.py -q
```

Expected: all pass.

- [ ] **Step 2: Run sample pipeline**

Run:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit code `0`, prediction output printed, report JSON generated.

- [ ] **Step 3: Run full test suite**

Run:

```powershell
pytest -q
```

Expected: all tests pass. Existing pandas FutureWarnings are acceptable if unchanged.

- [ ] **Step 4: Inspect diff and guardrails**

Run:

```powershell
git diff --stat HEAD~4..HEAD
git diff HEAD~4..HEAD -- src/domain/signal_policy.py src/config/settings.py
```

Confirm:

- No news/disclosure fields affect recommendation.
- `recommendation_from_signal` ignores signal score, probability, and uncertainty.
- Default thresholds remain `2.0` and `-2.0`.

- [ ] **Step 5: Push and open draft PR**

Because AGENTS.md requires PR after repository changes, push branch and create a draft PR.

Expected PR:

- Head: `p2-signal-config-thresholds`
- Base: `p2-publish-news-runtime-mode`
- Summary:
  - Adds `SignalConfig` recommendation thresholds.
  - Validates threshold signs/order.
  - Aligns scalar/vectorized recommendation policy.
  - Threads thresholds into latest prediction finalization.
- Tests:
  - impacted tests,
  - sample pipeline,
  - full pytest.

---

## Self-Review

- Spec coverage:
  - Config fields/validation: Task 1.
  - Scalar config support: Task 2.
  - Vectorized equivalence: Task 3.
  - Pipeline pass-through: Task 4.
  - Verification/PR: Task 5.
- Placeholder scan: no TBD/TODO/deferred steps.
- Type consistency:
  - `signal_cfg` means `SignalConfig | None` everywhere.
  - Config field names match spec: `recommendation_buy_threshold_pct`, `recommendation_sell_threshold_pct`.
  - Inclusivity matches existing behavior: buy uses `>`, sell uses `<=`.
