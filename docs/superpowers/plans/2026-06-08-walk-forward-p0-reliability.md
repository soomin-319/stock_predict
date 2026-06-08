# Walk-Forward P0 Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate overlapping walk-forward OOF predictions, enforce leakage-free tune/eval and calibration flow, and report insufficient evaluation data explicitly.

**Architecture:** Fold execution continues producing raw OOF predictions, but `walk_forward.py` adds provenance and converts raw rows into one deterministic `Date + Symbol` evaluation row. `support.py` owns strict temporal splitting and fit/transform probability calibration. `pipeline.py` consumes these structured results, tunes only on tune rows, backtests only on eval rows, and writes audit diagnostics.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn isotonic regression, pytest.

---

## File Structure

- Modify `src/validation/walk_forward.py`: fold provenance, OOF aggregation, duplicate diagnostics, structured validation result.
- Modify `src/validation/support.py`: strict temporal split, reusable fitted probability calibrator, split calibration metrics.
- Modify `src/pipeline.py`: leakage-free validation order, eval-only backtest, latest-prediction calibration, report fields.
- Modify `tests/test_walk_forward.py`: provenance and aggregation behavior.
- Modify `tests/test_pipeline_smoke.py`: strict split and report integration.
- Modify `tests/test_probability_calibration_guard.py`: calibrator fit/transform isolation.
- Modify `docs/WALK_FORWARD_GUIDE.md`: document implemented P0 behavior.

### Task 1: Fold Provenance and Deterministic OOF Aggregation

**Files:**
- Modify: `src/validation/walk_forward.py`
- Modify: `tests/test_walk_forward.py`

- [ ] **Step 1: Write failing provenance and aggregation tests**

Add tests that call a new `aggregate_oof_predictions(raw_oof)` helper and verify:

```python
def test_aggregate_oof_predictions_averages_duplicate_predictions():
    raw = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 2,
            "Symbol": ["A", "A"],
            "target_log_return": [0.02, 0.02],
            "target_up": [1, 1],
            "predicted_return": [0.01, 0.03],
            "up_probability": [0.6, 0.8],
            "quantile_low": [-0.01, 0.00],
            "quantile_mid": [0.01, 0.03],
            "quantile_high": [0.04, 0.06],
            "fold_id": [0, 1],
            "train_start": [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-02-01")],
            "train_end": [pd.Timestamp("2023-12-20"), pd.Timestamp("2023-12-27")],
            "valid_start": [pd.Timestamp("2024-01-02")] * 2,
            "valid_end": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")],
        }
    )

    aggregated, diagnostics = aggregate_oof_predictions(raw)

    assert len(aggregated) == 1
    assert aggregated.loc[0, "predicted_return"] == pytest.approx(0.02)
    assert aggregated.loc[0, "up_probability"] == pytest.approx(0.7)
    assert aggregated.loc[0, "oof_prediction_count"] == 2
    assert aggregated.loc[0, "fold_ids"] == [0, 1]
    assert diagnostics["duplicate_row_count"] == 1
    assert diagnostics["duplicate_ratio"] == pytest.approx(0.5)


def test_aggregate_oof_predictions_rejects_conflicting_targets():
    raw = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 2,
            "Symbol": ["A", "A"],
            "target_log_return": [0.01, 0.02],
            "target_up": [1, 1],
            "predicted_return": [0.01, 0.02],
            "up_probability": [0.6, 0.7],
        }
    )

    with pytest.raises(ValueError, match="Conflicting OOF target values"):
        aggregate_oof_predictions(raw)
```

Extend fake-fold tests to assert OOF contains `fold_id`, `train_start`, `train_end`, `valid_start`, and `valid_end`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_walk_forward.py -q
```

Expected: FAIL because `aggregate_oof_predictions` and provenance columns do not exist.

- [ ] **Step 3: Implement provenance and aggregation**

In `src/validation/walk_forward.py`:

```python
@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oof: pd.DataFrame
    oof_diagnostics: dict


def aggregate_oof_predictions(raw_oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if raw_oof.empty:
        return raw_oof.copy(), {
            "policy_version": "date_symbol_mean_v1",
            "raw_row_count": 0,
            "unique_row_count": 0,
            "duplicate_row_count": 0,
            "duplicate_ratio": 0.0,
        }

    key = ["Date", "Symbol"]
    target_cols = [c for c in ("target_log_return", "target_up") if c in raw_oof.columns]
    for target in target_cols:
        if raw_oof.groupby(key, dropna=False)[target].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting OOF target values for Date + Symbol: {target}")

    prediction_cols = [
        c for c in ("predicted_return", "up_probability", "quantile_low", "quantile_mid", "quantile_high")
        if c in raw_oof.columns
    ]
    provenance_cols = [c for c in ("fold_id", "train_start", "train_end", "valid_start", "valid_end") if c in raw_oof.columns]
    stable_cols = [c for c in raw_oof.columns if c not in {*key, *prediction_cols, *provenance_cols}]
    for column in stable_cols:
        if raw_oof.groupby(key, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting OOF stable values for Date + Symbol: {column}")

    rows = []
    for values, group in raw_oof.groupby(key, sort=True, dropna=False):
        row = dict(zip(key, values))
        row.update({column: group[column].iloc[0] for column in stable_cols})
        row.update({column: float(pd.to_numeric(group[column], errors="coerce").mean()) for column in prediction_cols})
        row["oof_prediction_count"] = int(len(group))
        row["fold_ids"] = sorted(pd.to_numeric(group["fold_id"], errors="coerce").dropna().astype(int).unique().tolist()) if "fold_id" in group else []
        for column in ("train_start", "train_end", "valid_start", "valid_end"):
            if column in group:
                row[f"{column}_values"] = sorted(pd.to_datetime(group[column]).dropna().unique().tolist())
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    raw_count = int(len(raw_oof))
    unique_count = int(len(aggregated))
    return aggregated, {
        "policy_version": "date_symbol_mean_v1",
        "raw_row_count": raw_count,
        "unique_row_count": unique_count,
        "duplicate_row_count": raw_count - unique_count,
        "duplicate_ratio": float((raw_count - unique_count) / raw_count),
    }
```

Change fold input to include chronological `fold_id`. Add `train_start` to `FoldResult`, and attach all provenance columns in `_run_fold`:

```python
oof["fold_id"] = fold_id
oof["train_start"] = train_df["Date"].min()
oof["train_end"] = train_end_date
oof["valid_start"] = valid_start_date
oof["valid_end"] = valid_end_date
```

Add:

```python
def walk_forward_validate_result(df, feature_columns, cfg) -> WalkForwardResult:
    executed = _execute_folds(list(enumerate(_iter_folds(df, cfg))), feature_columns, cfg)
    if not executed:
        empty, diagnostics = aggregate_oof_predictions(pd.DataFrame())
        return WalkForwardResult([], empty, diagnostics)
    folds = [result for result, _ in executed]
    raw_oof = pd.concat([fold for _, fold in executed], ignore_index=True)
    oof, diagnostics = aggregate_oof_predictions(raw_oof)
    return WalkForwardResult(folds, oof, diagnostics)
```

Keep `walk_forward_validate_with_oof` as a compatibility wrapper returning `.folds, .oof`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
pytest tests/test_walk_forward.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/validation/walk_forward.py tests/test_walk_forward.py
git commit -m "Deduplicate walk-forward OOF predictions"
```

### Task 2: Strict Temporal Split Without Data Reuse

**Files:**
- Modify: `src/validation/support.py`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing strict-split tests**

Add:

```python
def test_split_oof_reports_insufficient_data_without_reusing_dates():
    df = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=6),
            "Symbol": ["A"] * 6,
            "target_log_return": [0.0] * 6,
        }
    )

    result = split_oof_for_tuning_and_eval(df, tune_ratio=0.7, min_tune_dates=5, min_eval_dates=3)

    assert result.status == "insufficient_data"
    assert set(result.tune["Date"]).isdisjoint(set(result.eval["Date"]))
    assert result.eval.empty


def test_split_oof_returns_ordered_disjoint_dates():
    df = make_sample_df(days=40)[["Date", "Symbol"]]

    result = split_oof_for_tuning_and_eval(df, tune_ratio=0.7, min_tune_dates=5, min_eval_dates=5)

    assert result.status == "ok"
    assert result.tune["Date"].max() < result.eval["Date"].min()
    assert set(result.tune["Date"]).isdisjoint(set(result.eval["Date"]))
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_pipeline_smoke.py -q -k split_oof
```

Expected: FAIL because split returns a tuple and reuses all rows for short histories.

- [ ] **Step 3: Implement structured strict split**

In `src/validation/support.py`:

```python
@dataclass
class TemporalOOFSplit:
    tune: pd.DataFrame
    eval: pd.DataFrame
    status: str
    reason: str | None
    diagnostics: dict


def split_oof_for_tuning_and_eval(
    scored_oof: pd.DataFrame,
    tune_ratio: float = 0.7,
    *,
    min_tune_dates: int = 5,
    min_eval_dates: int = 5,
) -> TemporalOOFSplit:
    dates = sorted(pd.to_datetime(scored_oof["Date"]).dropna().dt.normalize().unique())
    if len(dates) < min_tune_dates + min_eval_dates:
        tune_dates = set(dates[: min(len(dates), min_tune_dates)])
        tune = scored_oof[pd.to_datetime(scored_oof["Date"]).dt.normalize().isin(tune_dates)].copy()
        return TemporalOOFSplit(
            tune=tune,
            eval=scored_oof.iloc[0:0].copy(),
            status="insufficient_data",
            reason="insufficient_unique_oof_dates",
            diagnostics={"unique_date_count": len(dates), "tune_date_count": len(tune_dates), "eval_date_count": 0},
        )

    split_idx = max(min_tune_dates, min(len(dates) - min_eval_dates, int(len(dates) * tune_ratio)))
    tune_dates, eval_dates = set(dates[:split_idx]), set(dates[split_idx:])
    normalized = pd.to_datetime(scored_oof["Date"]).dt.normalize()
    tune = scored_oof[normalized.isin(tune_dates)].copy()
    eval_df = scored_oof[normalized.isin(eval_dates)].copy()
    return TemporalOOFSplit(
        tune=tune,
        eval=eval_df,
        status="ok",
        reason=None,
        diagnostics={
            "unique_date_count": len(dates),
            "tune_date_count": len(tune_dates),
            "eval_date_count": len(eval_dates),
            "tune_row_count": len(tune),
            "eval_row_count": len(eval_df),
        },
    )
```

Update the pipeline compatibility wrapper:

```python
def _split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7):
    return validation_split_oof_for_tuning_and_eval(scored_oof, tune_ratio=tune_ratio)
```

Update existing tests to use `.tune` and `.eval`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
pytest tests/test_pipeline_smoke.py -q -k split_oof
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/validation/support.py tests/test_pipeline_smoke.py src/pipeline.py
git commit -m "Enforce strict temporal OOF split"
```

### Task 3: Fit Calibration on Tune Data Only

**Files:**
- Modify: `src/validation/support.py`
- Modify: `tests/test_probability_calibration_guard.py`

- [ ] **Step 1: Write failing calibration isolation tests**

Add:

```python
def test_fitted_calibrator_does_not_depend_on_eval_targets():
    tune = pd.DataFrame(
        {
            "up_probability": [0.1, 0.3, 0.5, 0.7, 0.9],
            "target_log_return": [-0.02, -0.01, 0.01, 0.02, 0.03],
        }
    )
    eval_a = pd.DataFrame({"up_probability": [0.2, 0.8], "target_log_return": [-0.1, 0.1]})
    eval_b = eval_a.assign(target_log_return=-eval_a["target_log_return"])

    calibrator = fit_up_probability_calibrator(tune)

    pd.testing.assert_series_equal(calibrator.transform(eval_a["up_probability"]), calibrator.transform(eval_b["up_probability"]))


def test_calibration_report_separates_tune_and_eval_metrics():
    tune = pd.DataFrame({"up_probability": [0.1, 0.9] * 10, "target_log_return": [-0.01, 0.01] * 10})
    eval_df = pd.DataFrame({"up_probability": [0.2, 0.8] * 10, "target_log_return": [-0.01, 0.01] * 10})

    calibrator = fit_up_probability_calibrator(tune)
    report = calibration_split_metrics(tune, eval_df, calibrator)

    assert set(report) == {"fit", "tune", "eval"}
    assert report["tune"]["sample_count"] == 20
    assert report["eval"]["sample_count"] == 20
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_probability_calibration_guard.py -q
```

Expected: FAIL because reusable calibrator helpers do not exist.

- [ ] **Step 3: Implement fitted calibrator**

In `src/validation/support.py`, add a small class with `transform`:

```python
@dataclass
class UpProbabilityCalibrator:
    model: object | None
    status: str
    reason: str | None

    def transform(self, probabilities) -> pd.Series:
        raw = pd.Series(probabilities, dtype=float).clip(0.0, 1.0)
        if self.model is None:
            return raw
        calibrated = pd.Series(self.model.predict(raw.values), dtype=float).clip(0.0, 1.0)
        if raw.round(6).nunique() >= 4 and calibrated.round(6).nunique() <= 2:
            return (0.3 * calibrated + 0.7 * raw).clip(0.0, 1.0)
        return calibrated
```

Implement:

```python
def fit_up_probability_calibrator(tune_oof: pd.DataFrame) -> UpProbabilityCalibrator:
    required = {"up_probability", "target_log_return"}
    if tune_oof.empty or not required.issubset(tune_oof.columns):
        return UpProbabilityCalibrator(None, "identity", "missing_or_empty_tune_oof")
    cal = tune_oof[list(required)].dropna()
    if cal["up_probability"].nunique() < 3:
        return UpProbabilityCalibrator(None, "identity", "insufficient_probability_diversity")
    y = (cal["target_log_return"] > 0).astype(int)
    if y.nunique() < 2:
        return UpProbabilityCalibrator(None, "identity", "insufficient_label_diversity")
    try:
        from sklearn.isotonic import IsotonicRegression
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(cal["up_probability"].astype(float).values, y.values)
        return UpProbabilityCalibrator(model, "fitted", None)
    except Exception as exc:
        return UpProbabilityCalibrator(None, "identity", f"fit_failed:{type(exc).__name__}")


def calibration_split_metrics(tune_df, eval_df, calibrator) -> dict:
    def metrics(frame):
        if frame.empty:
            return probability_calibration_metrics([], [])
        probabilities = calibrator.transform(frame["up_probability"])
        labels = (frame["target_log_return"] > 0).astype(int)
        return probability_calibration_metrics(labels.values, probabilities.values)

    return {
        "fit": {"status": calibrator.status, "reason": calibrator.reason},
        "tune": metrics(tune_df),
        "eval": metrics(eval_df),
    }
```

Retain compatibility wrapper:

```python
def calibrate_up_probability(oof_df, up_probs):
    return fit_up_probability_calibrator(oof_df).transform(up_probs)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
pytest tests/test_probability_calibration_guard.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/validation/support.py tests/test_probability_calibration_guard.py
git commit -m "Fit probability calibration on tuning OOF"
```

### Task 4: Integrate Leakage-Free Validation and Reporting

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing pipeline/report tests**

Update smoke assertions:

```python
assert payload["oof_policy"]["duplicate_policy"] == "date_symbol_mean"
assert payload["oof_policy"]["diagnostics"]["unique_row_count"] > 0
assert payload["validation_split"]["status"] == "ok"
assert set(payload["probability_calibration"]) == {"fit", "tune", "eval"}
assert payload["probability_calibration"]["eval"]["sample_count"] == payload["backtest_samples"]
```

Add a focused `_run_pipeline_validation` test using monkeypatched walk-forward output with too few unique dates:

```python
assert result["validation_split"]["status"] == "insufficient_data"
assert result["eval_df"].empty
assert result["backtest_input"].empty
assert result["backtest_status"] == "insufficient_data"
```

Add uniqueness assertion:

```python
assert not result["backtest_input"].duplicated(["Date", "Symbol"]).any()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_pipeline_smoke.py -q
```

Expected: FAIL because report fields and strict eval-only behavior are missing.

- [ ] **Step 3: Integrate structured validation flow**

In `src/pipeline.py`:

1. Use `walk_forward_validate_result` to obtain folds, deduplicated OOF, and diagnostics.
2. Split deduplicated raw OOF before calibration.
3. Fit calibrator from tune raw OOF only.
4. Transform tune and eval probabilities independently, then build scored frames.
5. Tune weights using scored tune only.
6. Recompute signal scores for both splits with selected weights.
7. Concatenate scored tune/eval only for diagnostics and figures.
8. Run backtest only when eval is non-empty; otherwise use:

```python
backtest_input = scored_eval
if backtest_input.empty:
    backtest_status = "insufficient_data"
    backtest = {
        "series": [],
        "sample_count": 0,
        "status": "insufficient_data",
        "reason": split.reason,
    }
else:
    backtest_status = "ok"
    backtest = run_long_only_topk_backtest(backtest_input, cfg.backtest)
```
9. Return the fitted calibrator for `_predict_pipeline_latest`; do not fit latest calibration from `scored_oof`.
10. Add report fields:

```python
"oof_policy": {
    "duplicate_policy": "date_symbol_mean",
    "diagnostics": validation_result["oof_diagnostics"],
},
"validation_split": validation_result["validation_split"],
"probability_calibration": validation_result["probability_calibration"],
```

Remove the old whole-OOF probability calibration report calculation.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
pytest tests/test_pipeline_smoke.py tests/test_probability_calibration_guard.py tests/test_walk_forward.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/pipeline.py tests/test_pipeline_smoke.py
git commit -m "Prevent leakage in OOF evaluation flow"
```

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `docs/WALK_FORWARD_GUIDE.md`
- Modify: `docs/walkforward_todo.md`

- [ ] **Step 1: Update documentation**

Document:

- overlapping folds remain allowed;
- duplicate `Date + Symbol` predictions are averaged;
- `oof_prediction_count` and fold provenance are retained;
- tune/eval dates never overlap;
- calibrator fits on tune only;
- insufficient eval data prevents backtest evaluation;
- report field names and meanings.

Mark only completed P0 checklist entries in `docs/walkforward_todo.md`.

- [ ] **Step 2: Run documentation checks**

Run:

```powershell
git diff --check
Select-String -Path docs\WALK_FORWARD_GUIDE.md -Pattern "OOF|보정|insufficient_data|oof_prediction_count"
```

Expected: no diff errors; all implemented concepts appear.

- [ ] **Step 3: Run impacted tests**

Run:

```powershell
pytest tests/test_walk_forward.py tests/test_probability_calibration_guard.py tests/test_pipeline_smoke.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```powershell
pytest -q
```

Expected: PASS with no unexpected warnings or errors.

- [ ] **Step 5: Run sample pipeline smoke command**

Run:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit code 0; generated report under `result/` includes `oof_policy`, `validation_split`, and split probability calibration fields.

- [ ] **Step 6: Commit documentation**

```powershell
git add docs/WALK_FORWARD_GUIDE.md docs/walkforward_todo.md
git commit -m "Document reliable walk-forward evaluation"
```

- [ ] **Step 7: Request code review and create PR**

Use `superpowers:requesting-code-review`, address actionable findings, then use the repository GitHub publishing workflow to push and open a pull request. PR body must include summary, test commands/results, and changed report fields.
