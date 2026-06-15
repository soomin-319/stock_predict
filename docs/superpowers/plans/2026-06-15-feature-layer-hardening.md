# Feature Layer Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct feature-layer leakage and unsafe values, unify indicator implementations, improve grouped feature performance, expose missing-data diagnostics, and align documentation.

**Architecture:** External-market timing rules live in `external_features.py`; reusable technical formulas live only in `technical_indicators.py`; `price_features.py` orchestrates grouped calculations and continuous price features; configurable investment thresholds remain in `investment_signals.py`. Tests pin each boundary before production changes.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest

---

## File Map

- Modify `src/features/external_features.py`: classify same-date versus delayed external symbols, delay unavailable-at-close observations, remove backfill.
- Modify `src/features/technical_indicators.py`: make indicator outputs finite-safe and provide a single grouped technical block helper.
- Modify `src/features/price_features.py`: sort/restore rows, reuse technical helpers, add price-limit policy, neutral-value/missing diagnostics.
- Modify `src/features/investment_signals.py`: remain sole owner of configured 52-week-high threshold flag.
- Modify `src/features/feature_selection.py`: include approved missing-indicator features, exclude diagnostics that should not train models.
- Modify `src/pipeline.py`: include feature missing-rate diagnostics in pipeline report context without affecting decisions.
- Modify `docs/03_features.md`: rewrite as valid UTF-8 Korean documentation matching implementation.
- Modify focused test files and add `tests/test_feature_layer_hardening.py`.

### Task 1: External-Market Availability and No-Backfill Policy

**Files:**
- Modify: `tests/test_external_features.py`
- Modify: `src/features/external_features.py`

- [ ] **Step 1: Write failing availability tests**

Add tests that monkeypatch `_safe_download` with deterministic series and assert:

```python
def test_overseas_external_features_are_delayed_one_observation(monkeypatch):
    # ^GSPC close published on Jan 2 must first appear on Korean Jan 3.
    ...
    assert pd.isna(out.loc[out["Date"].eq("2024-01-02"), "gspc_close"]).all()
    assert out.loc[out["Date"].eq("2024-01-03"), "gspc_close"].iloc[0] == 100.0


def test_korean_external_features_remain_same_date(monkeypatch):
    ...
    assert out.loc[out["Date"].eq("2024-01-02"), "ks11_close"].iloc[0] == 100.0


def test_external_features_do_not_backfill_leading_dates(monkeypatch):
    ...
    assert pd.isna(out.loc[out["Date"].eq("2024-01-02"), "gspc_close"]).all()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_external_features.py -q`

Expected: delayed/no-backfill assertions fail because all symbols currently join same-date and `bfill()` fills leading dates.

- [ ] **Step 3: Implement conservative availability rules**

In `src/features/external_features.py`:

```python
SAME_DATE_EXTERNAL_SYMBOLS = frozenset({"^KS11", "^KQ11"})


def _apply_availability_lag(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if symbol in SAME_DATE_EXTERNAL_SYMBOLS:
        return frame
    out = frame.copy()
    value_columns = [column for column in out.columns if column != "Date"]
    out[value_columns] = out[value_columns].shift(1)
    return out
```

Apply the lag before merging each successful frame. Replace:

```python
ext = ext.sort_values("Date").ffill().bfill()
```

with:

```python
ext = ext.sort_values("Date").ffill()
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_external_features.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/features/external_features.py tests/test_external_features.py
git commit -m "Prevent external feature look-ahead"
```

### Task 2: Finite-Safe Technical Indicator Source of Truth

**Files:**
- Modify: `tests/test_feature_module_boundaries.py`
- Add: `tests/test_feature_layer_hardening.py`
- Modify: `src/features/technical_indicators.py`

- [ ] **Step 1: Write failing indicator safety tests**

Add tests:

```python
def test_obv_change_is_finite_when_obv_crosses_zero():
    block = compute_technical_indicator_block(frame, rsi_period=14, stochastic_period=14, cci_period=20)
    assert np.isfinite(block["obv_change_5d"].dropna()).all()


def test_technical_indicator_block_matches_public_helpers():
    block = compute_technical_indicator_block(frame, rsi_period=14, stochastic_period=14, cci_period=20)
    assert_series_equal(block["atr_14"], compute_atr(frame["High"], frame["Low"], frame["Close"]))
    assert_series_equal(block["obv"], compute_obv(frame["Close"], frame["Volume"]))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_feature_module_boundaries.py tests/test_feature_layer_hardening.py -q`

Expected: import failure because `compute_technical_indicator_block` does not exist.

- [ ] **Step 3: Implement the grouped technical block**

Add to `technical_indicators.py`:

```python
def finite_or_nan(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def compute_technical_indicator_block(
    frame: pd.DataFrame,
    *,
    rsi_period: int,
    stochastic_period: int,
    cci_period: int,
) -> pd.DataFrame:
    macd, macd_signal, macd_hist = compute_macd(frame["Close"])
    stoch_k, stoch_d = compute_stochastic(
        frame["High"], frame["Low"], frame["Close"], stochastic_period
    )
    obv = compute_obv(frame["Close"], frame["Volume"])
    block = pd.DataFrame(
        {
            "rsi_14": compute_rsi(frame["Close"], rsi_period),
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "atr_14": compute_atr(frame["High"], frame["Low"], frame["Close"]),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "cci_20": compute_cci(frame["High"], frame["Low"], frame["Close"], cci_period),
            "obv": obv,
            "obv_change_5d": finite_or_nan(obv.pct_change(5)),
        },
        index=frame.index,
    )
    return block.replace([np.inf, -np.inf], np.nan)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_feature_module_boundaries.py tests/test_feature_layer_hardening.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/features/technical_indicators.py tests/test_feature_module_boundaries.py tests/test_feature_layer_hardening.py
git commit -m "Add finite-safe technical indicator block"
```

### Task 3: Price Feature Orchestration, Row Order, and DRY Indicators

**Files:**
- Modify: `tests/test_feature_layer_hardening.py`
- Modify: `src/features/price_features.py`

- [ ] **Step 1: Write failing orchestration tests**

Add tests asserting:

```python
def test_build_features_restores_input_row_order():
    ...
    assert out["row_id"].tolist() == input_frame["row_id"].tolist()


def test_build_features_technical_columns_match_indicator_block():
    ...
    assert_series_equal(out["atr_14"], expected["atr_14"], check_names=False)


def test_vol_ratio_20_is_current_volume_over_twenty_day_average():
    ...
    assert out["vol_ratio_20"].iloc[-1] == pytest.approx(
        frame["Volume"].iloc[-1] / frame["Volume"].rolling(20).mean().iloc[-1]
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_feature_layer_hardening.py -q`

Expected: row-order/indicator-source assertions fail.

- [ ] **Step 3: Refactor `build_features()`**

At entry:

```python
out = df.copy()
out["_feature_input_order"] = np.arange(len(out))
out = out.sort_values(["Symbol", "Date", "_feature_input_order"], kind="stable")
```

Replace duplicated RSI/MACD/ATR/stochastic/CCI/OBV formulas with one grouped
technical-block pass:

```python
technical_blocks = [
    compute_technical_indicator_block(
        group,
        rsi_period=cfg.rsi_period,
        stochastic_period=cfg.stochastic_period,
        cci_period=cfg.cci_period,
    )
    for _, group in out.groupby("Symbol", sort=False)
]
technical = pd.concat(technical_blocks).sort_index()
```

Before return, replace infinities, restore original order, and remove helper:

```python
out = out.replace([np.inf, -np.inf], np.nan)
out = out.sort_values("_feature_input_order", kind="stable").drop(columns="_feature_input_order")
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_feature_layer_hardening.py tests/test_investor_features.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/features/price_features.py tests/test_feature_layer_hardening.py
git commit -m "Reuse technical indicators in price features"
```

### Task 4: Configured 52-Week Threshold Ownership

**Files:**
- Modify: `tests/test_feature_layer_hardening.py`
- Modify: `tests/test_investment_signal_features.py`
- Modify: `src/features/price_features.py`
- Modify: `src/features/investment_signals.py`

- [ ] **Step 1: Write failing threshold-ownership tests**

Add tests proving `build_features()` creates `close_to_52w_high` but does not
hardcode the final near-high flag, and that two different
`near_52w_distance_threshold` values yield different flags in
`add_investment_signal_features()`.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_feature_layer_hardening.py tests/test_investment_signal_features.py -q`

Expected: failure because `build_features()` currently creates the hardcoded
`>= 0.95` flag and uses it in `investor_event_score`.

- [ ] **Step 3: Remove the hardcoded threshold**

In `price_features.py`, remove `near_52w_high_flag` calculation and exclude it
from `investor_event_score`; keep `close_to_52w_high` and `breakout_52w_flag`.
In `investment_signals.py`, preserve:

```python
out["near_52w_high_flag"] = (
    out["distance_to_52w_high"] <= float(cfg.near_52w_distance_threshold)
).astype(int)
```

Do not add the flag back into any news/disclosure composite.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_feature_layer_hardening.py tests/test_investment_signal_features.py tests/test_display_only_feature_guard.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/features/price_features.py src/features/investment_signals.py tests/test_feature_layer_hardening.py tests/test_investment_signal_features.py
git commit -m "Use configured 52-week high threshold"
```

### Task 5: Historical Price-Limit Policy

**Files:**
- Modify: `tests/test_feature_layer_hardening.py`
- Modify: `src/features/price_features.py`

- [ ] **Step 1: Write failing historical and override tests**

Add:

```python
def test_price_limit_flags_use_historical_krx_thresholds():
    # 20% move is limit hit before 2015-06-15 but not after.
    ...


def test_price_limit_flags_use_explicit_row_override():
    # price_limit_pct=0.10 marks a 10% move.
    ...
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_feature_layer_hardening.py -q`

Expected: fail because flags use fixed `0.295`.

- [ ] **Step 3: Implement price-limit helper**

In `price_features.py`:

```python
KRX_PRICE_LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-15")


def _price_limit_pct(df: pd.DataFrame) -> pd.Series:
    explicit = next(
        (column for column in ("price_limit_pct", "PriceLimitPct") if column in df.columns),
        None,
    )
    default = pd.Series(
        np.where(pd.to_datetime(df["Date"]) < KRX_PRICE_LIMIT_CHANGE_DATE, 0.15, 0.30),
        index=df.index,
        dtype=float,
    )
    if explicit is None:
        return default
    return pd.to_numeric(df[explicit], errors="coerce").fillna(default)
```

Use a small tolerance:

```python
threshold = _price_limit_pct(out) - 0.005
limit_hit_up_flag = out["daily_return"].ge(threshold).astype(float)
limit_hit_down_flag = out["daily_return"].le(-threshold).astype(float)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_feature_layer_hardening.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/features/price_features.py tests/test_feature_layer_hardening.py
git commit -m "Apply historical Korean price limits"
```

### Task 6: Neutral Values and Missing Diagnostics

**Files:**
- Modify: `tests/test_feature_layer_hardening.py`
- Modify: `src/features/feature_selection.py`
- Modify: `src/features/price_features.py`
- Modify: `src/pipeline.py`

- [ ] **Step 1: Write failing neutral-value and diagnostic tests**

Add tests:

```python
def test_feature_neutral_values_and_missing_indicators_are_explicit():
    ...
    assert out["rsi_14"].iloc[0] == 50.0
    assert out["stoch_k"].iloc[0] == 50.0
    assert out["ma_120_missing"].iloc[0] == 1.0


def test_feature_missing_rate_summary_reports_selected_columns():
    summary = feature_missing_rate_summary(frame, ["ma_120", "rsi_14"])
    assert summary["ma_120"] == pytest.approx(1.0)
    assert summary["rsi_14"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_feature_layer_hardening.py -q`

Expected: missing helper/indicator columns.

- [ ] **Step 3: Implement explicit policy**

In `price_features.py`, add:

```python
NEUTRAL_FEATURE_VALUES = {
    "rsi_14": 50.0,
    "stoch_k": 50.0,
    "stoch_d": 50.0,
    "macd": 0.0,
    "macd_signal": 0.0,
    "macd_hist": 0.0,
}


def feature_missing_rate_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {
        column: float(df[column].isna().mean())
        for column in columns
        if column in df.columns
    }
```

Before neutral filling, add missing flags for history-dependent selected
features:

```python
for column in ("ma_120", "vol_60", "atr_14", "cci_20", "obv_change_5d"):
    if column in out:
        out[f"{column}_missing"] = out[column].isna().astype(float)
for column, neutral in NEUTRAL_FEATURE_VALUES.items():
    if column in out:
        out[column] = out[column].fillna(neutral)
```

Add `"_missing"` to allowed feature prefixes. In `pipeline.py`, calculate a
missing-rate diagnostics dictionary for the selected model features and include
it in the existing report/coverage metadata path only; do not use it in scores.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_feature_layer_hardening.py tests/test_display_only_feature_guard.py tests/test_pipeline_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/features/price_features.py src/features/feature_selection.py src/pipeline.py tests/test_feature_layer_hardening.py
git commit -m "Add feature missing-value diagnostics"
```

### Task 7: Documentation Alignment

**Files:**
- Modify: `docs/03_features.md`

- [ ] **Step 1: Rewrite documentation**

Write valid UTF-8 Korean documentation covering:

- module responsibilities and pipeline order;
- `vol_ratio_20` as volume/current-to-20-day-average ratio;
- external same-date/delayed symbol policy and no-backfill rule;
- configured 52-week-high threshold ownership;
- historical/default price-limit behavior and unsupported exceptions;
- technical helper reuse and infinity normalization;
- neutral values and missing diagnostics;
- display-only news/disclosure guardrail.

- [ ] **Step 2: Verify documentation**

Run:

```powershell
python -c "from pathlib import Path; p=Path('docs/03_features.md'); t=p.read_text(encoding='utf-8'); assert 'vol_ratio_20' in t; assert 'bfill' in t; print(len(t))"
```

Expected: prints positive character count without decoding error.

- [ ] **Step 3: Commit**

```powershell
git add docs/03_features.md
git commit -m "Align feature documentation with implementation"
```

### Task 8: Performance and Full Verification

**Files:**
- Modify only if a verified regression requires a focused fix.

- [ ] **Step 1: Run focused feature suite**

Run:

```powershell
pytest tests/test_external_features.py tests/test_feature_module_boundaries.py tests/test_feature_layer_hardening.py tests/test_investment_signal_features.py tests/test_investor_features.py tests/test_display_only_feature_guard.py -q
```

Expected: PASS.

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_pipeline_smoke.py -q`

Expected: PASS.

- [ ] **Step 3: Benchmark feature build**

Run a deterministic inline Python benchmark that builds at least 50 symbols ×
300 business days twice and reports the second-run elapsed time. Record the
result in the PR; investigate only if runtime exceeds 10 seconds in this
workspace.

- [ ] **Step 4: Run full suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 5: Verify repository diff**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended tracked changes plus pre-existing
untracked user files.

### Task 9: Review and Pull Request

**Files:**
- No source changes unless review finds a defect.

- [ ] **Step 1: Request code review**

Use `superpowers:requesting-code-review` and address correctness findings.

- [ ] **Step 2: Re-run verification after review fixes**

Run focused tests, smoke test, and full `pytest -q`.

- [ ] **Step 3: Publish**

Use `github:yeet` to confirm scope, push the branch, and create a draft pull
request containing:

- feature-layer hardening summary;
- exact test commands and results;
- benchmark result;
- changed artifact/document paths;
- explicit statement that news/disclosures remain display-only.
