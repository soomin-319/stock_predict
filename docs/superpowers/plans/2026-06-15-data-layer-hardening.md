# Data Layer Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all correctness, resilience, quality-policy, reporting, and documentation improvements in `docs/02_data.md`.

**Architecture:** Preserve current public APIs while adding focused helpers for market-aware symbol resolution, bounded provider retries, and fetch coverage. Keep quality flags in cleaned history, then exclude flagged rows at the pipeline model-input boundary.

**Tech Stack:** Python 3.10+, pandas, yfinance, pytest, argparse, JSON reports

---

## File Map

- Modify `src/data/krx_universe.py`: expose ticker-to-provider-symbol lookup.
- Modify `src/data/fetch_real_data.py`: centralized defaults, market candidates, retries, adjusted prices, coverage, BOM-safe saves.
- Modify `src/data/loaders.py`: BOM-safe loading and deterministic duplicate selection.
- Modify `src/data/cleaners.py`: deterministic duplicate policy and quality flags.
- Modify `src/features/external_features.py`: adjusted downloads and bounded retries.
- Modify `src/pipeline.py`: exclude quality-flagged rows, use shared default, report fetch coverage.
- Modify `docs/02_data.md`: document implemented contracts.
- Create `tests/test_data_layer_hardening.py`: focused deterministic regression tests.
- Modify `tests/test_fetch_real_fallback.py`, `tests/test_external_features.py`, `tests/test_pipeline_smoke.py`: integration contracts.

### Task 1: BOM-Safe Loading and Deterministic Cleaning

**Files:**
- Create: `tests/test_data_layer_hardening.py`
- Modify: `src/data/loaders.py`
- Modify: `src/data/cleaners.py`

- [ ] **Step 1: Write failing loader and cleaner tests**

```python
def test_load_ohlcv_csv_accepts_utf8_bom_and_symbol_argument(tmp_path):
    path = tmp_path / "bom.csv"
    _ohlcv_frame_without_symbol().to_csv(path, index=False, encoding="utf-8-sig")
    out = load_ohlcv_csv(path, symbol="005930.KS")
    assert out["Symbol"].unique().tolist() == ["005930.KS"]


def test_clean_ohlcv_selects_highest_volume_duplicate_and_flags_quality():
    out = clean_ohlcv(_duplicate_and_quality_frame())
    assert out.loc[out["Date"].eq(pd.Timestamp("2024-01-02")), "Volume"].item() == 500
    assert out["is_zero_volume"].tolist() == [False, False, True]
    assert out["is_extreme_return"].tolist() == [False, True, False]
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_data_layer_hardening.py -v`

Expected: BOM/quality-policy assertions fail.

- [ ] **Step 3: Implement minimal loader and cleaner behavior**

```python
# loaders.py
df = pd.read_csv(path, encoding="utf-8-sig")

# cleaners.py, after numeric coercion and validation
out["_input_order"] = range(len(out))
out = out.sort_values(["Date", "Symbol", "Volume", "_input_order"])
out = out.drop_duplicates(["Date", "Symbol"], keep="last")
out["is_zero_volume"] = out["Volume"].eq(0)
returns = out.groupby("Symbol", sort=False)["Close"].pct_change()
out["is_extreme_return"] = returns.abs().gt(0.40)
out = out.drop(columns="_input_order")
```

Also make `clean_ohlcv` coerce `Date` and fill a missing `Symbol` with
`"UNKNOWN"` before required-field filtering.

- [ ] **Step 4: Verify focused tests pass**

Run: `pytest tests/test_data_layer_hardening.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_data_layer_hardening.py src/data/loaders.py src/data/cleaners.py
git commit -m "Harden OHLCV loading and cleaning"
```

### Task 2: Market-Aware Symbol Resolution and Fallback

**Files:**
- Modify: `tests/test_data_layer_hardening.py`
- Modify: `src/data/krx_universe.py`
- Modify: `src/data/fetch_real_data.py`

- [ ] **Step 1: Write failing symbol-resolution tests**

```python
def test_normalize_known_kosdaq_ticker_uses_kq():
    assert normalize_user_symbols(["247540"]) == ["247540.KQ"]


def test_fetch_unknown_korean_ticker_falls_back_to_kq(monkeypatch):
    calls = []
    monkeypatch.setattr(fr, "_safe_download_ohlcv", fake_ks_empty_kq_success(calls))
    out = fr.fetch_real_ohlcv(["999999.KS"], start="2024-01-01")
    assert calls == ["999999.KS", "999999.KQ"]
    assert out["Symbol"].unique().tolist() == ["999999.KQ"]
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_data_layer_hardening.py -k "kosdaq or falls_back" -v`

Expected: normalization returns `.KS` and fallback test raises total-fetch failure.

- [ ] **Step 3: Implement lookup and candidate helpers**

```python
# krx_universe.py
def get_provider_symbol_for_ticker(ticker: str) -> str | None:
    normalized = str(ticker).strip().zfill(6)
    df = _load_krx_symbol_name_df()
    matches = df.loc[df["Ticker"].eq(normalized), "Symbol"]
    return None if matches.empty else str(matches.iloc[0])


# fetch_real_data.py
def _to_yfinance_symbol(user_input: str) -> str:
    ...
    mapped = get_provider_symbol_for_ticker(s) if s.isdigit() and len(s) == 6 else None
    return mapped or f"{s}.KS"


def _provider_symbol_candidates(symbol: str) -> list[str]:
    if symbol.endswith(".KS"):
        return [symbol, f"{symbol[:-3]}.KQ"]
    if symbol.endswith(".KQ"):
        return [symbol, f"{symbol[:-3]}.KS"]
    return [symbol]
```

Update `_fetch_single_symbol` to try candidates and write the successful
provider symbol into `Symbol`.

- [ ] **Step 4: Verify symbol tests pass**

Run: `pytest tests/test_data_layer_hardening.py -k "kosdaq or falls_back" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_data_layer_hardening.py src/data/krx_universe.py src/data/fetch_real_data.py
git commit -m "Resolve KRX symbols by market"
```

### Task 3: Adjusted Prices, Bounded Retries, and Fetch Coverage

**Files:**
- Modify: `tests/test_data_layer_hardening.py`
- Modify: `src/data/fetch_real_data.py`
- Modify: `src/features/external_features.py`
- Modify: `tests/test_external_features.py`

- [ ] **Step 1: Write failing retry, adjustment, and coverage tests**

```python
def test_real_download_uses_adjusted_prices_and_retries(monkeypatch):
    ticker = TransientTicker()
    monkeypatch.setattr(fr, "_get_yfinance", lambda: FakeYFinance(ticker))
    monkeypatch.setattr(fr, "_sleep", lambda _: None)
    out = fr.fetch_real_ohlcv(["005930.KS"], start="2024-01-01")
    assert ticker.calls == [True, True]
    assert not out.empty
    assert fr.get_last_fetch_coverage()["retried_symbols"] == ["005930.KS"]


def test_fetch_coverage_reports_partial_failure(monkeypatch):
    ...
    coverage = fr.get_last_fetch_coverage()
    assert coverage["requested"] == 2
    assert coverage["successful"] == 1
    assert coverage["failed_symbols"] == ["BAD"]
    assert coverage["success_ratio"] == 0.5


def test_external_download_uses_adjusted_prices_and_retries(monkeypatch):
    ...
    assert calls == [True, True]
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_data_layer_hardening.py tests/test_external_features.py -k "adjusted or retries or coverage" -v`

Expected: `auto_adjust=False`, no retries, or missing coverage accessor failures.

- [ ] **Step 3: Implement retry and coverage**

```python
# fetch_real_data.py
DEFAULT_REAL_START_DATE = "2020-01-01"
MAX_DOWNLOAD_ATTEMPTS = 3
_LAST_FETCH_COVERAGE = {}
_sleep = time.sleep

def get_last_fetch_coverage() -> dict:
    return copy.deepcopy(_LAST_FETCH_COVERAGE)

def _safe_download_ohlcv(...):
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            df = yf.Ticker(symbol).history(..., auto_adjust=True)
            if df is not None and not df.empty:
                return df, attempt
        except Exception as exc:
            last_exc = exc
        if attempt < MAX_DOWNLOAD_ATTEMPTS:
            _sleep(2 ** (attempt - 1))
    return pd.DataFrame(), MAX_DOWNLOAD_ATTEMPTS
```

Adapt fetch helpers to collect resolved symbol, attempts, fallback use, failed
symbols, and success ratio. Preserve public DataFrame and Path return types.
Implement the equivalent bounded loop with `auto_adjust=True` in
`external_features._safe_download`.

- [ ] **Step 4: Verify focused tests pass**

Run: `pytest tests/test_data_layer_hardening.py tests/test_external_features.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_data_layer_hardening.py tests/test_external_features.py src/data/fetch_real_data.py src/features/external_features.py
git commit -m "Retry adjusted market data downloads"
```

### Task 4: BOM-Safe Real-Data Saves and Shared Start Date

**Files:**
- Modify: `tests/test_data_layer_hardening.py`
- Modify: `tests/test_fetch_real_fallback.py`
- Modify: `src/data/fetch_real_data.py`
- Modify: `src/pipeline.py`

- [ ] **Step 1: Write failing save-encoding and default tests**

```python
def test_save_real_ohlcv_csv_writes_utf8_bom(tmp_path, monkeypatch):
    ...
    fr.save_real_ohlcv_csv(path, ["005930.KS"])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_cli_real_start_uses_data_layer_default():
    args = pipeline.build_cli_parser().parse_args([])
    assert args.real_start == fr.DEFAULT_REAL_START_DATE
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_data_layer_hardening.py tests/test_fetch_real_fallback.py -k "bom or real_start" -v`

Expected: output lacks BOM or CLI default is a literal independent of constant.

- [ ] **Step 3: Implement BOM-safe saves and shared default**

Use `encoding="utf-8-sig"` in save/append reads and writes where applicable.
Replace fetch/save/append and CLI default literals with
`DEFAULT_REAL_START_DATE`.

- [ ] **Step 4: Verify focused tests pass**

Run: `pytest tests/test_data_layer_hardening.py tests/test_fetch_real_fallback.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_data_layer_hardening.py tests/test_fetch_real_fallback.py src/data/fetch_real_data.py src/pipeline.py
git commit -m "Standardize real data CSV contracts"
```

### Task 5: Exclude Quality-Flagged Rows from Model Input

**Files:**
- Modify: `tests/test_data_layer_hardening.py`
- Modify: `src/pipeline.py`

- [ ] **Step 1: Write failing pipeline-boundary test**

```python
def test_load_pipeline_config_excludes_quality_flagged_rows(tmp_path):
    path = tmp_path / "quality.csv"
    _pipeline_quality_frame().to_csv(path, index=False)
    _, _, cleaned, data, _ = pipeline._load_pipeline_config_and_data(
        str(path), None, None, {}
    )
    assert cleaned[["is_zero_volume", "is_extreme_return"]].any(axis=1).sum() == 2
    assert not data[["is_zero_volume", "is_extreme_return"]].any(axis=1).any()
```

- [ ] **Step 2: Verify test fails**

Run: `pytest tests/test_data_layer_hardening.py -k excludes_quality -v`

Expected: flagged rows remain in `data`.

- [ ] **Step 3: Implement model-input exclusion**

```python
quality_excluded = cleaned[
    ~cleaned[["is_zero_volume", "is_extreme_return"]].fillna(False).any(axis=1)
].copy()
```

Apply universe filtering to `quality_excluded`, while continuing to return the
full `cleaned` frame for diagnostics.

- [ ] **Step 4: Verify focused and smoke tests pass**

Run: `pytest tests/test_data_layer_hardening.py tests/test_pipeline_smoke.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_data_layer_hardening.py src/pipeline.py
git commit -m "Exclude flagged market rows from modeling"
```

### Task 6: Pipeline Fetch Coverage Reporting

**Files:**
- Modify: `tests/test_pipeline_smoke.py`
- Modify: `tests/test_fetch_real_fallback.py`
- Modify: `src/pipeline.py`

- [ ] **Step 1: Write failing report and CLI propagation tests**

```python
def test_pipeline_report_contains_data_fetch_coverage(...):
    report = run_pipeline(..., data_fetch_coverage={"enabled": True, "requested": 2})
    assert report["data_fetch_coverage"]["requested"] == 2


def test_main_passes_refresh_coverage_to_pipeline(monkeypatch):
    ...
    assert captured["data_fetch_coverage"]["successful"] == 1
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_pipeline_smoke.py tests/test_fetch_real_fallback.py -k "data_fetch_coverage or refresh_coverage" -v`

Expected: unsupported argument or missing report key.

- [ ] **Step 3: Implement report propagation**

Add optional `data_fetch_coverage: dict | None = None` to `run_pipeline` and
`_write_pipeline_artifacts`. Write a stable disabled default when absent. In
`main`, capture `get_last_fetch_coverage()` after save/append refresh and pass
it to `run_pipeline`.

- [ ] **Step 4: Verify integration tests pass**

Run: `pytest tests/test_pipeline_smoke.py tests/test_fetch_real_fallback.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_smoke.py tests/test_fetch_real_fallback.py src/pipeline.py
git commit -m "Report real data fetch coverage"
```

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `docs/02_data.md`

- [ ] **Step 1: Update documented contracts**

Document market-aware `.KS/.KQ` behavior, adjusted prices, three-attempt
exponential retry, fetch coverage fields, BOM-safe I/O, duplicate selection,
quality flags and model exclusion, `symbol=`, normalized/fuzzy name matching,
and `DEFAULT_REAL_START_DATE`.

- [ ] **Step 2: Run documentation and source checks**

Run: `git diff --check`

Expected: no output, exit code 0.

- [ ] **Step 3: Run impacted tests**

Run: `pytest tests/test_data_layer_hardening.py tests/test_fetch_real_fallback.py tests/test_external_features.py tests/test_pipeline_smoke.py tests/test_krx_symbol_names.py tests/test_universe_loader.py -v`

Expected: all pass.

- [ ] **Step 4: Run smoke command**

Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`

Expected: exit code 0 and artifacts under `result/`.

- [ ] **Step 5: Run full suite**

Run: `pytest`

Expected: all tests pass.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/02_data.md
git commit -m "Document hardened data layer contracts"
```

- [ ] **Step 7: Review scope and prepare PR**

Run: `git status --short; git log --oneline --max-count=10`

Expected: only intentional user-existing untracked files remain; implementation
commits are present. Create a draft PR with summary, test results, and artifact
paths.
