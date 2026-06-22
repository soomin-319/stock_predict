# P2 Publish News Runtime Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instructions forbid subagents.

**Goal:** Persist actual news impact runtime mode through pipeline report and publish metadata.

**Architecture:** Add a small runtime-result value in `src/reports/news_impact_context.py`, return it from the pipeline prediction stage, then let `src/ops/publish_predictions.py` use it as the publish source of truth. Existing dataframe helpers remain compatible.

**Tech Stack:** Python 3.10+, pandas, pytest, existing pipeline/publish modules.

---

## File Structure

- Modify `src/reports/news_impact_context.py`
  - Add `NewsImpactRuntimeResult`.
  - Add metadata wrappers around generated/rule and Gemma context appending.
- Modify `src/pipeline.py`
  - Return news runtime metadata from `_predict_pipeline_latest()`.
  - Add `news_impact_runtime` to `pipeline_report.json`.
- Modify `src/ops/published_store.py`
  - Extend `PublishMeta` with requested mode and fallback fields.
  - Include new fields in `publish_meta.json` and `published/index.json`.
- Modify `src/ops/publish_predictions.py`
  - Derive publish metadata from `report["news_impact_runtime"]`.
  - Keep old fallback when report lacks runtime metadata.
- Modify tests:
  - `tests/test_news_impact_context.py`
  - `tests/test_publish_predictions.py`
  - Add/adjust pipeline smoke expectations only if existing tests fail.

---

## Task 1: Runtime metadata object and generated-rule wrapper

**Files:**
- Modify: `src/reports/news_impact_context.py`
- Test: `tests/test_news_impact_context.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_news_impact_context.py`:

```python
def test_append_generated_news_impact_context_with_runtime_records_rule_mode():
    pred_df = pd.DataFrame(
        [
            {
                "Date": "2026-06-17",
                "Symbol": "005930.KS",
                "symbol_name": "Samsung",
                "predicted_return": 1.0,
                "recommendation": "관망",
                "signal_score": 0.0,
            }
        ]
    )
    context_raw_df = pd.DataFrame(
        [
            {
                "Date": "2026-06-17",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "Samsung wins HBM supply contract",
                "summary": "positive supply news",
                "url": "https://example.com/news",
            }
        ]
    )

    result = append_generated_news_impact_context_with_runtime(pred_df, context_raw_df)

    assert result.requested_mode == "rule"
    assert result.actual_mode == "rule_based"
    assert result.fallback_used is False
    assert result.fallback_reason is None
    assert result.to_metadata() == {
        "requested_mode": "rule",
        "actual_mode": "rule_based",
        "fallback_used": False,
        "fallback_reason": None,
    }
    assert "news_impact_final_score" in result.frame.columns
    assert result.frame["predicted_return"].tolist() == [1.0]
    assert result.frame["recommendation"].tolist() == ["관망"]
    assert result.frame["signal_score"].tolist() == [0.0]


def test_append_generated_news_impact_context_with_runtime_records_none_without_context():
    pred_df = pd.DataFrame(
        [{"Date": "2026-06-17", "Symbol": "005930.KS", "predicted_return": 1.0}]
    )

    result = append_generated_news_impact_context_with_runtime(pred_df, pd.DataFrame())

    assert result.frame.equals(pred_df)
    assert result.to_metadata() == {
        "requested_mode": "rule",
        "actual_mode": "none",
        "fallback_used": False,
        "fallback_reason": "no_context_rows",
    }
```

Also add import:

```python
from src.reports.news_impact_context import append_generated_news_impact_context_with_runtime
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_news_impact_context.py::test_append_generated_news_impact_context_with_runtime_records_rule_mode tests/test_news_impact_context.py::test_append_generated_news_impact_context_with_runtime_records_none_without_context -q
```

Expected: FAIL because `append_generated_news_impact_context_with_runtime` is missing.

- [ ] **Step 3: Implement minimal runtime object**

In `src/reports/news_impact_context.py`, add import:

```python
from dataclasses import dataclass
```

Add after `NEWS_IMPACT_COLUMNS`:

```python
@dataclass(frozen=True)
class NewsImpactRuntimeResult:
    frame: pd.DataFrame
    requested_mode: str
    actual_mode: str
    fallback_used: bool = False
    fallback_reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "actual_mode": self.actual_mode,
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
        }
```

Add wrapper near `append_generated_news_impact_context()`:

```python
def append_generated_news_impact_context_with_runtime(
    pred_df: pd.DataFrame,
    context_raw_df: pd.DataFrame | None,
) -> NewsImpactRuntimeResult:
    scored = append_generated_news_impact_context(pred_df, context_raw_df)
    actual_mode = "rule_based" if "news_impact_final_score" in scored.columns else "none"
    return NewsImpactRuntimeResult(
        frame=scored,
        requested_mode="rule",
        actual_mode=actual_mode,
        fallback_used=False,
        fallback_reason=None if actual_mode != "none" else "no_context_rows",
    )
```

- [ ] **Step 4: Run tests and verify pass**

Run same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reports/news_impact_context.py tests/test_news_impact_context.py
git commit -m "Add news impact runtime metadata wrapper"
```

---

## Task 2: Gemma success/fallback runtime wrapper

**Files:**
- Modify: `src/reports/news_impact_context.py`
- Test: `tests/test_news_impact_context.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_news_impact_context.py`:

```python
def test_append_llm_news_impact_context_with_runtime_records_gemma_success(tmp_path):
    pred_df = pd.DataFrame(
        [{"Date": "2026-06-17", "Symbol": "005930.KS", "predicted_return": 1.0}]
    )
    context_raw_df = pd.DataFrame(
        [{"Date": "2026-06-17", "Symbol": "005930.KS", "source_type": "news", "title": "HBM"}]
    )

    def fake_run_daily_pipeline(inputs):
        report_path = Path(inputs.output_dir) / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "date": "2026-06-17",
                            "ticker": "005930",
                            "final_score": 42.0,
                            "top_reason": "Gemma success",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return type("Result", (), {"artifact_paths": {"report.json": str(report_path)}})()

    result = append_llm_news_impact_context_with_runtime(
        pred_df,
        context_raw_df,
        llm_config_path="configs/news_impact.gemma.example.json",
        symbols=["005930.KS"],
        symbol_name_map={"005930.KS": "Samsung"},
        run_date="2026-06-17",
        _run_daily_pipeline=fake_run_daily_pipeline,
    )

    assert result.to_metadata() == {
        "requested_mode": "gemma",
        "actual_mode": "gemma",
        "fallback_used": False,
        "fallback_reason": None,
    }
    assert float(result.frame.loc[0, "news_impact_final_score"]) == 42.0


def test_append_llm_news_impact_context_with_runtime_records_fallback():
    pred_df = pd.DataFrame(
        [
            {
                "Date": "2026-06-17",
                "Symbol": "005930.KS",
                "symbol_name": "Samsung",
                "predicted_return": 1.0,
            }
        ]
    )
    context_raw_df = pd.DataFrame(
        [{"Date": "2026-06-17", "Symbol": "005930.KS", "source_type": "news", "title": "HBM"}]
    )

    def failing_run_daily_pipeline(inputs):
        raise RuntimeError("gemma down")

    result = append_llm_news_impact_context_with_runtime(
        pred_df,
        context_raw_df,
        llm_config_path="configs/news_impact.gemma.example.json",
        symbols=["005930.KS"],
        symbol_name_map={"005930.KS": "Samsung"},
        run_date="2026-06-17",
        _run_daily_pipeline=failing_run_daily_pipeline,
    )

    assert result.requested_mode == "gemma"
    assert result.actual_mode == "rule_based"
    assert result.fallback_used is True
    assert result.fallback_reason == "RuntimeError: gemma down"
    assert "news_impact_final_score" in result.frame.columns
```

Add imports if missing:

```python
import json
from pathlib import Path
from src.reports.news_impact_context import append_llm_news_impact_context_with_runtime
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_news_impact_context.py::test_append_llm_news_impact_context_with_runtime_records_gemma_success tests/test_news_impact_context.py::test_append_llm_news_impact_context_with_runtime_records_fallback -q
```

Expected: FAIL because `append_llm_news_impact_context_with_runtime` is missing.

- [ ] **Step 3: Refactor LLM helper into runtime wrapper**

In `src/reports/news_impact_context.py`, add:

```python
def _fallback_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
```

Add a new wrapper:

```python
def append_llm_news_impact_context_with_runtime(
    pred_df: pd.DataFrame,
    context_raw_df: pd.DataFrame | None,
    *,
    llm_config_path: str,
    symbols,
    symbol_name_map: dict[str, str],
    run_date: str,
    _run_daily_pipeline=run_daily_pipeline,
) -> NewsImpactRuntimeResult:
    if pred_df.empty or context_raw_df is None or context_raw_df.empty:
        fallback = append_generated_news_impact_context_with_runtime(pred_df, context_raw_df)
        return NewsImpactRuntimeResult(
            frame=fallback.frame,
            requested_mode="gemma",
            actual_mode=fallback.actual_mode,
            fallback_used=fallback.actual_mode == "rule_based",
            fallback_reason=None if fallback.actual_mode == "rule_based" else fallback.fallback_reason,
        )
    try:
        with tempfile.TemporaryDirectory(prefix="news_impact_gemma_") as tmp:
            bundle = build_news_impact_fixture(
                context_raw_df=context_raw_df,
                symbols=symbols,
                symbol_name_map=symbol_name_map,
                run_date=run_date,
                output_dir=tmp,
            )
            result = _run_daily_pipeline(
                DailyPipelineInputs(
                    run_date=run_date,
                    watchlist_path=bundle.watchlist_path,
                    company_master_path=bundle.company_master_path,
                    input_fixture_path=bundle.fixture_path,
                    output_dir=tmp,
                    semantic_clustering=False,
                    llm_config_path=llm_config_path,
                )
            )
            report_path = result.artifact_paths["report.json"]
            scored = append_news_impact_context(pred_df, report_path)
            if "news_impact_final_score" not in scored.columns:
                fallback = append_generated_news_impact_context_with_runtime(pred_df, context_raw_df)
                return NewsImpactRuntimeResult(
                    frame=fallback.frame,
                    requested_mode="gemma",
                    actual_mode=fallback.actual_mode,
                    fallback_used=True,
                    fallback_reason="gemma_no_valid_rows",
                )
            return NewsImpactRuntimeResult(
                frame=scored,
                requested_mode="gemma",
                actual_mode="gemma",
                fallback_used=False,
                fallback_reason=None,
            )
    except Exception as exc:
        reason = _fallback_reason(exc)
        print(f"[NEWS IMPACT][gemma] 실패 -> 규칙 기반 대체: {reason}")
        fallback = append_generated_news_impact_context_with_runtime(pred_df, context_raw_df)
        return NewsImpactRuntimeResult(
            frame=fallback.frame,
            requested_mode="gemma",
            actual_mode=fallback.actual_mode,
            fallback_used=True,
            fallback_reason=reason,
        )
```

Replace `append_llm_news_impact_context()` body with compatibility shim:

```python
def append_llm_news_impact_context(
    pred_df: pd.DataFrame,
    context_raw_df: pd.DataFrame | None,
    *,
    llm_config_path: str,
    symbols,
    symbol_name_map: dict[str, str],
    run_date: str,
    _run_daily_pipeline=run_daily_pipeline,
) -> pd.DataFrame:
    """Judge news/disclosure impact with gemma; fall back to rule-based scoring."""
    return append_llm_news_impact_context_with_runtime(
        pred_df,
        context_raw_df,
        llm_config_path=llm_config_path,
        symbols=symbols,
        symbol_name_map=symbol_name_map,
        run_date=run_date,
        _run_daily_pipeline=_run_daily_pipeline,
    ).frame
```

- [ ] **Step 4: Run tests and verify pass**

Run same two-test command. Expected: PASS.

- [ ] **Step 5: Run existing news context tests**

Run:

```bash
pytest tests/test_news_impact_context.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/reports/news_impact_context.py tests/test_news_impact_context.py
git commit -m "Track gemma news impact fallback mode"
```

---

## Task 3: Add runtime metadata to pipeline report

**Files:**
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_smoke.py` if existing smoke can assert report field, otherwise rely on existing smoke plus targeted unit tests.

- [ ] **Step 1: Write failing smoke assertion**

Find existing smoke test that calls `run_pipeline()` in `tests/test_pipeline_smoke.py` and add:

```python
assert report["news_impact_runtime"]["requested_mode"] in {"rule", "gemma", "none"}
assert report["news_impact_runtime"]["actual_mode"] in {"rule_based", "gemma", "none"}
assert isinstance(report["news_impact_runtime"]["fallback_used"], bool)
assert "fallback_reason" in report["news_impact_runtime"]
```

- [ ] **Step 2: Run smoke test and verify failure**

Run:

```bash
pytest tests/test_pipeline_smoke.py -q
```

Expected: FAIL because `news_impact_runtime` is missing.

- [ ] **Step 3: Modify pipeline imports**

In `src/pipeline.py`, update import block from `src.reports.news_impact_context` to include:

```python
    append_generated_news_impact_context_with_runtime,
    append_llm_news_impact_context_with_runtime,
```

- [ ] **Step 4: Modify `_predict_pipeline_latest()` return type and default metadata**

Change return annotation:

```python
) -> tuple[pd.DataFrame, pd.DataFrame, dict, MultiHeadStockModel, dict[str, Any]]:
```

Before the news impact `try`, add:

```python
    news_impact_runtime = {
        "requested_mode": "none",
        "actual_mode": "none",
        "fallback_used": False,
        "fallback_reason": None,
    }
```

Replace news impact block with:

```python
    try:
        if news_impact_report:
            pred_df = append_news_impact_context(pred_df, news_impact_report)
            news_impact_runtime = {
                "requested_mode": "none",
                "actual_mode": "none",
                "fallback_used": False,
                "fallback_reason": None,
            }
        elif news_impact_llm_config:
            runtime_result = append_llm_news_impact_context_with_runtime(
                pred_df,
                context_raw_df,
                llm_config_path=news_impact_llm_config,
                symbols=issue_summary_symbols,
                symbol_name_map=symbol_name_map,
                run_date=pd.to_datetime(pred_df["Date"]).max().strftime("%Y-%m-%d"),
            )
            pred_df = runtime_result.frame
            news_impact_runtime = runtime_result.to_metadata()
        else:
            runtime_result = append_generated_news_impact_context_with_runtime(pred_df, context_raw_df)
            pred_df = runtime_result.frame
            news_impact_runtime = runtime_result.to_metadata()
    except Exception as exc:
        warning = f"news impact context unavailable: {exc}"
        warnings.append(warning)
        _LOGGER.warning("%s", warning)
        news_impact_runtime = {
            "requested_mode": "gemma" if news_impact_llm_config else "rule",
            "actual_mode": "none",
            "fallback_used": bool(news_impact_llm_config),
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }
```

Change return:

```python
    return pred_df, latest, oof_diagnostics, model, news_impact_runtime
```

- [ ] **Step 5: Thread value through `run_pipeline()`**

Change unpacking:

```python
        pred_df, latest, oof_diagnostics, final_model, news_impact_runtime = _predict_pipeline_latest(
```

Pass into `_write_pipeline_artifacts()` by adding parameter:

```python
            news_impact_runtime=news_impact_runtime,
```

Change `_write_pipeline_artifacts()` signature:

```python
    news_impact_runtime: dict[str, Any],
```

Add to report dict near diagnostics/model metadata:

```python
        "news_impact_runtime": news_impact_runtime,
```

- [ ] **Step 6: Run smoke test and verify pass**

Run:

```bash
pytest tests/test_pipeline_smoke.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline.py tests/test_pipeline_smoke.py
git commit -m "Expose news impact runtime in pipeline report"
```

---

## Task 4: Publish meta/index actual mode fields

**Files:**
- Modify: `src/ops/published_store.py`
- Modify: `src/ops/publish_predictions.py`
- Test: `tests/test_publish_predictions.py`

- [ ] **Step 1: Write failing tests**

Update `tests/test_publish_predictions.py`.

In `test_publish_artifacts_writes_latest_history_index`, call:

```python
        requested_news_mode="gemma",
        news_fallback_used=False,
        news_fallback_reason=None,
```

Add assertions:

```python
    latest_meta = json.loads((published_root / "latest" / "publish_meta.json").read_text(encoding="utf-8"))
    assert latest_meta["news_mode"] == "gemma"
    assert latest_meta["requested_news_mode"] == "gemma"
    assert latest_meta["news_fallback_used"] is False
    assert latest_meta["news_fallback_reason"] is None
    index_entry = read_index(published_root)["entries"][0]
    assert index_entry["requested_news_mode"] == "gemma"
    assert index_entry["news_fallback_used"] is False
```

Add new test:

```python
def test_run_publish_records_actual_news_runtime_from_pipeline_report(tmp_path: Path):
    project_root = tmp_path
    run_dir = project_root / "result" / "runs" / "rid-runtime"
    _make_run_dir(run_dir)
    (project_root / "result" / "latest_manifest.json").write_text(
        '{"run_id": "rid-runtime"}', encoding="utf-8"
    )

    def fake_pipeline(news_impact_llm_config, full_refresh, config_json=None):
        return {
            "manifest": {"promoted": True, "status": "pass", "run_id": "rid-runtime"},
            "news_impact_runtime": {
                "requested_mode": "gemma",
                "actual_mode": "rule_based",
                "fallback_used": True,
                "fallback_reason": "RuntimeError: gemma down",
            },
        }

    result = run_publish(
        _Args(news_mode="gemma"),
        project_root=project_root,
        pipeline_fn=fake_pipeline,
        git_fn=lambda *a, **k: None,
        provenance_fn=lambda: ("abc1234", "feat/publish"),
    )

    assert result["news_mode"] == "rule_based"
    assert result["requested_news_mode"] == "gemma"
    assert result["news_fallback_used"] is True
    assert result["news_fallback_reason"] == "RuntimeError: gemma down"

    meta = json.loads((project_root / "published" / "latest" / "publish_meta.json").read_text(encoding="utf-8"))
    assert meta["news_mode"] == "rule_based"
    assert meta["requested_news_mode"] == "gemma"
    assert meta["news_fallback_used"] is True
```

- [ ] **Step 2: Run publish tests and verify failure**

Run:

```bash
pytest tests/test_publish_predictions.py -q
```

Expected: FAIL because new publish fields are missing.

- [ ] **Step 3: Extend `PublishMeta`**

In `src/ops/published_store.py`, update dataclass:

```python
@dataclass(frozen=True)
class PublishMeta:
    generated_at_kst: str
    trading_date: str
    news_mode: str
    source_run_id: str
    symbol_count: int
    git_commit: str | None = None
    git_branch: str | None = None
    requested_news_mode: str | None = None
    news_fallback_used: bool = False
    news_fallback_reason: str | None = None
```

Update `to_dict()`:

```python
        return {
            "generated_at_kst": self.generated_at_kst,
            "trading_date": self.trading_date,
            "news_mode": self.news_mode,
            "requested_news_mode": self.requested_news_mode or self.news_mode,
            "news_fallback_used": bool(self.news_fallback_used),
            "news_fallback_reason": self.news_fallback_reason,
            "source_run_id": self.source_run_id,
            "symbol_count": self.symbol_count,
            "git": {"commit": self.git_commit, "branch": self.git_branch},
        }
```

Update `update_index()` entry:

```python
    entry = {
        "trading_date": meta.trading_date,
        "generated_at_kst": meta.generated_at_kst,
        "news_mode": meta.news_mode,
        "requested_news_mode": meta.requested_news_mode or meta.news_mode,
        "news_fallback_used": bool(meta.news_fallback_used),
        "news_fallback_reason": meta.news_fallback_reason,
        "symbol_count": meta.symbol_count,
        "source_run_id": meta.source_run_id,
    }
```

- [ ] **Step 4: Extend publish artifact API**

In `src/ops/publish_predictions.py`, update `publish_artifacts()` signature:

```python
    requested_news_mode: str | None = None,
    news_fallback_used: bool = False,
    news_fallback_reason: str | None = None,
```

Pass into `PublishMeta`:

```python
        requested_news_mode=requested_news_mode,
        news_fallback_used=bool(news_fallback_used),
        news_fallback_reason=news_fallback_reason,
```

Add helper near `_news_config_for_mode()`:

```python
def _runtime_meta_from_report(report: dict[str, Any], configured_mode: str) -> dict[str, Any]:
    runtime = report.get("news_impact_runtime") if isinstance(report, dict) else None
    if isinstance(runtime, dict):
        requested = str(runtime.get("requested_mode") or configured_mode)
        actual = str(runtime.get("actual_mode") or ("rule_based" if configured_mode == "rule" else configured_mode))
        return {
            "requested_news_mode": requested,
            "news_mode": actual,
            "news_fallback_used": bool(runtime.get("fallback_used")),
            "news_fallback_reason": runtime.get("fallback_reason"),
        }
    return {
        "requested_news_mode": configured_mode,
        "news_mode": "rule_based" if configured_mode == "rule" else configured_mode,
        "news_fallback_used": False,
        "news_fallback_reason": None,
    }
```

In `run_publish()`, replace local `news_mode` calculation:

```python
    runtime_meta = _runtime_meta_from_report(report, str(args.news_mode))
```

Pass to `publish_artifacts()`:

```python
        news_mode=str(runtime_meta["news_mode"]),
        requested_news_mode=str(runtime_meta["requested_news_mode"]),
        news_fallback_used=bool(runtime_meta["news_fallback_used"]),
        news_fallback_reason=runtime_meta["news_fallback_reason"],
```

- [ ] **Step 5: Run publish tests and verify pass**

Run:

```bash
pytest tests/test_publish_predictions.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ops/published_store.py src/ops/publish_predictions.py tests/test_publish_predictions.py
git commit -m "Record publish news runtime mode"
```

---

## Task 5: Full verification and PR

**Files:**
- No planned source edits unless verification reveals a bug.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest tests/test_news_impact_context.py tests/test_publish_predictions.py tests/test_pipeline_smoke.py -q
```

Expected: PASS.

- [ ] **Step 2: Run sample pipeline smoke command**

Run:

```bash
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: exit code 0 and report written under `result/runs/<run_id>/pipeline_report.json`.

- [ ] **Step 3: Run full test suite**

Run:

```bash
pytest -q
```

Expected: PASS. Existing non-failing warnings are acceptable if unchanged.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git status --short
git log --oneline --decorate -6
git diff --stat origin/main...HEAD
```

Expected: clean except intended commits/files.

- [ ] **Step 5: Push and create draft PR**

Push branch:

```bash
git push -u origin p2-publish-news-runtime-mode
```

Create draft PR with GitHub connector:

- Base: `p0-display-only-feature-guard`
- Head: `p2-publish-news-runtime-mode`
- Title: `Expose publish news runtime mode`
- Body includes:
  - Summary
  - Test results
  - Stacked on P0 PR #311
  - Note: no change to recommendation/ranking/signal policy

---

## Self-Review

- Spec coverage:
  - Actual Gemma/rule fallback mode surfaced: Tasks 1-4.
  - Pipeline report source of truth: Task 3.
  - Publish meta/index fields: Task 4.
  - Display-only guard preserved: Tests assert policy columns unchanged; no signal code touched.
  - Verification/PR: Task 5.
- Placeholder scan: no `TBD`, no deferred steps, no unspecified tests.
- Type consistency:
  - `NewsImpactRuntimeResult.to_metadata()` produces keys consumed by publish helper.
  - Publish fields use `requested_news_mode`, `news_fallback_used`, `news_fallback_reason` consistently.
