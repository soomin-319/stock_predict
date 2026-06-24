# Target News Impact LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gemma news-impact judging call the LLM only for the ticker attached to each collected news row, not every ticker in the watchlist.

**Architecture:** Preserve the existing fixture-to-pipeline flow. Add optional per-news ticker metadata to the canonical `NewsItem`, write it from `context_raw_df.Symbol`, and make `_build_llm_judged_events` use `item.ticker` when present, falling back to the full watchlist for legacy fixtures.

**Tech Stack:** Python dataclasses, pandas fixture builder, pytest.

---

### Task 1: Add regression test for targeted LLM calls

**Files:**
- Modify: `tests/test_news_impact_fixture.py`
- Test target: `src.news_impact.pipeline._build_llm_judged_events`

- [ ] **Step 1: Write failing test**

Add a test that builds two news rows for two symbols, reads them through `_news_item`, then calls `_build_llm_judged_events` with a three-ticker watchlist and a fake LLM client. Assert the fake client sees only two prompts: one for `005930`, one for `000660`.

- [ ] **Step 2: Run RED**

Run:

```powershell
pytest tests/test_news_impact_fixture.py::test_llm_judging_uses_news_row_symbol_not_full_watchlist -q
```

Expected before implementation: failure because calls include all watchlist tickers or news rows cannot carry ticker metadata.

### Task 2: Preserve source ticker on news fixtures

**Files:**
- Modify: `src/news_impact/schema.py`
- Modify: `src/reports/news_impact_fixture.py`
- Modify: `src/news_impact/pipeline.py`

- [ ] **Step 1: Add optional ticker to `NewsItem`**

Add `ticker: str | None = None` at the end of `NewsItem` and validate it with `_require_ticker` when present.

- [ ] **Step 2: Write ticker in fixture news rows**

When `source_type == "news"`, convert row `Symbol` to a six-digit ticker and include it as `"ticker"` if valid.

- [ ] **Step 3: Target LLM ticker loop**

In `_build_llm_judged_events`, replace the unconditional `for ticker in watchlist_tickers` with a helper that returns `[item.ticker]` when present and in watchlist, otherwise the legacy `watchlist_tickers`.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
pytest tests/test_news_impact_fixture.py::test_llm_judging_uses_news_row_symbol_not_full_watchlist -q
```

Expected: pass.

### Task 3: Verify impacted behavior

**Files:**
- Test: `tests/test_news_impact_fixture.py`
- Test: `tests/test_news_impact_context.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Run impacted tests**

```powershell
pytest tests/test_news_impact_fixture.py tests/test_news_impact_context.py -q
```

- [ ] **Step 2: Run required smoke**

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

- [ ] **Step 3: Run smoke test**

```powershell
pytest tests/test_pipeline_smoke.py -q
```

Expected: all pass. Generated outputs stay under `result/`.
