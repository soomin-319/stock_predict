# LLM Cache Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository instruction forbids subagents, so execute inline.

**Goal:** Add lifecycle controls to `FileLLMResponseCache`: TTL expiry, metadata validation, and maximum-entry pruning.

**Architecture:** Keep the cache API backward-compatible. Add optional constructor settings, validate enveloped cache metadata on read, keep legacy bare JSON readable, and prune oldest cache files after writes.

**Tech Stack:** Python 3.10+, pytest, `pathlib`, JSON cache files.

---

## File Structure

- Modify `src/news_impact/llm_client.py`
  - Extend `FileLLMResponseCache.__init__` with optional lifecycle arguments.
  - Add read-time envelope validation and expiry.
  - Add write-time cache pruning.
- Modify `tests/test_news_impact_llm_cache.py`
  - Add tests for TTL expiry, metadata mismatch, max-entry pruning, and legacy compatibility.

## Tasks

### Task 1: TTL expiry

**Files:**
- Modify: `tests/test_news_impact_llm_cache.py`
- Modify: `src/news_impact/llm_client.py`

- [ ] **Step 1: Write failing test**

Add a test that writes an enveloped cache entry with an old `cached_at` value and expects a miss:

```python
def test_cache_get_returns_none_for_expired_enveloped_entry(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path, ttl_seconds=60)
    (tmp_path / "expired.json").write_text(
        json.dumps(
            {
                "schema": "stock-news-impact.llm_cache.v1",
                "metadata": {"cached_at": "2000-01-01T00:00:00+00:00"},
                "response": {"direction": "positive"},
            }
        ),
        encoding="utf-8",
    )

    assert cache.get("expired") is None
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
pytest tests/test_news_impact_llm_cache.py::test_cache_get_returns_none_for_expired_enveloped_entry -q
```

Expected: FAIL because constructor does not accept `ttl_seconds`.

- [ ] **Step 3: Implement minimal TTL support**

Change `FileLLMResponseCache` to accept `ttl_seconds`, write `cached_at` metadata for enveloped entries, and return `None` for expired entries.

- [ ] **Step 4: Run passing test**

Run same pytest command. Expected: PASS.

### Task 2: Metadata validation

**Files:**
- Modify: `tests/test_news_impact_llm_cache.py`
- Modify: `src/news_impact/llm_client.py`

- [ ] **Step 1: Write failing test**

Add a test that expects read-time metadata mismatch to miss:

```python
def test_cache_get_returns_none_when_expected_metadata_differs(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path)
    cache.set("k1", {"direction": "positive"}, metadata={"model": "gemma"})

    assert cache.get("k1", expected_metadata={"model": "other"}) is None
    assert cache.get("k1", expected_metadata={"model": "gemma"}) == {
        "direction": "positive"
    }
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
pytest tests/test_news_impact_llm_cache.py::test_cache_get_returns_none_when_expected_metadata_differs -q
```

Expected: FAIL because `get()` does not accept `expected_metadata`.

- [ ] **Step 3: Implement metadata validation**

Extend `LLMResponseCache.get()` protocol and `FileLLMResponseCache.get()` with `expected_metadata`. Make `LlamaCppClient.chat_json()` pass expected metadata built from prompts/config before returning cached values.

- [ ] **Step 4: Run passing test**

Run same pytest command. Expected: PASS.

### Task 3: Maximum-entry pruning

**Files:**
- Modify: `tests/test_news_impact_llm_cache.py`
- Modify: `src/news_impact/llm_client.py`

- [ ] **Step 1: Write failing test**

Add a test that keeps only newest entries:

```python
def test_cache_set_prunes_oldest_files_when_max_entries_exceeded(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path, max_entries=2)

    cache.set("k1", {"x": 1})
    cache.set("k2", {"x": 2})
    cache.set("k3", {"x": 3})

    remaining = sorted(path.stem for path in tmp_path.glob("*.json"))
    assert remaining == ["k2", "k3"]
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
pytest tests/test_news_impact_llm_cache.py::test_cache_set_prunes_oldest_files_when_max_entries_exceeded -q
```

Expected: FAIL because constructor does not accept `max_entries`.

- [ ] **Step 3: Implement pruning**

After each `set()`, sort JSON files by modified time and remove oldest until count is `max_entries`.

- [ ] **Step 4: Run passing test**

Run same pytest command. Expected: PASS.

### Task 4: Regression verification and docs

**Files:**
- Modify: `docs/08_news_impact.md`

- [ ] **Step 1: Update docs**

Change the P2 section to note TTL, metadata mismatch validation, and max-entry pruning are implemented.

- [ ] **Step 2: Run impacted tests**

Run:

```powershell
pytest tests/test_news_impact_llm_cache.py -q
```

Expected: PASS.

- [ ] **Step 3: Run required smoke tests**

Run:

```powershell
pytest tests/test_pipeline_smoke.py -q
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

Expected: both commands pass.

- [ ] **Step 4: Commit, push, PR**

Commit only files changed for this task:

```powershell
git add src/news_impact/llm_client.py tests/test_news_impact_llm_cache.py docs/08_news_impact.md docs/superpowers/plans/2026-06-19-llm-cache-lifecycle.md
git commit -m "Add LLM cache lifecycle controls"
git push
```

Open a pull request with summary, tests, and artifact path `pipeline_report_smoke.json`.
