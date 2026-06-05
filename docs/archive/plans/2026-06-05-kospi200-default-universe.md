# KOSPI200 Default Universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bundled KOSPI200 symbols the sole default universe and fetch all 200 symbols when no explicit fetch universe is supplied.

**Architecture:** Reuse `data/kospi200_symbol_name_map.csv` through the existing universe loader. Remove the implicit five-symbol fallback limit, then update recommendation, chatbot cache, configuration, tests, and user-facing documentation to reference the same source.

**Tech Stack:** Python 3.10+, pandas, pytest

---

### Task 1: Default universe loader and full fallback fetch

**Files:**
- Modify: `tests/test_universe_loader.py`
- Modify: `tests/test_fetch_real_fallback.py`
- Modify: `src/data/universe.py`
- Modify: `src/data/cli_refresh.py`
- Modify: `src/pipeline.py`

- [ ] **Step 1: Write failing loader and fallback tests**

Update assertions so `load_default_universe_symbols()` returns 200 KOSPI200 symbols from `data/kospi200_symbol_name_map.csv`, and `_fallback_symbols_from_input_or_default()` returns the same complete list.

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_universe_loader.py tests/test_fetch_real_fallback.py -q`

Expected: failures showing old 100-symbol path/count and five-symbol fallback.

- [ ] **Step 3: Implement minimal loader and fallback changes**

Point `DEFAULT_UNIVERSE_CSV` at `data/kospi200_symbol_name_map.csv`. Remove `limit=5` defaults and slicing from both fallback wrappers while retaining compatible optional non-positive/full-list behavior only if still required by callers.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_universe_loader.py tests/test_fetch_real_fallback.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_universe_loader.py tests/test_fetch_real_fallback.py src/data/universe.py src/data/cli_refresh.py src/pipeline.py
git commit -m "Use full KOSPI200 default fetch universe"
```

### Task 2: Recommendation, chatbot cache, and configuration alignment

**Files:**
- Modify: `tests/test_realtime_close_betting.py`
- Modify: `tests/test_kakao_colab_bot.py`
- Modify: `tests/test_universe_loader.py`
- Modify: `src/recommendation/realtime_close_betting.py`
- Modify: `src/chatbot/kakao_colab_bot.py`
- Modify: `src/config/settings.py`

- [ ] **Step 1: Write failing alignment tests**

Assert default recommendation symbols come from the bundled KOSPI200 file, chatbot cache signatures track `data/kospi200_symbol_name_map.csv`, and `UniverseConfig()` has `name == "KOSPI200"` and `expected_size == 200`.

- [ ] **Step 2: Run tests and verify expected failure**

Run: `pytest tests/test_realtime_close_betting.py tests/test_kakao_colab_bot.py tests/test_universe_loader.py -q`

Expected: failures showing old default-universe paths and old configuration values.

- [ ] **Step 3: Implement minimal alignment changes**

Update recommendation and chatbot constants/paths to KOSPI200. Change `UniverseConfig` defaults to `KOSPI200` and `200`. Preserve explicit `universe_csv` and `universe_limit` behavior.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_realtime_close_betting.py tests/test_kakao_colab_bot.py tests/test_universe_loader.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_realtime_close_betting.py tests/test_kakao_colab_bot.py tests/test_universe_loader.py src/recommendation/realtime_close_betting.py src/chatbot/kakao_colab_bot.py src/config/settings.py
git commit -m "Align services with KOSPI200 universe"
```

### Task 3: Remove obsolete data and update documentation

**Files:**
- Delete: `data/default_universe_kospi50_kosdaq50.csv`
- Modify: `docs/PROJECT_FEATURES_OVERVIEW.md`
- Modify: `README.md`

- [ ] **Step 1: Delete obsolete bundled universe**

Delete `data/default_universe_kospi50_kosdaq50.csv`.

- [ ] **Step 2: Update user-facing documentation**

Document that the default universe is KOSPI200 and that unspecified `--fetch-real` / `--auto-refresh-real` operations use all 200 bundled symbols.

- [ ] **Step 3: Verify no active code or tests reference obsolete defaults**

Run:

```powershell
rg -n "default_universe_kospi50_kosdaq50|KOSPI200_KOSDAQ150|KOSPI200 \+ KOSDAQ150" src tests README.md docs/PROJECT_FEATURES_OVERVIEW.md
```

Expected: no matches.

- [ ] **Step 4: Commit**

```powershell
git add data/default_universe_kospi50_kosdaq50.csv docs/PROJECT_FEATURES_OVERVIEW.md README.md
git commit -m "Document KOSPI200 default universe"
```

### Task 4: Full verification and publication

**Files:**
- Verify all modified files

- [ ] **Step 1: Run focused regression tests**

Run:

```powershell
pytest tests/test_universe_loader.py tests/test_fetch_real_fallback.py tests/test_realtime_close_betting.py tests/test_kakao_colab_bot.py tests/test_pipeline_smoke.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 3: Inspect final diff**

Run: `git status --short --branch; git diff --check; git diff origin/HEAD...HEAD --stat`

Expected: only intended KOSPI200 migration changes; no whitespace errors.

- [ ] **Step 4: Push branch and create draft pull request**

Push the current branch and open a draft PR containing summary, tests, and note that default real-data fetch now requests all 200 KOSPI200 symbols.
