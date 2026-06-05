# Legacy Market Library Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unused legacy Korean market library integration and all stale repository references without adding FDR.

**Architecture:** Keep the bundled CSV as the sole default recommendation universe source. Delete the unreachable dynamic KOSPI200 loader, simplify its regression test, and correct documentation that describes the removed integration.

**Tech Stack:** Python, pandas, pytest, PowerShell, Git

---

### Task 1: Lock in bundled-universe-only behavior

**Files:**
- Modify: `tests/test_realtime_close_betting.py`
- Modify: `src/recommendation/realtime_close_betting.py`

- [ ] **Step 1: Write the failing test**

Rename the bundled-universe test to `test_default_realtime_service_uses_bundled_universe` and add:

```python
assert not hasattr(service, "_load_kospi200_symbols")
```

Remove the import blocker because the deleted integration must no longer exist.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_realtime_close_betting.py::test_default_realtime_service_uses_bundled_universe -v`

Expected: FAIL because `_load_kospi200_symbols` still exists.

- [ ] **Step 3: Write minimal implementation**

Delete `_load_kospi200_symbols()` from `RealTimeCloseBettingRecommendationService`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_realtime_close_betting.py::test_default_realtime_service_uses_bundled_universe -v`

Expected: PASS.

### Task 2: Remove stale repository references

**Files:**
- Modify: `AGENTS.md`
- Modify: `tests/test_p0_import_and_encoding.py`
- Modify: `docs/CODEBASE_ANALYSIS.md`
- Modify: `docs/EXTERNAL_DATA_INTEGRATION_GUIDE.md`
- Modify: `docs/PROJECT_OPERATION_GUIDE.md`
- Modify: `docs/archive/analysis-reports/CODEBASE_IMPROVEMENTS_2026-06-02.md`
- Modify: `docs/archive/analysis-reports/EXPERT_CODEBASE_REVIEW.md`
- Modify: `docs/archive/specs/2026-06-05-remove-legacy-market-library-design.md`

- [ ] **Step 1: Remove stale references**

Remove the retired library from blocked-import lists and documentation. Describe investor-flow collection as unavailable instead of claiming an active integration. Preserve historical report meaning while removing obsolete package-specific advice.

- [ ] **Step 2: Verify no tracked-file references remain**

Run: `git grep -n -i <retired-library-name>`

Expected: no output and exit code 1.

- [ ] **Step 3: Run focused tests**

Run: `pytest tests/test_realtime_close_betting.py tests/test_p0_import_and_encoding.py -q`

Expected: all tests pass.

### Task 3: Full verification and delivery

**Files:**
- No production-file changes expected.

- [ ] **Step 1: Run full tests**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Check diff**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 3: Commit implementation**

```bash
git add AGENTS.md src/recommendation/realtime_close_betting.py tests docs
git commit -m "Remove unused legacy market integration"
```

- [ ] **Step 4: Create pull request**

Push the branch and open a pull request containing summary and test results.
