# Graph Generation Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove graph generation and every graph-facing interface without changing prediction, validation, backtest, recommendation, CSV, or non-graph JSON calculations.

**Architecture:** Delete the downstream visualization layer and connect validation results directly to final prediction and artifact writing. Remove graph-only parameters and artifact metadata from pipeline, Colab, and Kakao interfaces while leaving calculation modules untouched.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, LightGBM, pytest, argparse

---

## File Structure

- Delete `src/reports/visualize.py`: graph-only implementation.
- Modify `src/pipeline.py`: remove graph imports, stage, parameters, and report artifacts.
- Modify `colab/stock_predict_colab.py`: remove graph parameter forwarding/output.
- Modify `src/chatbot/kakao_colab_bot.py`: remove graph runtime configuration and CLI forwarding.
- Modify `pyproject.toml`, `requirements.txt`: remove matplotlib.
- Modify pipeline, Colab, and Kakao tests; delete graph-only test.
- Modify active README/docs references; preserve archived documents.

### Task 1: Lock graph-free pipeline behavior

**Files:**
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing tests**

Update the smoke test to call `run_pipeline` without `figure_dir`, assert the
report keeps numeric calculation sections and non-graph artifacts, and assert no
artifact key contains `figure`, `plot`, or `.png`. Update the parser test to
assert `figure_dir` and `symbol_figure_limit` are absent and that parsing either
removed CLI option raises `SystemExit`.

- [ ] **Step 2: Verify RED**

Run:
`pytest tests/test_pipeline_smoke.py::test_run_pipeline_generates_report_without_graph_artifacts tests/test_pipeline_smoke.py::test_build_cli_parser_removes_graph_options -v`

Expected: failures because graph arguments and artifacts still exist.

- [ ] **Step 3: Remove graph behavior from pipeline**

In `src/pipeline.py`:

- Remove `src.reports.visualize` imports and `_save_pipeline_figures`.
- Remove `figure_dir` and `symbol_figure_limit` from `run_pipeline`.
- Reduce `total_steps` from 13 to 12.
- Keep validation/backtest calculation intact, but remove `create_figures` timing.
- Renumber final prediction and save stages.
- Remove graph arguments from `_predict_pipeline_latest` and
  `_write_pipeline_artifacts`.
- Remove `visualization_note`, `figure_dir`, and graph artifacts from reports.
- Remove `--figure-dir` and `--symbol-figure-limit` from parser and `main`.

- [ ] **Step 4: Verify GREEN**

Run the two focused tests from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `Remove graph generation from pipeline`

### Task 2: Remove graph interfaces from Colab and Kakao

**Files:**
- Modify: `tests/test_colab_runner.py`
- Modify: `tests/test_kakao_colab_bot.py`
- Modify: `colab/stock_predict_colab.py`
- Modify: `src/chatbot/kakao_colab_bot.py`

- [ ] **Step 1: Write failing interface tests**

Update tests to assert:

- `run_colab_pipeline` does not pass `figure_dir` and does not return
  `figure_dir`.
- `PipelineRuntimeConfig.build_command` never emits `--figure-dir`.
- Prewarm signature and prewarm pipeline call do not contain `figure_dir`.
- Kakao parser/config does not expose a graph directory.

- [ ] **Step 2: Verify RED**

Run:
`pytest tests/test_colab_runner.py tests/test_kakao_colab_bot.py -v`

Expected: graph-interface assertions fail.

- [ ] **Step 3: Remove graph interfaces**

Remove graph parameters, fields, command arguments, prewarm signature entries,
and returned paths from `colab/stock_predict_colab.py` and
`src/chatbot/kakao_colab_bot.py`.

- [ ] **Step 4: Verify GREEN**

Run the tests from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `Remove graph options from runtime integrations`

### Task 3: Delete graph module and dependency

**Files:**
- Delete: `src/reports/visualize.py`
- Delete: `tests/test_visualize_recent_month.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add a failing repository-boundary test**

Add a test in `tests/test_pipeline_smoke.py` asserting:

```python
assert not Path("src/reports/visualize.py").exists()
assert not Path("tests/test_visualize_recent_month.py").exists()
assert "matplotlib" not in Path("requirements.txt").read_text().lower()
assert "matplotlib" not in Path("pyproject.toml").read_text().lower()
```

- [ ] **Step 2: Verify RED**

Run the new test. Expected: FAIL because files and dependency still exist.

- [ ] **Step 3: Delete graph-only code**

Delete the graph module/test and remove both matplotlib dependency declarations.

- [ ] **Step 4: Verify GREEN**

Run the new test and `pytest tests/test_pipeline_smoke.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

Commit message: `Delete graph module and matplotlib dependency`

### Task 4: Update active documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/PROJECT_FEATURES_OVERVIEW.md`
- Modify: `docs/PROJECT_OPERATION_GUIDE.md`
- Modify: `docs/CODEBASE_ANALYSIS.md`
- Modify: `docs/RESULT_CLEANUP.md`

- [ ] **Step 1: Remove active graph instructions**

Remove graph CLI examples, graph artifact descriptions, visualization module
descriptions, and graph cleanup instructions. Describe backtest as calculation
and JSON reporting only. Do not modify `docs/archive/`.

- [ ] **Step 2: Verify references**

Run:

```powershell
Get-ChildItem README.md,docs,colab,src,tests -Recurse -File |
  Where-Object { $_.FullName -notmatch 'docs\\archive|__pycache__|superpowers' } |
  Select-String -Pattern 'figure_dir|figure-dir|matplotlib|reports.visualize|그래프 생성'
```

Expected: no functional graph-generation references.

- [ ] **Step 3: Commit**

Commit message: `Update documentation after graph removal`

### Task 5: Verify calculation invariance and full suite

**Files:**
- Modify only if a regression is exposed by verification.

- [ ] **Step 1: Run impacted tests**

Run:
`pytest tests/test_pipeline_smoke.py tests/test_colab_runner.py tests/test_kakao_colab_bot.py -v`

Expected: PASS.

- [ ] **Step 2: Run deterministic sample pipeline**

Run:
`python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`

Expected: succeeds, emits CSV/JSON artifacts, and creates no graph artifact.

- [ ] **Step 3: Verify calculated outputs**

Confirm `result/result_detail.csv` contains `predicted_return`, report contains
`walk_forward`, `tuned_signal`, `backtest`, `probability_calibration`, and no
graph artifact keys. Confirm no calculation module under `src/features`,
`src/models`, `src/validation`, `src/inference`, `src/domain`, or
`src/recommendation` changed.

- [ ] **Step 4: Run full suite**

Run: `pytest`

Expected: PASS.

- [ ] **Step 5: Final focused commit if needed**

Commit message: `Verify graph removal preserves calculations`

### Task 6: Review and publish

**Files:**
- No intended source changes.

- [ ] **Step 1: Review diff and status**

Run: `git status --short` and `git diff HEAD~4 --stat`.

- [ ] **Step 2: Request code review**

Use `superpowers:requesting-code-review`; resolve actionable findings.

- [ ] **Step 3: Push and open draft PR**

Use `github:yeet`. PR summary must state graph interfaces and matplotlib were
removed, calculation modules were unchanged, and list test commands/results.
