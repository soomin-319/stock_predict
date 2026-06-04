# Codebase Optimization A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-risk diagnostics and modularization while preserving pipeline and Kakao bot behavior.

**Architecture:** Keep public entry points stable. Extract pipeline work into private helpers in `src/pipeline.py`; add a small diagnostics collector; extract pure Kakao helper modules only.

**Tech Stack:** Python 3.10+, pandas, pytest, existing stock_predict modules.

---

## File Structure

- Modify `src/pipeline.py`: add diagnostics helpers, dataclasses, private pipeline stage helpers, and keep `run_pipeline()` as orchestrator.
- Create `src/chatbot/responses.py`: pure Kakao response payload builders.
- Create `src/chatbot/intent.py`: pure utterance normalization and simple intent predicates used by bot.
- Modify `src/chatbot/kakao_colab_bot.py`: delegate simple response/intent helpers while preserving public classes/functions.
- Modify `tests/test_pipeline_smoke.py`: assert diagnostics report fields.
- Add `tests/test_chatbot_helpers.py`: test extracted helper modules.
- Modify `docs/CODEBASE_OPTIMIZATION_REPORT.md`: mark completed items for diagnostics, pipeline split, and partial Kakao split.

---

### Task 1: Pipeline diagnostics tests

**Files:**
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Add assertions to existing smoke report test**

Add after existing report payload assertions in `test_run_pipeline_smoke`:

```python
diagnostics = payload["diagnostics"]
assert "timings_seconds" in diagnostics
assert "row_counts" in diagnostics
assert "coverage_summary" in diagnostics
assert diagnostics["row_counts"]["raw_input"] > 0
assert diagnostics["row_counts"]["features"] > 0
assert diagnostics["row_counts"]["oof_predictions"] > 0
assert diagnostics["row_counts"]["latest_predictions"] > 0
assert diagnostics["coverage_summary"]["coverage_gate_status"] in {"pass", "warn", "halt"}
```

- [ ] **Step 2: Run focused test to see failure**

Run:

```powershell
pytest tests/test_pipeline_smoke.py::test_run_pipeline_smoke -q --basetemp result/.pytest_tmp/optimization_a_t1
```

Expected before implementation: `KeyError: 'diagnostics'`.

---

### Task 2: Implement pipeline diagnostics and stage helper extraction

**Files:**
- Modify: `src/pipeline.py`

- [ ] **Step 1: Add imports and helper dataclasses**

Add imports:

```python
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator
```

Add near helper functions:

```python
@dataclass(slots=True)
class PipelineDiagnostics:
    timings_seconds: dict[str, float] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def time_stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.timings_seconds[name] = self.timings_seconds.get(name, 0.0) + (perf_counter() - started)

    def set_rows(self, name: str, frame: pd.DataFrame | None) -> None:
        self.row_counts[name] = 0 if frame is None else int(len(frame))

    def to_report(self, coverage_summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "timings_seconds": {k: round(float(v), 6) for k, v in self.timings_seconds.items()},
            "row_counts": dict(self.row_counts),
            "coverage_summary": coverage_summary,
        }
```

- [ ] **Step 2: Extract configuration/data loading helper**

Create `_load_pipeline_config_and_data(...)` with arguments matching override options and return `(cfg, raw, cleaned, data, requested_universe_symbols)`. Move lines 291-341 into it without behavior changes.

- [ ] **Step 3: Extract context helper**

Create `_prepare_pipeline_context(...)` returning `(data, investor_context_coverage, context_raw_df)`. Move lines 343-384 into it.

- [ ] **Step 4: Extract feature helper**

Create `_build_pipeline_feature_matrix(data, cfg, use_external)` returning `(feat, external_coverage, feature_columns)`. Move lines 386-395 into it.

- [ ] **Step 5: Extract validation helper**

Create `_run_pipeline_validation(feat, feature_columns, cfg, use_external, external_coverage, investor_context_coverage)` returning a dict containing `folds`, `oof`, `wf_summary`, `baseline_summary`, `prediction_context`, `scored_oof`, `tune_df`, `eval_df`, `tuned`, `backtest_input`, `backtest`, `backtest_series`, `external_coverage_ratio`, `investor_coverage_ratio`, and `coverage_gate_status`. Move lines 397-466 except figures.

- [ ] **Step 6: Extract latest prediction helper**

Create `_predict_pipeline_latest(...)` returning `(pred_df, latest, symbol_summary_artifacts, oof_diagnostics)`. Move lines 479-529 except figure save dependency passed in as `figure_dir_path`.

- [ ] **Step 7: Extract artifact/report helper**

Create `_write_pipeline_artifacts(...)` and move lines 531-677. Include diagnostics in report:

```python
coverage_summary = {
    "external_coverage_ratio": external_coverage_ratio,
    "investor_coverage_ratio": investor_coverage_ratio,
    "coverage_gate_status": coverage_gate_status,
}
report["diagnostics"] = diagnostics.to_report(coverage_summary)
```

- [ ] **Step 8: Keep `run_pipeline()` as orchestrator**

Wrap each helper in `diagnostics.time_stage(...)`, set row counts after each stage:

```python
diagnostics.set_rows("raw_input", raw)
diagnostics.set_rows("cleaned_input", cleaned)
diagnostics.set_rows("context_input", data)
diagnostics.set_rows("features", feat)
diagnostics.set_rows("oof_predictions", validation_result["scored_oof"])
diagnostics.set_rows("latest_predictions", pred_df)
```

- [ ] **Step 9: Run focused pipeline test**

Run:

```powershell
pytest tests/test_pipeline_smoke.py::test_run_pipeline_smoke -q --basetemp result/.pytest_tmp/optimization_a_t2
```

Expected: PASS.

---

### Task 3: Extract Kakao helper modules

**Files:**
- Create: `src/chatbot/responses.py`
- Create: `src/chatbot/intent.py`
- Modify: `src/chatbot/kakao_colab_bot.py`
- Add: `tests/test_chatbot_helpers.py`

- [ ] **Step 1: Write helper tests**

Create tests:

```python
from src.chatbot.intent import normalize_utterance, is_status_utterance, is_help_utterance
from src.chatbot.responses import simple_text_response


def test_normalize_utterance_strips_whitespace():
    assert normalize_utterance("  도움말  ") == "도움말"


def test_status_and_help_intents():
    assert is_status_utterance("상태")
    assert is_status_utterance("status")
    assert is_help_utterance("도움말")
    assert is_help_utterance("help")


def test_simple_text_response_shape():
    assert simple_text_response("hello") == {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": "hello"}}]},
    }
```

- [ ] **Step 2: Implement `intent.py`**

```python
from __future__ import annotations


def normalize_utterance(value: object) -> str:
    return str(value or "").strip()


def is_status_utterance(value: object) -> bool:
    return normalize_utterance(value).lower() in {"상태", "status"}


def is_help_utterance(value: object) -> bool:
    return normalize_utterance(value).lower() in {"도움", "도움말", "help", "?"}
```

- [ ] **Step 3: Implement `responses.py`**

```python
from __future__ import annotations

from typing import Any


def simple_text_response(text: str) -> dict[str, Any]:
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": text}}]}}
```

- [ ] **Step 4: Delegate in `kakao_colab_bot.py`**

Import helpers and replace exact duplicated payload construction/status/help predicates where safe. Keep method names and class behavior unchanged.

- [ ] **Step 5: Run helper and Kakao tests**

Run:

```powershell
pytest tests/test_chatbot_helpers.py tests/test_kakao_colab_bot.py -q --basetemp result/.pytest_tmp/optimization_a_t3
```

Expected: PASS.

---

### Task 4: Update optimization report and verify all

**Files:**
- Modify: `docs/CODEBASE_OPTIMIZATION_REPORT.md`

- [ ] **Step 1: Mark completed items**

Change remaining checklist:

```markdown
- [x] `run_pipeline()` 단계 함수 분해
- [x] `kakao_colab_bot.py` 저위험 helper 모듈 일부 분리
- [ ] 외부 market feature 캐시 추가 (이번 범위 제외)
- [x] `pipeline_report.json` timing/row-count/coverage diagnostics 추가
```

- [ ] **Step 2: Run compileall**

Run:

```powershell
python -m compileall -q src news_impact
```

Expected: exit code 0.

- [ ] **Step 3: Run full tests**

Run:

```powershell
pytest -q --basetemp result/.pytest_tmp/optimization_a_full
```

Expected: all tests pass.

- [ ] **Step 4: Commit implementation**

Run:

```powershell
git add src tests docs/CODEBASE_OPTIMIZATION_REPORT.md
git commit -m "Refactor pipeline stages and add diagnostics"
```
