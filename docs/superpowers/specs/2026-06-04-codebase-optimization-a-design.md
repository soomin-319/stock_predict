# Codebase Optimization A Design

Date: 2026-06-04
Scope: Proceed with optimization approach A. Exclude external market feature cache.

## Goals

Implement low-risk codebase optimization while preserving current behavior and public interfaces.

1. Add `pipeline_report.json` diagnostics for timings, row counts, and coverage summary.
2. Split `run_pipeline()` into smaller private stage helpers without changing CLI/API behavior or output schemas.
3. Start Kakao bot modularization by extracting low-risk helpers only, while keeping existing imports and behavior compatible.

## Non-goals

- Do not implement external market feature cache/parquet.
- Do not change prediction policy, ranking, recommendation rules, or model feature selection.
- Do not let news/disclosure context affect `predicted_return` or signal decisions.
- Do not rewrite Kakao bot architecture in one large change.

## Approach

Use incremental extraction. First add diagnostics with tests, then split pipeline stages, then extract small Kakao helper modules. Existing function signatures remain stable.

## Pipeline Diagnostics

Add a lightweight timing collector around major `run_pipeline()` stages:

- config/input loading
- investor context preparation
- feature building
- validation/backtest/model scoring
- latest prediction
- report/artifact writing

Add report fields under `diagnostics`:

```json
{
  "diagnostics": {
    "timings_seconds": {"stage": 0.123},
    "row_counts": {"raw_input": 0, "features": 0, "oof_predictions": 0, "latest_predictions": 0},
    "coverage_summary": {
      "external_coverage_ratio": 1.0,
      "investor_coverage_ratio": 1.0,
      "coverage_gate_status": "pass"
    }
  }
}
```

Existing top-level coverage fields stay unchanged.

## `run_pipeline()` Split

Extract private helpers in `src/pipeline.py` first to minimize import churn:

- `_load_pipeline_config_and_data(...)`
- `_prepare_pipeline_context(...)`
- `_build_pipeline_feature_matrix(...)`
- `_run_pipeline_validation(...)`
- `_predict_pipeline_latest(...)`
- `_write_pipeline_artifacts(...)`

Use small dataclasses only where they reduce argument churn. Keep `run_pipeline()` as orchestration wrapper.

## Kakao Bot Split

Extract low-risk helper logic into new modules while preserving `src.chatbot.kakao_colab_bot` public classes/functions:

- `src/chatbot/responses.py`: Kakao response payload builders.
- `src/chatbot/intent.py`: utterance normalization and intent classification helpers.
- `src/chatbot/cache.py`: CSV mtime cache helpers if simple enough.

Do not move background process registry, Flask app creation, ngrok launch, or live external fetch in this pass unless tests show a very safe seam.

## Testing

Run at minimum:

- `python -m compileall -q src news_impact`
- `pytest -q --basetemp result/.pytest_tmp/optimization_a`

Add/update tests for:

- diagnostics fields in pipeline report
- preserved pipeline smoke behavior
- extracted Kakao response/intent helpers where practical

## Risks and Mitigations

- Risk: behavior changes from refactor. Mitigation: private helper extraction, existing tests, output schema preservation.
- Risk: Kakao bot import breakage. Mitigation: keep public API in `kakao_colab_bot.py`, extract only pure helpers first.
- Risk: diagnostics overhead. Mitigation: use `time.perf_counter()` only; no heavy profiling.
