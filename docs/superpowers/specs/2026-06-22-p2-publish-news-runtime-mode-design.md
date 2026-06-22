# P2 Publish News Runtime Mode Design

## Goal

Make `stock-predict-publish` expose the news impact mode that actually ran.
Operators must be able to tell whether a publish requested Gemma, successfully used
Gemma, or silently fell back to rule-based scoring.

News and disclosure impact remains display-only context. This change must not affect
`predicted_return`, ranking, recommendation, `signal_score`, or trading gates.

## Current Behavior

`src.ops.publish_predictions.run_publish()` maps CLI mode to pipeline arguments:

- `--news-mode gemma` passes `configs/news_impact.gemma.example.json`.
- `--news-mode rule` passes no LLM config.

The publish metadata records the configured/requested mode only. The actual fallback
inside `append_llm_news_impact_context()` is printed but not persisted in
`pipeline_report.json`, `publish_meta.json`, or `published/index.json`.

## Proposed Approach

Use the pipeline report as the source of truth for publish runtime mode.

Add a small runtime metadata contract:

```json
{
  "news_impact_runtime": {
    "requested_mode": "gemma",
    "actual_mode": "rule_based",
    "fallback_used": true,
    "fallback_reason": "RuntimeError: connection refused"
  }
}
```

Allowed values:

- `requested_mode`: `gemma`, `rule`, or `none`
- `actual_mode`: `gemma`, `rule_based`, or `none`
- `fallback_used`: boolean
- `fallback_reason`: short string or `null`

## Component Changes

### `src.reports.news_impact_context`

Keep existing dataframe helpers compatible. Add a metadata-returning wrapper used by
the pipeline:

- Gemma requested and succeeds: append Gemma context, `actual_mode="gemma"`.
- Gemma requested and errors/no usable rows: append generated rule context,
  `actual_mode="rule_based"`, `fallback_used=true`, with reason.
- Rule/generated path: append generated rule context, `actual_mode="rule_based"`.
- No context rows: preserve dataframe and use `actual_mode="none"`.

Existing public helpers can remain for direct tests and callers.

### `src.pipeline`

Record the wrapper result under `report["news_impact_runtime"]` before writing
`pipeline_report.json`. CSV output stays unchanged except existing display-only
news columns.

### `src.ops.publish_predictions`

Read `news_impact_runtime` from the returned pipeline report. Publish metadata should
reflect actual mode and also preserve requested mode:

- `news_mode`: actual runtime mode for backwards-compatible operator display
- `requested_news_mode`: CLI/requested mode
- `news_fallback_used`: boolean
- `news_fallback_reason`: short string or `null`

`published/index.json` entries include the same requested/fallback fields.

If an older pipeline report lacks `news_impact_runtime`, publish falls back to the
current configured behavior.

## Testing

Add deterministic pytest coverage:

1. Gemma success records `actual_mode="gemma"` and no fallback.
2. Gemma failure records `actual_mode="rule_based"` and fallback reason.
3. Rule mode records `actual_mode="rule_based"`.
4. No context records `actual_mode="none"`.
5. `run_publish()` writes actual mode plus requested/fallback fields into
   `publish_meta.json` and `index.json`.

Run:

- targeted news impact and publish tests
- `pytest tests/test_pipeline_smoke.py -q`
- final `pytest -q`
- sample pipeline smoke command from `AGENTS.md`

## Non-goals

- No change to recommendation thresholds or signal policy.
- No use of news/disclosure values for ranking or expected return.
- No external LLM/server dependency in tests.
- No generated publish artifacts checked in unless already required by tests.

## Open Questions Resolved

- Branching: implement as a stacked PR on top of P0 branch work.
- Runtime source: pipeline report, not CSV inference.
- Backward compatibility: `news_mode` remains present and now means actual mode.
