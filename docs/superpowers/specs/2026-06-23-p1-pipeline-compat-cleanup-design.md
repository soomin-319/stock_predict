# P1 Pipeline Compatibility Cleanup Design

## Goal

Remove a small, safe slice of `src/pipeline.py` compatibility wrappers after moving tests/importers to the canonical modules that own the behavior.

## Scope

This slice targets wrappers that are thin delegations and are currently used only by tests:

- `_feature_columns` -> `src.features.price_features.select_feature_columns`
- `_calibrate_up_probability` -> `src.validation.support.calibrate_up_probability`

If exploration during implementation shows another wrapper is equally isolated and test-only, it may be included only when the diff stays small and behavior-neutral.

## Non-goals

- Do not change `run_pipeline`, CLI behavior, model training, prediction math, ranking, recommendations, or signal policy.
- Do not change generated artifact schema.
- Do not change news/disclosure semantics; they remain display-only context and cannot affect expected returns, rankings, recommendations, or signals.
- Do not remove wrappers still used by runtime code or external-facing CLI paths in this slice.

## Architecture

`pipeline.py` should stay focused on orchestration. Utility behavior should be imported directly from its owning module in tests and future callers:

- Feature-column policy lives in `src.features.price_features`.
- Probability calibration lives in `src.validation.support`.

This reduces top-of-file wrapper clutter without changing the pipeline data flow.

## Data Flow

No runtime data flow changes. Tests will call canonical functions directly. `run_pipeline()` will continue using the same modules internally through existing imports.

## Error Handling

No new error handling is needed. Existing canonical helpers keep their current behavior.

## Testing

Use test-driven cleanup:

1. Update tests to import/call canonical helpers directly.
2. Run targeted tests and confirm failures only from now-unused pipeline wrappers if removed prematurely.
3. Remove wrappers from `pipeline.py`.
4. Run impacted tests, `tests/test_pipeline_smoke.py`, the required sample pipeline command, and full `pytest` before handoff.

## Rollback

If a wrapper is still needed by runtime code or many tests, keep it and defer that wrapper to a later slice.
