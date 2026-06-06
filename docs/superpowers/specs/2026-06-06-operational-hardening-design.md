# Operational Hardening Design

## Scope

Fix high-priority operational issues identified in `docs/codebase_analysis_2026-06-06.md`:

- Stop globally forcing pytest temporary files into one shared directory.
- Prevent timed-out chatbot work from spawning duplicate follow-up work.
- Make investor/news/disclosure collection failures observable in pipeline reports.
- Reject unknown configuration keys and invalid quantile definitions before training.

Large-function refactors, dependency locking, CI linting, and model-artifact trust policy remain separate follow-up work.

## Design

### Test isolation

`tests/conftest.py` will only establish repository import paths and CPU limits. It will not mutate `TMP`, `TEMP`, `TMPDIR`, or `tempfile.tempdir`. Pytest owns per-run/per-test temporary directories.

### Chatbot timeout and single-flight

Timed calls will be submitted to a bot-owned executor and tracked by a stable operation key. Repeated requests for the same live-event or issue-summary operation reuse the in-flight future. Timeout returns promptly but does not pretend cancellation succeeded. Completion callbacks remove tracked futures. Issue-summary timeout handling must not start a second background summary when the timed future is still running.

### Context collection observability

Coverage dictionaries retain current counters and add structured failure details:

- `status`: `disabled`, `success`, `partial_failure`, `collection_failed`, or `no_events`
- `failed_symbols`
- `error_types`
- `details`

The pipeline-level raw-event collection status is included in `investor_context_coverage`. Failures remain display-only and never change `predicted_return`, ranking, or recommendations.

### Fail-fast validation

Configuration loading rejects unknown keys with a path-aware `ValueError`. Numeric ranges and list constraints are validated after merging. Quantile validation runs at the start of model fitting and requires at least three unique, strictly increasing values in `(0, 1)`.

## Testing

Use TDD regression tests for each behavior. Run focused tests, full pytest, compile/import checks, and sample pipeline smoke command.
