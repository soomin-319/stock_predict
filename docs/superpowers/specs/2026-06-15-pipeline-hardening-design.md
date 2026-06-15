# Pipeline Hardening Design

## Goal

Implement all improvement proposals in `docs/01_pipeline.md`: preserve input
configuration, isolate optional-stage failures, retry statistically weak
walk-forward validation, verify diagnostics coverage, and document stable CLI
exit codes.

## Scope

- Keep `predicted_return`, expected-return ranking, recommendations, and signal
  decisions independent from news and disclosure context.
- Preserve the current sequential `run_pipeline()` architecture.
- Avoid broad pipeline class or workflow-engine refactoring.
- Add deterministic tests without live network calls.

## Architecture

### Standard Stage Diagnostics

Define the 12 documented pipeline stage names as one ordered constant. Extend
`PipelineDiagnostics` to record:

- elapsed time by stage;
- row counts;
- `stage_status` values of `ok`, `skipped`, or `error`;
- a short reason for skipped and failed stages;
- walk-forward fold count and whether adaptive retry ran;
- warnings for any standard stage missing from diagnostics.

A stage context manager will centralize timing and status recording. It will
record errors before re-raising them. Optional stages may catch those errors at
their boundary, record a caution, and return a safe empty or unchanged
fallback. Core stages continue to fail fast.

### Failure Policy

Optional operational context must not block core prediction:

- investor context failure returns unchanged price data, empty context events,
  and failure coverage metadata;
- external-market feature failure continues with price features and failure
  coverage metadata;
- issue-summary/news-impact enrichment failure preserves prediction rows with
  empty display-context columns.

Core data loading, feature creation, validation, model fitting, prediction, and
artifact writing failures remain fatal. When enough state exists to create the
artifact manager, a fatal failure writes an error `pipeline_report.json`
containing the failed stage, exception type/message, and traceback, then
re-raises the exception.

### Immutable Effective Configuration

Capture `config_input = app_config_to_dict(cfg)` before validation. Signal
weight tuning creates an effective signal object with
`dataclasses.replace(cfg.signal, **tuned)` rather than mutating `cfg.signal`.
Use that effective signal for tuned OOF scoring and latest prediction.

Reports expose:

- `config_input`: original user configuration;
- `signal_weights_tuned`: tuning output;
- existing `config` as the original configuration for compatibility.

Repeated pipeline calls in one process therefore cannot inherit prior tuning
results.

### Walk-Forward Adaptive Retry

Set `MIN_REQUIRED_WALK_FORWARD_FOLDS = 3`. Run validation with input training
configuration first. If produced fold count is below three, create a copied
adaptive training configuration and retry once. Select the retry result when it
produces at least as many folds as the original. Record original/final fold
counts and adaptive retry use in diagnostics.

Neither the input `TrainingConfig` nor `AppConfig` may be mutated.

### Diagnostics Coverage

After pipeline execution, compare recorded stage keys with the 12 standard
stage names. Missing keys become diagnostics warnings rather than fatal
errors. Tests verify the standard key set and warning behavior.

### CLI Exit Codes

`main()` returns an integer and the module entry point raises
`SystemExit(main())`.

| Exit code | Meaning |
|---|---|
| `0` | Pipeline completed without report warnings |
| `1` | Fatal data, validation, model, prediction, or artifact failure |
| `2` | Pipeline completed, but report status is `warning` or `caution` |

Console-script callers receive these codes while direct `run_pipeline()` users
retain exception-based fatal error handling.

## Data Flow

1. Load configuration and immediately snapshot `config_input`.
2. Execute stages sequentially through diagnostics boundaries.
3. Optional stages convert exceptions into cautions and safe fallbacks.
4. Validation retries once when fewer than three folds are produced.
5. Signal tuning creates a copied effective signal configuration.
6. Final prediction uses effective signal configuration.
7. Artifact report writes input configuration, tuned weights, stage statuses,
   fold diagnostics, warnings, and final status.
8. CLI maps report status or fatal exception to a stable exit code.

## Error Reporting

Error reports redact neither arbitrary stack frames nor messages because they
may contain secrets. Existing secret-redaction helpers must sanitize exception
content and traceback before writing JSON. If failure occurs before an artifact
manager can be initialized, the CLI prints a concise sanitized error and exits
with code `1`.

## Testing

Use TDD for each behavior:

- input signal/training configuration remains unchanged after validation;
- tuned effective signal reaches latest prediction and report separately;
- one or two initial folds trigger adaptive retry;
- retry never replaces a better original result with a worse result;
- optional investor, external, and issue-summary failures yield cautions and
  predictions;
- core-stage failure records sanitized error details and remains fatal;
- all 12 stage keys are defined and missing keys produce warnings;
- CLI returns `0`, `1`, and `2` according to the documented contract;
- pipeline smoke test verifies diagnostics and report fields;
- documentation cross-links the KOSDAQ symbol caveat in `docs/02_data.md`.

## Compatibility

- Existing `run_pipeline()` arguments and normal return shape remain valid.
- Existing `config` and `tuned_signal` report keys remain available.
- CSV encodings and result paths remain unchanged.
- Optional-context failures never alter expected return or signal decisions.
