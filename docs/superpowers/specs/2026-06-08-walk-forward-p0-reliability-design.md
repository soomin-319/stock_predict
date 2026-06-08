# Walk-Forward P0 Reliability Design

## Scope

Implement the P0 items from `docs/walkforward_todo.md`:

1. Make overlapping-fold OOF handling explicit and deterministic.
2. Prevent evaluation targets from influencing probability calibration.
3. Never reuse the same OOF dates as both tuning and evaluation data when history is insufficient.

P1 lookback selection, aggregate metric expansion, baseline alignment, purge naming, caching, and broader diagnostics remain outside this change.

## Decisions

### OOF duplicate policy

Keep the current overlapping validation-window capability. After all folds execute, aggregate duplicate `Date + Symbol` OOF rows by averaging prediction columns.

Before aggregation, require duplicate rows to have identical actual target values and stable context values. Conflicting target values are an integrity error and stop validation.

Each aggregated row includes:

- `oof_prediction_count`: number of fold predictions averaged.
- Fold provenance represented as deterministic, sorted metadata values.

The validation result also exposes diagnostics containing raw row count, unique row count, duplicate row count, and duplicate ratio.

### Fold provenance

Each fold receives deterministic `fold_id` in chronological fold order. OOF rows carry:

- `fold_id`
- `train_start`
- `train_end`
- `valid_start`
- `valid_end`

Aggregated OOF rows retain provenance so duplicate handling remains auditable.

### Temporal tune/eval split

Split deduplicated OOF by unique normalized dates. Tune dates are strictly earlier than eval dates.

Add explicit minimum-date requirements. If either side cannot meet its minimum:

- Do not return the same data for both sides.
- Return the available safe split or empty side.
- Mark split status `insufficient_data` with reason and date counts.

Evaluation and backtesting require a non-empty valid eval split. They must not silently fall back to all OOF.

### Probability calibration

Fit probability calibration using tune OOF only. The fitted calibrator may transform:

- tune OOF, for tuning diagnostics;
- eval OOF, for unbiased evaluation;
- latest predictions, for operational output.

Evaluation targets must never be passed to calibrator fitting. Calibration reporting is separated into tune and eval sections, each with Brier score and ECE.

When tune data cannot fit a useful isotonic calibrator, use an explicit identity calibrator and report the fallback reason.

## Components

### `src/validation/walk_forward.py`

- Assign fold IDs and expose complete fold boundaries.
- Add fold provenance columns to raw OOF.
- Aggregate overlapping `Date + Symbol` rows.
- Validate duplicate target consistency.
- Produce OOF duplicate diagnostics.

Public validation helpers continue returning folds and deduplicated OOF for compatibility. Diagnostics are carried through an added structured result path used by the pipeline.

### `src/validation/support.py`

- Introduce a structured temporal split result containing tune/eval frames and status metadata.
- Introduce fit/transform calibration helpers.
- Preserve the existing convenience wrappers where practical.

### `src/pipeline.py`

Validation flow becomes:

```text
raw fold predictions
→ deduplicated OOF
→ strict temporal tune/eval split
→ fit calibrator on tune only
→ transform tune/eval
→ build scored frames
→ tune signal weights on tune only
→ backtest on eval only
```

Latest predictions use the tune-fitted calibrator returned by validation.

If eval is insufficient, pipeline still produces latest predictions and a report, but:

- backtest is not presented as valid evaluation;
- validation status is `insufficient_data`;
- report contains the blocking reason.

### Reporting

Pipeline JSON adds:

- `oof_policy`: policy version, duplicate policy, and duplicate diagnostics.
- `validation_split`: status, reason, tune/eval date and row counts.
- `probability_calibration.tune` and `.eval`: Brier/ECE diagnostics.

Existing top-level fields remain where compatibility is required, but must not imply valid evaluation when eval data is unavailable.

## Error Handling

- Conflicting actual targets for duplicate `Date + Symbol`: raise a clear integrity error.
- No OOF rows after adaptive settings: preserve current runtime error.
- Insufficient dates for strict split: return/report `insufficient_data`; never reuse dates.
- Calibration fit failure or unusable tune data: identity transform plus reported reason.

## Testing

Use TDD for every behavior change.

Unit tests cover:

- fold provenance columns;
- deterministic duplicate averaging and `oof_prediction_count`;
- conflicting duplicate targets fail;
- sequential and parallel execution produce the same deduplicated OOF;
- tune/eval dates are disjoint and ordered;
- insufficient dates never cause tune/eval reuse;
- changing eval targets cannot alter the fitted calibrator;
- tune/eval calibration metrics are separate.

Pipeline smoke tests verify:

- backtest input has unique `Date + Symbol`;
- report includes OOF policy, split status, and split calibration diagnostics;
- valid smoke data still produces evaluation;
- insufficient eval data is explicitly reported rather than evaluated using reused OOF.

## Compatibility and Guardrails

- Prediction and recommendation decisions remain based on expected return policy; news/disclosures remain display-only.
- Generated artifacts remain under `result/`.
- Existing CLI options and default overlapping fold settings remain unchanged.
- No P1/P2 behavior is introduced in this change.
