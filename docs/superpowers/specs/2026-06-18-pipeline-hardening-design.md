# Pipeline Hardening Design

## Goal

Implement all improvement items listed in `docs/01_pipeline.md`: preserve input config reproducibility, isolate optional-stage failures, retry under-sized walk-forward validation, expose diagnostics coverage, and document CLI exit-code behavior.

## Scope

This work changes pipeline robustness only. It must not change the domain rule that buy/sell/hold decisions use next-day `predicted_return`; news/disclosure context remains display-only.

## Approach

Use a minimal safe implementation rather than a full pipeline runner rewrite.

1. Keep `cfg.signal` immutable during signal tuning by creating a tuned signal copy with `dataclasses.replace`.
2. Preserve old report fields while adding clearer aliases: `config_input` and `signal_weights_tuned`.
3. Retry walk-forward validation when fold count is below `min_required_folds` (default 3), not only when there are zero folds.
4. Extend `PipelineDiagnostics` with stage status and warnings, plus expected stage coverage validation.
5. Contain optional failures for external features and display-only prediction context. Core load/train/predict/write failures continue to fail.
6. Update tests and docs.

## Components

- `src/pipeline.py`
  - Add diagnostic stage status/warning helpers.
  - Add expected stage keys.
  - Add tuned signal copy logic.
  - Add adaptive retry metadata.
  - Add optional failure guards.
- `tests/test_pipeline_smoke.py`
  - Add deterministic unit tests for config immutability, retry behavior, diagnostics shape, and optional failure containment.
- `docs/01_pipeline.md`
  - Mark implemented improvements and add CLI exit-code table.

## Data Flow

Config loads once. Signal tuning returns `tuned`, then a separate `tuned_signal_cfg` is passed to scoring. The original `cfg` remains the source for `config` and `config_input` report fields. Diagnostics are updated as stages run and checked before report serialization.

## Error Handling

External market feature failures degrade to original price features plus failed coverage metadata. Investor/display context failures degrade to empty context or unchanged predictions with warnings. Validation, final model training, latest prediction, and artifact writing remain hard failures.

## Testing

Use TDD-style small tests before implementation. Run impacted tests, then smoke command:

```bash
pytest tests/test_pipeline_smoke.py
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```
