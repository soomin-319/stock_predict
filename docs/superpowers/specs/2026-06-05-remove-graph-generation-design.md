# Graph Generation Removal Design

## Goal

Remove all graph-generation functionality while preserving prediction, validation,
backtest, recommendation, CSV, and JSON calculations and outputs.

## Scope

Remove:

- `src/reports/visualize.py` and its graph-only tests.
- Pipeline graph-generation imports, stage, arguments, diagnostics, report fields,
  and artifact paths.
- `--figure-dir` and `--symbol-figure-limit` CLI options.
- Colab and Kakao bot graph-directory configuration and command arguments.
- The runtime `matplotlib` dependency.
- Active documentation that instructs users to generate or inspect graphs.

Preserve:

- Feature engineering, walk-forward validation, signal tuning, backtesting, final
  model training, latest predictions, recommendation policy, and display-only
  news/disclosure context.
- Existing CSV and non-graph JSON artifacts.
- Existing numeric calculations and calculated output columns.
- Archived historical documentation.

## Architecture

The graph layer is downstream of validation and backtest calculations. Delete that
layer and pass calculation results directly from validation into final prediction
and artifact writing. Symbol summary graph artifacts will no longer be produced or
reported, but symbol-summary calculations used by prediction/reporting remain
unchanged.

Pipeline progress steps will be reduced by one. The former graph stage and its
`create_figures` timing entry disappear; all calculation stages retain their
existing order and implementation.

## Interfaces

- `run_pipeline` no longer accepts `figure_dir` or `symbol_figure_limit`.
- Pipeline CLI rejects `--figure-dir` and `--symbol-figure-limit`.
- Kakao bot and Colab helpers no longer expose or forward graph-directory options.
- Pipeline report `artifacts` contains only generated non-graph files.
- Pipeline report no longer contains `visualization_note`.

## Calculation Invariance

No code under feature engineering, models, validation, inference, domain signal
policy, or recommendation logic will be changed. Regression tests will compare
core calculated outputs from the same deterministic sample input and verify that
graph removal changes only graph-related interface and artifact behavior.

## Testing

- Add/update pipeline tests to assert graph CLI options are absent, no graph
  artifacts are reported, and all numeric report sections and prediction outputs
  remain present.
- Update Colab and Kakao tests to assert graph options are not forwarded.
- Delete graph-only visualization tests.
- Run impacted tests, pipeline smoke test, then the full pytest suite.

## Documentation and Delivery

Update active user and operations documentation to remove graph commands and
artifact descriptions. Keep archive documents unchanged as historical records.
Commit focused changes and create a pull request with summary and test results.
