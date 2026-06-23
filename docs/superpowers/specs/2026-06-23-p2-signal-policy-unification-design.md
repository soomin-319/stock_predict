# P2 Signal Policy Row/Vectorized Unification Design

Date: 2026-06-23
Branch: `p2-signal-policy-unification`
Base: `p2-signal-config-thresholds`

## Goal

Unify duplicated row-wise and vectorized policy logic in `src/domain/signal_policy.py` so policy behavior has one source of truth. Preserve all public outputs and safety guardrails.

## Non-goals

- Do not change recommendation semantics.
- Do not change event boost math.
- Do not add news/disclosure inputs to recommendations, rankings, or signals.
- Do not refactor unrelated pipeline, publish, or chatbot code.

## Guardrails

- Buy/sell/hold recommendation remains based only on next-day `predicted_return` plus `SignalConfig` thresholds.
- `signal_score`, `up_probability`, `uncertainty_score`, news, disclosures, and display-only context must not affect recommendation.
- News/disclosure columns remain display-only.
- Existing Korean output labels must remain unchanged.

## Current Problem

`signal_policy.py` maintains two implementations for similar logic:

- Scalar row helpers: `risk_flag`, `_position_size_hint`, `prediction_reason`, `_jongbae_score`, `build_pm_summary_fields`.
- Vectorized helpers: `_risk_flag_series`, `_position_size_hint_series`, `_prediction_reason_series`, `_jongbae_score_series`, `_pm_summary_frame`.

The main pipeline uses vectorized helpers. Tests and compatibility callers use scalar helpers. Keeping both as independent logic risks future drift.

## Recommended Approach

Use vectorized helpers as the source of truth. Re-implement scalar helpers as 1-row adapters over vectorized helpers.

### Adapter Pattern

Add small internal helpers:

- `_row_frame(row: pd.Series) -> pd.DataFrame`: wraps a row in a single-row frame while preserving columns and index.
- `_first_scalar(series: pd.Series)`: returns the first value from a 1-row series.
- `_first_record(frame: pd.DataFrame) -> dict[str, str]`: returns first record from a 1-row frame.

Then rewrite scalar helpers:

- `risk_flag(row)` calls `_risk_flag_series(_row_frame(row)).iloc[0]`.
- `prediction_reason(row, cfg)` calls `_prediction_reason_series(_row_frame(row), cfg).iloc[0]`.
- `_jongbae_score(row, cfg)` calls `_jongbae_score_series(_row_frame(row), cfg).iloc[0]`.
- `build_pm_summary_fields(row, cfg, signal_cfg)` calls `_pm_summary_frame(_row_frame(row), cfg, signal_cfg).iloc[0].to_dict()`.
- `_position_size_hint(confidence_score, risk_flag_value)` may stay as a small compatibility helper or use a tiny frame adapter, because it has a narrower signature than row policy. It is not called by the pipeline after `build_pm_summary_fields` changes.

## Data Flow

Pipeline path remains:

1. `build_prediction_policy_frame(pred_df, cfg, signal_cfg)`
2. `vectorized_event_signal_boost(pred_df, cfg)`
3. `_pm_summary_frame(out, cfg, signal_cfg)`
4. `_jongbae_score_series(out, cfg)`
5. `_prediction_reason_series(out, cfg)`

Scalar compatibility path becomes:

1. caller passes row
2. row wrapped into 1-row DataFrame
3. same vectorized helper computes output
4. first value returned

## Testing Strategy

Add tests before implementation:

1. Confirm scalar `risk_flag(row)` equals `_risk_flag_series(frame).iloc[0]` for representative rows.
2. Confirm scalar `prediction_reason(row)` equals `_prediction_reason_series(frame).iloc[0]`.
3. Confirm scalar `_jongbae_score(row)` equals `_jongbae_score_series(frame).iloc[0]`.
4. Confirm scalar `build_pm_summary_fields(row)` equals `_pm_summary_frame(frame).iloc[0].to_dict()`.
5. Confirm custom `SignalConfig` thresholds still flow through scalar and vectorized recommendation paths.
6. Confirm recommendation-only guard still passes: changing `signal_score`, `up_probability`, `uncertainty_score`, news-like columns does not change recommendation when `predicted_return` is constant.

Existing tests should continue to pass:

- `tests/test_signal_policy.py`
- `tests/test_signal_policy_contract.py`
- `tests/test_signal_policy_recommendation.py`
- `tests/test_pipeline_smoke.py`

## Compatibility

Public helper names and return types remain unchanged:

- `risk_flag(row) -> str`
- `prediction_reason(row, cfg=None) -> str`
- `recommendation_from_signal(...) -> str`
- `build_pm_summary_fields(row, cfg=None, signal_cfg=None) -> dict[str, str]`
- `build_prediction_policy_frame(pred_df, cfg=None, signal_cfg=None) -> pd.DataFrame`

Private helpers may be rearranged, but tests can keep using private helpers where current tests already do.

## Risks and Mitigations

- Risk: 1-row DataFrame conversion changes missing-value behavior.
  - Mitigation: tests cover NaN/missing columns and current representative policy rows.
- Risk: scalar adapter has small overhead.
  - Mitigation: scalar helpers are compatibility/test/report helpers; pipeline remains vectorized.
- Risk: output label drift due ordering differences.
  - Mitigation: exact string equality tests for PM summary, risk flags, reasons, and jongbae score.

## Verification

Run:

```powershell
pytest tests/test_signal_policy.py tests/test_signal_policy_contract.py tests/test_signal_policy_recommendation.py tests/test_pipeline_smoke.py -q
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
pytest -q
```

Expected: all tests pass. Sample pipeline exits 0.
