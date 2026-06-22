# P0 Display-only Feature Guard Design

## Goal

Strengthen the safety rail that news, disclosure, and impact context is display-only. These columns must never become model features, rankings, recommendations, or signals. Buy/sell/hold remains based on next-day `predicted_return`.

## Scope

This P0 change is intentionally small:

- Add pattern-based regression coverage around `select_feature_columns()`.
- Verify newly introduced `news_*`, `*_news_*`, `*_impact_*`, and disclosure-context style columns are excluded even when they are not in the explicit display-only column list.
- Do not change recommendation, ranking, signal-score, or report behavior unless the new test exposes a real guard gap.

## Approach

Add a deterministic pytest case in `tests/test_display_only_feature_guard.py`.

The test will build a synthetic `DataFrame` containing normal model features plus plausible future display-only columns such as:

- `news_sentiment_raw`
- `latest_news_headline`
- `foo_impact_score`
- `disclosure_impact_label`

It will call `src.features.feature_selection.select_feature_columns()` and assert that only valid model feature columns remain.

## Components

- `src/features/feature_selection.py`: Existing source of truth for model feature selection. Prefer no code change if current pattern guards already satisfy the new test.
- `tests/test_display_only_feature_guard.py`: Add the new pattern regression test.

## Data Flow

1. A candidate feature matrix arrives with mixed model and context columns.
2. `select_feature_columns()` applies base feature allowlisting and display-only exclusion.
3. The model receives only selected feature columns.
4. Display-only context remains available for reports/chatbot output only.

## Error Handling

No new runtime error path. This is test-only unless a guard bug appears. If a bug appears, fix by tightening feature-selection exclusion rules while preserving existing public behavior.

## Testing

Run:

```bash
pytest tests/test_display_only_feature_guard.py
pytest tests/test_feature_module_boundaries.py
pytest tests/test_pipeline_smoke.py
```

The new test proves future display-only column names cannot silently enter model features through naming drift.

## Out of Scope

- Refactoring chatbot or pipeline helpers.
- Moving signal thresholds into config.
- Changing published output schema.
- Changing actual recommendations or investment signals.
