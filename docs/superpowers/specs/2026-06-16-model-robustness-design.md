# Model Robustness and Documentation Design

Date: 2026-06-16

## Scope

Implement the approved `docs/04_model.md` improvement set for the stock prediction pipeline without changing the core investment guardrails:

- Buy/sell/hold decisions remain based only on next-day `predicted_return`.
- News and disclosures remain display-only context and must not affect model features, expected returns, rankings, recommendations, or signals.
- Changes must be deterministic and covered by pytest tests.

## Design

### 1. Quantile ordering and uncertainty safety

`MultiHeadStockModel.predict()` will enforce row-wise quantile monotonicity before returning `MultiHeadPrediction`. `build_prediction_frame()` will clip `uncertainty_width` at zero as a second safety layer. This prevents quantile crossing from producing negative uncertainty widths or inverted bands.

### 2. Missing feature imputation

`MultiHeadStockModel.fit()` will compute and store per-feature imputation values from the training frame. Medians are used by default, with domain-neutral defaults for oscillator-like indicators such as RSI/stochastic/CCI and binary flags. `predict()` will use these stored values instead of unconditional `fillna(0)`. Artifact metadata/save/load will persist imputer values and increment the artifact version for compatibility.

### 3. Training validation

`fit()` will raise clear `ValueError`s for empty feature sets, no usable training rows, and single-class classification targets. This converts unclear model-backend errors into actionable user-facing errors.

### 4. Walk-forward/final-window alignment

`TrainingConfig` will gain `walk_forward_lookback_days`, default `0` for current expanding-window behavior. When positive, each validation fold will train on only the most recent N trading dates before that fold's `train_end`. Fold provenance already records `train_start`, so tests can verify rolling behavior.

### 5. Model artifact and feature importance outputs

The final model will be saved under the current artifact manager run directory as `model/model.pkl` and metadata sidecar. Feature importances from available heads will be exported as `csv/model_feature_importance.csv`. The pipeline report will include artifact paths and metadata.

### 6. Guardrail tests

Tests will verify news/disclosure columns remain excluded from model features and recommendations remain controlled by `predicted_return` only.

## Testing

- Unit tests for quantile ordering, imputation, fit validation, persistence, feature importance export, and guardrails.
- Walk-forward test for rolling training window behavior.
- Existing smoke tests plus pipeline sample command.

## Non-goals

- No investment advice or automated trading behavior.
- No news/disclosure use in model or signal scoring.
- No early stopping in this pass because current model API does not carry validation frames; this remains documented as future work.
