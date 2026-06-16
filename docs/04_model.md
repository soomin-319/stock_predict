# 04. Model Training - LightGBM Multi-Head

`src/models/lgbm_heads.py` owns model training, prediction, persistence, and feature-importance export for the prediction pipeline. Outputs are research and operations support only; they are not investment advice or an automated trading system.

## Core guardrails

- Buy/sell/hold decisions must use next-day expected return only: `predicted_return`.
- `signal_score`, `up_probability`, `uncertainty_score`, news, disclosures, and issue summaries must not change the recommendation label.
- News and disclosures are display-only context. They must not change model features, expected returns, rankings, recommendations, or signals.
- `src.features.feature_selection.DISPLAY_ONLY_CONTEXT_COLUMNS` excludes news/disclosure context columns from model features.

## Main classes

### `MultiHeadStockModel`

```python
class MultiHeadStockModel:
    def __init__(self, random_state=42, n_jobs=-1, use_gpu=False, head_n_jobs=1)
    def fit(self, df: pd.DataFrame, feature_columns: list[str], quantiles: list[float]) -> None
    def predict(self, df: pd.DataFrame) -> MultiHeadPrediction
    def feature_importance_frame(self) -> pd.DataFrame
    def save(self, path: str | Path) -> Path
    def load(cls, path: str | Path) -> MultiHeadStockModel
```

The model trains one regression head, one binary direction classifier, and quantile regression heads.

If LightGBM is unavailable, the code falls back to scikit-learn `GradientBoostingClassifier` / `GradientBoostingRegressor`. The fallback is not speed- or accuracy-equivalent to LightGBM; check `model.backend` in `pipeline_report.json`.

### `MultiHeadPrediction`

```python
@dataclass
class MultiHeadPrediction:
    predicted_return: np.ndarray   # next-day predicted log return
    up_probability: np.ndarray     # probability of positive next-day return
    quantile_low: np.ndarray       # lower configured quantile
    quantile_mid: np.ndarray       # middle configured quantile
    quantile_high: np.ndarray      # upper configured quantile
```

`predicted_return` inside the model is log return. Display percentage return is computed as `np.expm1(predicted_log_return) * 100`.

## Model heads

| Head | Algorithm | Target | Output |
|------|-----------|--------|--------|
| Regression | LGBMRegressor, `objective=regression` | `target_log_return` | `predicted_return` |
| Direction | LGBMClassifier, `objective=binary` | `target_up` | `up_probability` |
| Lower quantile | LGBMRegressor, `objective=quantile` | `target_log_return` | `quantile_low` |
| Middle quantile | LGBMRegressor, `objective=quantile` | `target_log_return` | `quantile_mid` |
| Upper quantile | LGBMRegressor, `objective=quantile` | `target_log_return` | `quantile_high` |

Quantiles come from `TrainingConfig.quantiles`. The default is `[0.1, 0.5, 0.9]`. Low/mid/high mean the lower, middle, and upper configured values, not hard-coded 10/50/90 labels.

## Training configuration

```python
@dataclass
class TrainingConfig:
    min_train_size: int = 252 * 3
    test_size: int = 252
    step_size: int = 126
    quantiles: list[float] = [0.1, 0.5, 0.9]
    random_state: int = 42
    model_n_jobs: int = -1
    model_head_n_jobs: int = 1
    walk_forward_n_jobs: int = -1
    use_gpu: bool = False
    purge_gap_days: int = 1
    embargo_days: int = 0
    final_model_lookback_days: int = 252 * 3
    walk_forward_lookback_days: int = 0
```

- `final_model_lookback_days`: recent trading days used for the final prediction model. `0` means full history.
- `walk_forward_lookback_days`: recent trading days used per walk-forward fold. `0` keeps the expanding-window policy; positive values use a rolling window.

## Walk-forward training vs final training

### Walk-forward validation

- Each fold creates and trains a fresh `MultiHeadStockModel`.
- Default training data is all history up to `train_end_date`.
- `purge_gap_days` prevents target leakage near the validation window.
- `embargo_days` can shift validation start forward.
- `walk_forward_lookback_days > 0` limits each fold to the most recent N trading dates before `train_end_date`.
- Fold metrics such as MAE, RMSE, and AUC are stored in `FoldResult`.

### Final prediction model

```python
train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
if cfg.training.final_model_lookback_days > 0:
    cutoff_dates = sorted(train_df["Date"].unique())[-cfg.training.final_model_lookback_days:]
    train_df = train_df[train_df["Date"].isin(cutoff_dates)]
model = MultiHeadStockModel(...)
model.fit(train_df, feature_columns, cfg.training.quantiles)
latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
latest_pred = model.predict(latest)
```

The final model trains on the configured final window, then predicts the latest row for each symbol.

## Missing feature handling

`predict()` no longer applies unconditional `fillna(0)`.

- `fit()` stores per-feature imputer values.
- Default imputer value is the training median.
- Oscillator-like features use neutral defaults where defined: `rsi_14=50`, `stoch_k=50`, `stoch_d=50`, `cci_20=0`.
- `save()` and `load()` persist and restore imputer values.

Training validation raises clear `ValueError`s for:

- empty `feature_columns`,
- missing target or feature columns,
- no usable training rows,
- single-class `target_up`.

## Quantile monotonicity and uncertainty

Quantile heads are trained independently, so raw outputs can cross. `predict()` sorts the selected quantile outputs row-wise before returning:

```text
quantile_low <= quantile_mid <= quantile_high
```

Uncertainty width is computed with a second safety clip:

```python
uncertainty_width = max(quantile_high - quantile_low, 0)
uncertainty_score = percentile_score(uncertainty_width)
confidence_score = 1 - uncertainty_score
```

`uncertainty_score` is a cross-sectional percentile score in `[0, 1]`. Larger means relatively higher uncertainty.

## Probability calibration

Walk-forward OOF `up_probability` values are raw model probabilities and are calibrated before use.

```python
def fit_up_probability_calibrator(tune_df: pd.DataFrame) -> UpProbabilityCalibrator
def calibrate_up_probability(oof_df, up_probs) -> pd.Series
```

- OOF predictions are split into tuning and evaluation partitions.
- The tuning partition fits isotonic or fallback calibration.
- Final predictions apply `probability_calibrator.transform(up_probability)`.

## Model persistence and artifacts

```python
model.save("result/model.pkl")
loaded = MultiHeadStockModel.load("result/model.pkl")
```

- joblib serialization is used.
- `MODEL_ARTIFACT_VERSION = 2`.
- Artifact version and feature hash are checked.
- Imputer values are included in the artifact.

The default pipeline saves final model artifacts under the run directory:

- `result/runs/<run_id>/model/model.pkl`
- `result/runs/<run_id>/model/model.pkl.meta.json`
- `result/runs/<run_id>/csv/model_feature_importance.csv`

`pipeline_report.json` includes a `model` section and `artifacts.model_*` paths.

## Feature importance

`MultiHeadStockModel.feature_importance_frame()` returns long-form feature importance data.

| Column | Meaning |
|--------|---------|
| `head` | `regression`, `classification`, `quantile_0.1`, etc. |
| `feature` | feature name |
| `importance` | model-provided importance value |

Heads or backends without `feature_importances_` are omitted.

## Remaining improvement candidates

### P1 - Early stopping and regularization

LightGBM still uses fixed `n_estimators=400`, and the current model API does not accept validation frames. A future change can add `fit(..., eval_df=None)`, fold-internal validation splits, `early_stopping_rounds`, `reg_alpha`, `reg_lambda`, and `min_child_samples` in `TrainingConfig`.

### P2 - Fallback warning severity

If `model.backend == "sklearn"`, the report can expose a stronger warning because sklearn fallback is not equivalent to LightGBM.
