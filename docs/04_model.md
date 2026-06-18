# 04. 모델 학습 - LightGBM 멀티 헤드

`src/models/lgbm_heads.py`는 예측 파이프라인의 모델 학습, 예측, 영속화, 피처 중요도 내보내기를 담당합니다. 산출물은 리서치 및 운영 보조 용도일 뿐이며, 투자 자문이나 자동 매매 시스템이 아닙니다.

## 핵심 가드레일

- 매수/매도/보유 판단은 익일 기대수익률(`predicted_return`)만 사용해야 합니다.
- `signal_score`, `up_probability`, `uncertainty_score`, 뉴스, 공시, 이슈 요약은 추천 라벨을 변경해서는 안 됩니다.
- 뉴스와 공시는 표시 전용 컨텍스트입니다. 모델 피처, 기대수익률, 랭킹, 추천, 시그널을 변경해서는 안 됩니다.
- `src.features.feature_selection.DISPLAY_ONLY_CONTEXT_COLUMNS`는 뉴스/공시 컨텍스트 컬럼을 모델 피처에서 제외합니다.

## 주요 클래스

### `MultiHeadStockModel`

```python
class MultiHeadStockModel:
    def __init__(
        self,
        random_state=42,
        n_jobs=-1,
        use_gpu=False,
        head_n_jobs=1,
        early_stopping_rounds=0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=20,
    )
    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        quantiles: list[float],
        eval_df: pd.DataFrame | None = None,
    ) -> None
    def predict(self, df: pd.DataFrame) -> MultiHeadPrediction
    def feature_importance_frame(self) -> pd.DataFrame
    def save(self, path: str | Path) -> Path
    def load(cls, path: str | Path) -> MultiHeadStockModel
```

이 모델은 회귀 헤드 1개, 이진 방향 분류기 1개, 그리고 분위수 회귀 헤드들을 학습합니다.

LightGBM을 사용할 수 없는 경우, 코드는 scikit-learn의 `GradientBoostingClassifier` / `GradientBoostingRegressor`로 폴백합니다. 이 폴백은 LightGBM과 속도나 정확도가 동등하지 않으므로, `pipeline_report.json`의 `model.backend`를 확인하세요.

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

모델 내부의 `predicted_return`은 로그 수익률입니다. 표시용 퍼센트 수익률은 `np.expm1(predicted_log_return) * 100`으로 계산됩니다.

## 모델 헤드

| 헤드 | 알고리즘 | 타깃 | 출력 |
|------|-----------|--------|--------|
| 회귀 | LGBMRegressor, `objective=regression` | `target_log_return` | `predicted_return` |
| 방향 | LGBMClassifier, `objective=binary` | `target_up` | `up_probability` |
| 하위 분위수 | LGBMRegressor, `objective=quantile` | `target_log_return` | `quantile_low` |
| 중간 분위수 | LGBMRegressor, `objective=quantile` | `target_log_return` | `quantile_mid` |
| 상위 분위수 | LGBMRegressor, `objective=quantile` | `target_log_return` | `quantile_high` |

분위수는 `TrainingConfig.quantiles`에서 가져옵니다. 기본값은 `[0.1, 0.5, 0.9]`입니다. low/mid/high는 하드코딩된 10/50/90 라벨이 아니라 설정된 하위·중간·상위 값을 의미합니다.

## 학습 설정

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
    early_stopping_rounds: int = 0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    min_child_samples: int = 20
    purge_gap_days: int = 1
    embargo_days: int = 0
    final_model_lookback_days: int = 252 * 3
    walk_forward_lookback_days: int = 0
```

- `final_model_lookback_days`: 최종 예측 모델에 사용하는 최근 거래일 수입니다. `0`은 전체 이력을 의미합니다.
- `walk_forward_lookback_days`: 워크 포워드 폴드별로 사용하는 최근 거래일 수입니다. `0`은 확장 윈도우 정책을 유지하고, 양수 값은 롤링 윈도우를 사용합니다.
- `early_stopping_rounds`: LightGBM 조기 종료 라운드입니다. `0`이면 비활성화합니다.
- `reg_alpha`, `reg_lambda`, `min_child_samples`: LightGBM 정규화와 최소 리프 샘플 설정입니다. sklearn 폴백에서는 모델 metadata에 값만 기록되고 학습 파라미터로 쓰이지 않습니다.

## 워크 포워드 학습 vs 최종 학습

### 워크 포워드 검증

- 각 폴드는 새로운 `MultiHeadStockModel`을 생성하고 학습합니다.
- 기본 학습 데이터는 `train_end_date`까지의 전체 이력입니다.
- `purge_gap_days`는 검증 윈도우 근처의 타깃 누수를 방지합니다.
- `embargo_days`는 검증 시작 시점을 앞으로 이동시킬 수 있습니다.
- `walk_forward_lookback_days > 0`이면 각 폴드를 `train_end_date` 이전 최근 N개 거래일로 제한합니다.
- MAE, RMSE, AUC 같은 폴드 지표는 `FoldResult`에 저장됩니다.

### 최종 예측 모델

```python
train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
if cfg.training.final_model_lookback_days > 0:
    cutoff_dates = sorted(train_df["Date"].unique())[-cfg.training.final_model_lookback_days:]
    train_df = train_df[train_df["Date"].isin(cutoff_dates)]
model = MultiHeadStockModel(...)
model.fit(train_df, feature_columns, cfg.training.quantiles, eval_df=None)
latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
latest_pred = model.predict(latest)
```

최종 모델은 설정된 최종 윈도우로 학습한 뒤, 각 종목의 최신 행을 예측합니다.

## 결측 피처 처리

`predict()`는 더 이상 무조건적인 `fillna(0)`을 적용하지 않습니다.

- `fit()`은 피처별 임퓨터 값을 저장합니다.
- 기본 임퓨터 값은 학습 데이터의 중앙값입니다.
- 오실레이터 계열 피처는 정의된 경우 중립 기본값을 사용합니다: `rsi_14=50`, `stoch_k=50`, `stoch_d=50`, `cci_20=0`.
- `save()`와 `load()`는 임퓨터 값을 영속화하고 복원합니다.

학습 검증은 다음 경우에 대해 명확한 `ValueError`를 발생시킵니다:

- 빈 `feature_columns`,
- 누락된 타깃 또는 피처 컬럼,
- 사용 가능한 학습 행 없음,
- 단일 클래스 `target_up`.

## 분위수 단조성과 불확실성

분위수 헤드는 독립적으로 학습되므로 원시 출력이 교차할 수 있습니다. `predict()`는 반환하기 전에 선택된 분위수 출력을 행 단위로 정렬합니다:

```text
quantile_low <= quantile_mid <= quantile_high
```

불확실성 폭은 두 번째 안전 클립과 함께 계산됩니다:

```python
uncertainty_width = max(quantile_high - quantile_low, 0)
uncertainty_score = percentile_score(uncertainty_width)
confidence_score = 1 - uncertainty_score
```

`uncertainty_score`는 `[0, 1]` 범위의 횡단면 백분위 점수입니다. 값이 클수록 상대적으로 불확실성이 높음을 의미합니다.

## 확률 보정

워크 포워드 OOF `up_probability` 값은 원시 모델 확률이며, 사용 전에 보정됩니다.

```python
def fit_up_probability_calibrator(tune_df: pd.DataFrame) -> UpProbabilityCalibrator
def calibrate_up_probability(oof_df, up_probs) -> pd.Series
```

- OOF 예측은 튜닝과 평가 파티션으로 분할됩니다.
- 튜닝 파티션은 isotonic 또는 폴백 보정을 학습합니다.
- 최종 예측은 `probability_calibrator.transform(up_probability)`를 적용합니다.

## 모델 영속화와 아티팩트

```python
model.save("result/model.pkl")
loaded = MultiHeadStockModel.load("result/model.pkl")
```

- joblib 직렬화를 사용합니다.
- `MODEL_ARTIFACT_VERSION = 2`.
- 아티팩트 버전과 피처 해시를 검증합니다.
- 임퓨터 값이 아티팩트에 포함됩니다.

기본 파이프라인은 최종 모델 아티팩트를 실행 디렉터리 아래에 저장합니다:

- `result/runs/<run_id>/model/model.pkl`
- `result/runs/<run_id>/model/model.pkl.meta.json`
- `result/runs/<run_id>/csv/model_feature_importance.csv`

`pipeline_report.json`에는 `model` 섹션과 `artifacts.model_*` 경로가 포함됩니다.

`model.backend == "sklearn"`이면 LightGBM 대비 속도와 정확도가 동등하지 않다는 경고가 `model.warnings`와 `diagnostics.warnings`에 기록됩니다.

## 피처 중요도

`MultiHeadStockModel.feature_importance_frame()`은 롱 폼(long-form) 피처 중요도 데이터를 반환합니다.

| 컬럼 | 의미 |
|--------|---------|
| `head` | `regression`, `classification`, `quantile_0.1` 등 |
| `feature` | 피처 이름 |
| `importance` | 모델이 제공하는 중요도 값 |

`feature_importances_`가 없는 헤드나 백엔드는 생략됩니다.

## 구현된 개선

### P1 - 조기 종료와 정규화

`TrainingConfig`는 `early_stopping_rounds`, `reg_alpha`, `reg_lambda`, `min_child_samples`를 제공합니다. `MultiHeadStockModel.fit(..., eval_df=None)`는 LightGBM backend에서 검증 프레임이 있고 `early_stopping_rounds > 0`일 때 `lgb.early_stopping(..., verbose=False)` 콜백을 사용합니다.

기본값은 기존 동작을 보존하도록 `early_stopping_rounds=0`이며, 고정 `n_estimators=400`은 유지됩니다. 워크 포워드/최종 모델 생성부는 새 설정을 모델에 전달합니다.

### P2 - 폴백 경고 심각도

`model.backend == "sklearn"`인 경우, sklearn 폴백은 LightGBM과 동등하지 않다는 경고를 모델 metadata와 파이프라인 diagnostics에 노출합니다.
