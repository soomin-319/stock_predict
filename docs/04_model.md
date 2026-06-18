# 04. 모델 학습 — LightGBM 멀티 헤드

`src/models/lgbm_heads.py`는 예측 파이프라인의 모델 학습, 예측, 영속화, 피처 중요도 내보내기를
담당한다. 산출물은 리서치·운영 보조용이며 투자 자문이나 자동매매 시스템이 아니다.

## 핵심 가드레일

- 매수/매도/관망 판단은 익일 기대수익률(`predicted_return`)만 사용한다.
- `signal_score`, `up_probability`, `uncertainty_score`, 뉴스, 공시는 추천 라벨을 바꾸지 않는다.
- 뉴스/공시 컨텍스트 컬럼은 `feature_selection.DISPLAY_ONLY_CONTEXT_COLUMNS`로 모델 피처에서 제외된다([03](03_features.md)).

## 주요 클래스

### `MultiHeadStockModel`

```python
class MultiHeadStockModel:
    def __init__(self, random_state=42, n_jobs=-1, use_gpu=False, head_n_jobs=1,
                 early_stopping_rounds=0, reg_alpha=0.0, reg_lambda=0.0, min_child_samples=20)
    def fit(self, df, feature_columns, quantiles, eval_df=None) -> None
    def predict(self, df) -> MultiHeadPrediction
    def feature_importance_frame(self) -> pd.DataFrame
    def save(self, path) -> Path
    @classmethod
    def load(cls, path) -> MultiHeadStockModel
```

회귀 헤드 1개, 이진 방향 분류기 1개, 분위수 회귀 헤드들을 학습한다.

LightGBM이 없으면 scikit-learn의 `GradientBoostingClassifier`/`GradientBoostingRegressor`로 폴백한다.
이 폴백은 LightGBM과 속도·정확도가 동등하지 않으므로 `model.backend`(`"lightgbm"`/`"sklearn"`)를 확인한다.

### `MultiHeadPrediction`

```python
@dataclass
class MultiHeadPrediction:
    predicted_return: np.ndarray   # 익일 예측 로그 수익률
    up_probability: np.ndarray     # 익일 상승 확률
    quantile_low: np.ndarray       # 설정된 하위 분위수
    quantile_mid: np.ndarray       # 설정된 중간 분위수
    quantile_high: np.ndarray      # 설정된 상위 분위수
```

모델 내부 `predicted_return`은 로그 수익률이다. 표시용 퍼센트 수익률은 `np.expm1(log_return) * 100`으로 계산한다.

## 모델 헤드

| 헤드 | 알고리즘 | 타깃 | 출력 |
|------|-----------|--------|--------|
| 회귀 | LGBMRegressor `objective=regression` | `target_log_return` | `predicted_return` |
| 방향 | LGBMClassifier `objective=binary` | `target_up` | `up_probability` |
| 하위/중간/상위 분위수 | LGBMRegressor `objective=quantile` | `target_log_return` | `quantile_low/mid/high` |

분위수는 `TrainingConfig.quantiles`(기본 `[0.1, 0.5, 0.9]`)에서 가져온다. low/mid/high는 하드코딩된 라벨이
아니라 설정된 하위·중간·상위 값을 의미한다.

## 학습 설정 (`TrainingConfig` 발췌)

| 설정 | 기본값 | 의미 |
|------|--------|------|
| `final_model_lookback_days` | 756 | 최종 예측 모델 학습에 쓰는 최근 거래일 수 (`0`=전체) |
| `walk_forward_lookback_days` | 0 | 폴드별 학습 윈도우 (`0`=확장창, 양수=롤링) |
| `early_stopping_rounds` | 0 | LightGBM 조기 종료 (`0`=비활성) |
| `reg_alpha`, `reg_lambda`, `min_child_samples` | 0, 0, 20 | LightGBM 정규화/최소 리프 샘플 |
| `purge_gap_days`, `embargo_days` | 1, 0 | 검증 누수 방지 갭/엠바고 |

`early_stopping_rounds > 0`이고 `eval_df`가 주어지면 LightGBM 백엔드에서 `lgb.early_stopping` 콜백을 사용한다.
기본값(`0`)은 고정 `n_estimators=400` 동작을 보존한다. sklearn 폴백에서는 정규화 값이 모델 metadata에만 기록된다.

## 워크 포워드 학습 vs 최종 학습

- **워크 포워드**: 폴드마다 새 모델을 생성/학습한다. 기본 학습 데이터는 `train_end_date`까지의 전체 이력이며,
  `purge_gap_days`로 검증 윈도우 근처 타깃 누수를 막는다. `walk_forward_lookback_days > 0`이면 롤링 윈도우.
- **최종 예측 모델**: `final_model_lookback_days`로 최근 윈도우를 잘라 학습한 뒤 각 종목의 최신 행을 예측한다.

## 결측 피처 처리

`predict()`는 무조건적인 `fillna(0)`을 적용하지 않는다.

- `fit()`이 피처별 임퓨터 값을 저장한다(기본은 학습 데이터 중앙값).
- 오실레이터 계열은 중립 기본값을 사용한다: `rsi_14=50`, `stoch_k=50`, `stoch_d=50`, `cci_20=0`.
- `save()`/`load()`가 임퓨터 값을 영속화/복원한다.

학습 검증은 다음에 대해 명확한 `ValueError`를 발생시킨다: 빈 `feature_columns`, 누락 타깃/피처 컬럼,
학습 행 없음, 단일 클래스 `target_up`.

## 분위수 단조성과 불확실성

분위수 헤드는 독립 학습되어 출력이 교차할 수 있으므로 `predict()`는 반환 전 행 단위로 정렬한다
(`quantile_low <= quantile_mid <= quantile_high`).

```python
uncertainty_width = max(quantile_high - quantile_low, 0)
uncertainty_score = percentile_score(uncertainty_width)   # [0,1] 횡단면 백분위
confidence_score  = 1 - uncertainty_score
```

## 확률 보정

워크 포워드 OOF `up_probability`는 원시 확률이며 사용 전 보정된다(`src/validation/support.py`).
OOF를 튜닝/평가로 분할하고, 튜닝 파티션에서 isotonic(또는 폴백) 보정을 학습해 최종 예측에 적용한다.

## 모델 영속화와 아티팩트

- joblib 직렬화, `MODEL_ARTIFACT_VERSION = 2`. 로드 시 아티팩트 버전과 피처 해시를 검증한다.
- 임퓨터 값이 아티팩트에 포함된다.
- 기본 파이프라인 저장 경로:
  - `result/runs/<run_id>/model/model.pkl`
  - `result/runs/<run_id>/model/model.pkl.meta.json`
  - `result/runs/<run_id>/csv/model_feature_importance.csv`
- `pipeline_report.json`에 `model` 섹션과 `artifacts.model_*` 경로가 포함된다.
- `model.backend == "sklearn"`이면 LightGBM 대비 비동등 경고가 `model.warnings`와 `diagnostics.warnings`에 기록된다.

## 피처 중요도

`feature_importance_frame()`은 롱폼 데이터를 반환한다: `head`(regression/classification/quantile_0.1 등),
`feature`, `importance`. `feature_importances_`가 없는 헤드/백엔드는 생략된다.
