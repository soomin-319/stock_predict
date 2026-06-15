# 04. 모델 학습 — LightGBM 멀티헤드

`src/models/lgbm_heads.py`는 예측 파이프라인의 학습 및 추론 담당 모듈이다.

## 핵심 클래스

### `MultiHeadStockModel`

```python
# src/models/lgbm_heads.py:72
class MultiHeadStockModel:
    def __init__(self, random_state=42, n_jobs=-1, use_gpu=False, head_n_jobs=1)
    def fit(self, df: pd.DataFrame, feature_columns: list[str], quantiles: list[float]) -> None
    def predict(self, df: pd.DataFrame) -> MultiHeadPrediction
    def save(self, path: str | Path) -> None
    def load(cls, path: str | Path) -> MultiHeadStockModel
```

회귀 + 방향 분류 + 분위수 추정을 동시에 학습하는 멀티헤드 구조.

**LightGBM 미설치 시**: scikit-learn `GradientBoostingClassifier` / `GradientBoostingRegressor`로 자동 fallback.

### `MultiHeadPrediction`

```python
# src/models/lgbm_heads.py:64
@dataclass
class MultiHeadPrediction:
    predicted_return: np.ndarray   # 다음날 예상 수익률 (%)
    up_probability: np.ndarray     # 상승 확률 (0~1)
    quantile_low: np.ndarray       # 10% 분위수 수익률
    quantile_mid: np.ndarray       # 50% 분위수 수익률
    quantile_high: np.ndarray      # 90% 분위수 수익률
```

---

## 모델 헤드 구성

| 헤드 | 알고리즘 | 타겟 | 출력 |
|------|----------|------|------|
| **회귀** | LGBMRegressor (mse) | `target_log_return` | `predicted_return` |
| **분류** | LGBMClassifier (binary) | `target_up` | `up_probability` |
| **분위수 10%** | LGBMRegressor (quantile α=0.1) | `target_log_return` | `quantile_low` |
| **분위수 50%** | LGBMRegressor (quantile α=0.5) | `target_log_return` | `quantile_mid` |
| **분위수 90%** | LGBMRegressor (quantile α=0.9) | `target_log_return` | `quantile_high` |

분위수는 `TrainingConfig.quantiles` (기본 `[0.1, 0.5, 0.9]`)로 설정 가능.

---

## 학습 설정 (`TrainingConfig`)

```python
# src/config/settings.py:35
@dataclass
class TrainingConfig:
    min_train_size: int = 252 * 3      # 최소 학습 데이터: 3년치 (~756 거래일)
    test_size: int = 252               # 검증 윈도우: 1년
    step_size: int = 126               # 윈도우 이동 간격: 반년
    quantiles: list[float] = [0.1, 0.5, 0.9]
    random_state: int = 42
    model_n_jobs: int = -1             # LightGBM 쓰레드 수 (-1 = 전체)
    model_head_n_jobs: int = 1         # 헤드 병렬 학습 수
    walk_forward_n_jobs: int = -1      # Walk-Forward 폴드 병렬 수
    use_gpu: bool = False
    purge_gap_days: int = 1            # 타겟 누수 방지용 갭
    embargo_days: int = 0              # 시리얼 상관 완충 기간
    final_model_lookback_days: int = 252 * 3  # 최종 모델 학습 최근 N일 (0=전체)
```

---

## Walk-Forward에서의 학습 vs. 최종 학습

### Walk-Forward (검증용)

- 각 폴드마다 `MultiHeadStockModel`을 새로 생성하여 학습
- 훈련 데이터: `train_end_date` 이전 전체 (purge_gap_days 반영)
- 검증 데이터: `valid_start_date` ~ `valid_end_date`
- 폴드별 메트릭(MAE, RMSE, AUC 등)을 `FoldResult`에 저장

### 최종 모델 (예측용)

```python
# src/pipeline.py:676
def _predict_pipeline_latest(...):
    train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    if lookback > 0:
        # 최근 N 거래일만 사용
        cutoff_dates = sorted(train_df["Date"].unique())[-lookback:]
        train_df = train_df[train_df["Date"].isin(cutoff_dates)]
    model = MultiHeadStockModel(...)
    model.fit(train_df, feature_columns, cfg.training.quantiles)
    latest = feat.groupby("Symbol").tail(1)  # 각 종목의 최신 행
    latest_pred = model.predict(latest)
```

전체 히스토리(또는 최근 N일)로 최종 모델 학습 후, **각 종목의 가장 최신 행**에 대해 예측.

---

## 확률 캘리브레이션

Walk-Forward OOF 예측의 `up_probability`는 raw 확률이므로 보정이 필요하다.

```python
# src/validation/support.py
def fit_up_probability_calibrator(tune_df: pd.DataFrame) -> calibrator
def calibrate_up_probability(oof_df, up_probs) -> pd.Series
```

- **캘리브레이터**: Isotonic Regression (단조 변환) 또는 Platt Scaling
- OOF를 tuning/evaluation 분할 후 tune 분할로 캘리브레이터 학습
- 최종 예측 시 `probability_calibrator.transform(up_probability)` 적용

---

## 불확실성 점수 (`uncertainty_score`)

분위수 출력에서 예측 불확실성을 계산한다:

```
uncertainty_score = quantile_high - quantile_low
```

값이 클수록 모델의 예측 신뢰도가 낮으며, 시그널 점수 계산 시 패널티로 적용된다.

---

## 모델 저장/로드

```python
# src/models/lgbm_heads.py
model.save("result/model.pkl")
loaded = MultiHeadStockModel.load("result/model.pkl")
```

- joblib을 사용한 직렬화
- `MODEL_ARTIFACT_VERSION = 1` 버전 체크 포함
- SHA-256 해시로 무결성 검증

---

## 적응형 학습 설정 (`_adaptive_training_cfg`)

데이터가 적을 때 자동으로 학습 윈도우를 줄인다:

```python
# src/pipeline.py:135
def _adaptive_training_cfg(cfg, feat: pd.DataFrame):
    uniq = len(feat["Date"].unique())
    tuned.min_train_size = min(tuned.min_train_size, max(60, int(uniq * 0.6)))
    tuned.test_size = min(tuned.test_size, max(20, int(uniq * 0.2)))
    tuned.step_size = min(tuned.step_size, max(20, tuned.test_size // 2))
```

첫 번째 walk-forward가 폴드를 생성하지 못하면 적응형 설정으로 재시도.

---

## 개선 및 수정 제안

> 우선순위: **P0(정확성) > P1(견고성) > P2(품질/성능/문서)**.

### P0 — 분위수 교차(quantile crossing) 미보정

- **문제**: 분위수 헤드는 독립적으로 학습되어 `quantile_low > quantile_mid > quantile_high`처럼 **순서가 뒤집힐 수 있다**(`lgbm_heads.py:194-200`). 이때 `uncertainty_width = quantile_high - quantile_low`(`inference/predict.py:37`)가 **음수**가 되고, 그 백분위(`uncertainty_score`)와 `confidence_score`가 왜곡된다.
- **제안**: `predict()` 반환 직전 행 단위로 `np.sort`하여 분위수 단조성을 강제. 음수 width는 0으로 클립.

### P1 — 추론 시 무조건 `fillna(0)`

- **문제**: `predict()`가 입력 피처를 `fillna(0)`(`lgbm_heads.py:183`). RSI(중립 50)·stoch 등에서 0은 중립이 아니어서 결측 종목의 예측이 한쪽으로 치우친다. 또한 결측이 많아도 조용히 통과한다.
- **제안**: 학습 시 산출한 피처별 중립값/중앙값으로 대체(impute), 결측 비율이 임계 초과 종목은 `risk_flag`로 표시.

### P1 — 조기 종료/정규화 부재로 과적합 위험

- **문제**: LightGBM이 `n_estimators=400` 고정에 `early_stopping`·검증셋·`reg_alpha/reg_lambda/min_child_samples`가 없다(`lgbm_heads._lightgbm_params`). 폴드마다 학습 크기가 달라도 트리 수는 동일.
- **제안**: 폴드 내 검증 분할로 `early_stopping_rounds` 적용, L1/L2·`min_child_samples` 노출. 하이퍼파라미터를 `TrainingConfig`로 승격.

### P1 — sklearn 폴백 동작 차이 문서화/검증

- **문제**: LightGBM 미설치 시 `GradientBoostingRegressor/Classifier`로 폴백하지만 `n_estimators`(250 vs 400)·정규화·확률 보정 특성이 달라 결과가 체계적으로 달라진다. 분위수 회귀 GBDT는 대용량에서 매우 느리다.
- **제안**: 폴백 사용 시 `pipeline_report.json`의 `backend` 필드를 경고로 승격, 폴백 경로 전용 스모크 테스트 추가, 문서에 "정확도·속도 비동등" 명시.

### P1 — 최종 모델 학습창과 검증창의 불일치

- **문제**: walk-forward는 **확장창(전체 과거)**으로 학습하지만, 최종 모델은 `final_model_lookback_days`(기본 3년)만 사용(`pipeline.py:691-695`). 검증에서 본 데이터 양과 실제 배포 모델의 데이터 양이 달라, 검증 메트릭이 배포 모델 성능을 대표하지 못할 수 있다.
- **제안**: walk-forward에도 동일 lookback(롤링창) 옵션을 제공해 "검증=배포" 정합을 맞추거나, 두 설정의 차이를 리포트에 명시.

### P2 — `confidence_score` 단위/스케일 점검

- **문제**: `confidence_score = 1 - uncertainty_score`(`pipeline_support.py:109`). `uncertainty_score`는 width의 **백분위(0~1)**이므로 정의상 동작하지만(`inference/predict.py:43`), 분위수 교차(P0) 미보정 시 음수 width가 섞이면 신뢰도가 비단조가 된다. P0 수정과 함께 검증 필요.

### P2 — 모델 영속화가 기본 파이프라인에서 미사용

- **문제**: `save()/load()`와 `MODEL_ARTIFACT_VERSION`, feature-hash 검증이 구현되어 있으나(문서 04 "모델 저장/로드"), `run_pipeline` 본 경로는 매 실행 재학습하며 모델을 저장하지 않는 것으로 보인다.
- **제안**: 최종 모델을 `result/<run_id>/model.pkl`로 저장해 재현·핫리로드(`realtime_close_betting`) 가능하게 하거나, 문서에서 "현재 미저장"임을 명시.

### P2 — 피처 중요도/설명가능성 산출 부재

- 분위수·분류·회귀 헤드의 `feature_importances_`를 리포트/그림으로 노출하면 디버깅과 신뢰도 평가에 유용하다.
