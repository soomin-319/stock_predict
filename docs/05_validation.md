# 05. Walk-Forward 검증 및 백테스트

`src/validation/` 패키지는 모델 성능 검증, 백테스트, 신호 튜닝, 기준선 비교를 담당한다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `walk_forward.py` | Walk-Forward OOF 검증 |
| `backtest.py` | Long-only Top-K 백테스트 |
| `baselines.py` | 단순 기준선 평가 |
| `metrics.py` | 회귀/분류 메트릭 계산 |
| `support.py` | OOF 처리, 캘리브레이션, 분할 |
| `signal_tuning.py` | 시그널 가중치 튜닝 |
| `result_validity.py` | 백테스트 결과 유효성 검사 |

---

## Walk-Forward 검증 (`walk_forward.py`)

### 핵심 구조

```python
# src/validation/walk_forward.py
@dataclass
class FoldResult:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    metrics: dict[str, float]
    train_start: pd.Timestamp | None = None
    fold_id: int | None = None

@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oof: pd.DataFrame        # Out-of-Fold 예측 전체
    oof_diagnostics: dict

def walk_forward_validate_result(
    feat: pd.DataFrame,
    feature_columns: list[str],
    cfg: TrainingConfig,
) -> WalkForwardResult
```

### 폴드 생성 로직

```
전체 거래일: [d0, d1, ..., dN]

폴드 예시 (min_train_size=756, test_size=252, step_size=126):
  Fold 1: train=[d0..d755], purge_gap=1일, valid=[d757..d1008]
  Fold 2: train=[d0..d881], purge_gap=1일, valid=[d883..d1134]
  ...
```

- **purge_gap_days=1**: 타겟(다음날 수익률)과 검증 윈도우 겹침 방지
- **embargo_days=0**: 기본 비활성화, 시리얼 상관 우려 시 증가
- 폴드 병렬 실행: `walk_forward_n_jobs=-1` (기본 CPU 전체 활용)

### 폴드별 메트릭

각 `FoldResult.metrics`에 포함:

| 메트릭 | 유형 |
|--------|------|
| `mae` | 회귀: 평균 절대 오차 |
| `rmse` | 회귀: 제곱근 평균 제곱 오차 |
| `r2` | 회귀: 결정계수 |
| `ic` | 회귀: 정보계수 (스피어만 상관) |
| `auc` | 분류: ROC-AUC |
| `accuracy` | 분류: 정확도 |
| `f1` | 분류: F1 점수 |

---

## OOF 처리 (`support.py`)

```python
# src/validation/support.py
def split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7, ...) -> OOFSplit
def fit_up_probability_calibrator(tune_df) -> calibrator
def calibrate_up_probability(oof_df, up_probs) -> pd.Series
def calibration_split_metrics(tune_df, eval_df, calibrator) -> dict
def compute_oof_diagnostics(scored_oof) -> dict
def prediction_from_oof_df(oof: pd.DataFrame) -> MultiHeadPrediction
```

OOF를 7:3 비율로 분할:
- **tuning split (70%)**: 시그널 가중치 튜닝, 캘리브레이터 학습
- **eval split (30%)**: 최종 백테스트

`compute_oof_diagnostics()`는 중복 OOF 행, 날짜 범위, 종목 수 등 품질 지표를 반환한다.

---

## 백테스트 (`backtest.py`)

```python
# src/validation/backtest.py
def run_long_only_topk_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> dict
```

### 백테스트 전략

- **전략**: Long-only (매수만), Top-K 종목 선택
- **선택 기준**: `predicted_return` 내림차순 우선, `signal_score` 내림차순 보조 정렬 (`top_k=20`)
- **중요 원칙**: 매수/매도/보유 판단과 순위는 다음날 기대수익률(`predicted_return`)을 기준으로 한다. 뉴스·공시 정보는 표시용 컨텍스트이며 기대수익률, 순위, 추천, 신호를 바꾸지 않는다.
- **필터 조건**:
  - `up_probability >= min_up_probability` (기본 0.50)
  - `value_traded >= min_value_traded` (기본 30억원)
  - `max_capacity_notional >= portfolio_value / top_k`
  - `coverage_gate_status != "halt"`

### BacktestConfig

```python
# src/config/settings.py:80
@dataclass
class BacktestConfig:
    top_k: int = 20                    # 선택 종목 수
    portfolio_value: float = 1_000_000_000  # 포트폴리오 규모 (10억원)
    max_daily_participation: float = 0.10   # 일 거래대금 대비 최대 참여율
    fee_bps: float = 10.0              # 거래 수수료 (bps)
    slippage_bps: float = 5.0          # 슬리피지
    dynamic_slippage_bps: float = 10.0 # 동적 슬리피지 계수
    conservative_slippage_multiplier: float = 1.5
    aggressive_slippage_multiplier: float = 0.75
    min_up_probability: float = 0.50   # 최소 상승 확률
    min_signal_score: float = 0.0      # 현재 백테스트 필터에는 미적용
    turnover_limit: float = 0.5        # 일 최대 회전율
    min_value_traded: float = 3_000_000_000  # 최소 거래대금 (30억원)
    min_external_coverage_ratio: float = 0.0
    min_investor_coverage_ratio: float = 0.5
    max_positions_per_market_type: int = 12  # 시장별 최대 보유 종목 수
```

### 백테스트 출력

```python
{
    "days": int,                   # 총 백테스트 거래일 수
    "cum_return": float,           # 누적 수익률
    "avg_daily_return": float,     # 평균 일간 수익률
    "sharpe": float,               # 샤프 비율
    "max_drawdown": float,         # 최대 낙폭
    "avg_turnover": float,         # 평균 회전율
    "avg_selected_count": float,   # 평균 선택 종목 수
    "benchmark_cum_return": float, # 벤치마크(KOSPI) 누적 수익률
    "excess_cum_return": float,    # 초과 수익률
    "cost_scenarios": dict,        # 비용 시나리오별 누적 수익률
    "halted_days": int,            # 커버리지 게이트로 정지된 날 수
    "liquidity_blocked_days": int, # 유동성/용량 필터로 선택 종목이 없는 날 수
    "avg_market_type_count": float,# 일평균 선택 시장 유형 수
    "series": list[dict],          # 일별 시계열 데이터
}
```

### 커버리지 게이트

```python
# src/validation/backtest.py:25
def coverage_gate_status(cfg, external_coverage_ratio, investor_coverage_ratio) -> str
```

| 상태 | 조건 | 효과 |
|------|------|------|
| `normal` | 커버리지 모두 정상 | 백테스트 정상 실행 |
| `caution` | 외부·투자자 커버리지 중 하나라도 `max(0.7, min_*_coverage_ratio)` 미만 | 경고만 표시, 실행 계속 |
| `halt` | 최소 비율 미달 | 해당 날짜 거래 중단 |

---

## 기준선 평가 (`baselines.py`)

```python
# src/validation/baselines.py
def evaluate_baselines(feat: pd.DataFrame) -> dict
```

모델 성능과 비교하기 위한 단순 기준선들:

| 기준선 | 설명 |
|--------|------|
| `baseline_zero` | 항상 0 예측, 상승확률 0.5 |
| `baseline_prev_return` | 전일 `log_return`을 다음날 수익률 예측값으로 사용 |

---

## 시그널 가중치 튜닝 (`signal_tuning.py`)

```python
# src/validation/signal_tuning.py
def tune_signal_weights(tune_df: pd.DataFrame) -> dict[str, float]
```

OOF tune 분할에서 시그널 점수의 가중치를 최적화한다:

```
signal_score = return_weight × norm_return
             + up_prob_weight × up_probability
             - uncertainty_penalty × uncertainty_score
             + event_boost_score
```

기본 가중치 (`SignalConfig`):

| 가중치 | 기본값 |
|--------|--------|
| `return_weight` | 0.65 |
| `up_prob_weight` | 0.35 |
| `uncertainty_penalty` | 0.25 |

---

## 백테스트 유효성 검사 (`result_validity.py`)

```python
# src/validation/result_validity.py
def evaluate_backtest_validity(backtest: dict, tradable_count: int) -> dict
```

다음 조건을 검사하여 `backtest_valid: bool` 반환:

- 거래 가능 예측 행 존재 (`tradable_prediction_count > 0`)
- 평가일 존재 (`days > 0`)
- 전체 평가일이 커버리지 게이트로 중단되지 않음 (`halted_days < days`)
- 평균 선택 종목 수가 0보다 큼 (`avg_selected_count > 0`)

현재 구현은 샤프 비율·최대 낙폭·최소 평가일 수 임계값을 차단 조건으로 사용하지 않는다.

유효하지 않으면 `pipeline_report.json`에 `status: "warning"`와 `blocking_reasons` 목록 추가.

---

## 메트릭 계산 (`metrics.py`)

```python
# src/validation/metrics.py
def regression_metrics(y_true, y_pred) -> dict
def classification_metrics(y_true, y_prob) -> dict
```

| 유형 | 메트릭 |
|------|--------|
| 회귀 | MAE, RMSE, R², IC (스피어만 상관계수) |
| 분류 | AUC, Accuracy, F1, Precision, Recall |

---

## 개선 및 수정 제안

> 우선순위: **P0(정확성/문서 불일치) > P1(견고성) > P2(성능/품질)**.

### P1 — 시그널 가중치 튜닝의 과적합/탐색 빈약

- **문제**: `tune_signal_weights`는 **튜닝셋 in-sample 상위 10% 수익률**을 최대화하도록 3×3 격자만 탐색하고, `up_prob_weight`를 0.30으로 **고정**한다(`signal_tuning.py:11-12`). 교차검증·정규화가 없어 노이즈에 과적합되기 쉽고, 문서의 기본값(`up_prob_weight=0.35`)과도 어긋난다.
- **제안**: 튜닝셋 내부를 다시 시계열 분할해 검증, 탐색 격자 확대 또는 베이지안 탐색, 동률 시 단순(낮은 가중치) 선호, 선택된 가중치의 in/out 성능 차를 리포트.

### P1 — 백테스트의 로그수익률/단순수익률 혼용 및 등가중 가정

- **문제**: 일일 수익률은 선택 종목의 `target_log_return` **평균**(로그수익률)인데, 자산곡선은 `(1+series).cumprod()`로 **단순수익률처럼** 복리화한다(`backtest.py:172,205`). 또한 체결 용량(`max_capacity_notional`)을 계산하고도 비중은 등가중으로 가정한다.
- **제안**: 로그/단순 정의를 통일(`expm1` 후 단순수익률로 집계), 용량 제약을 실제 비중에 반영한 가중 포트폴리오 수익률 계산.

### P1 — 비용 시나리오가 실제 경로가 아닌 평탄 복리 근사

- **문제**: `cost_scenarios`는 `gross_daily_mean - scenario_cost`를 **일정값으로 N일 복리화**한 근사다(`backtest.py:272-275`). 일자별 변동·턴오버 분포를 반영하지 않아 보수/공격 시나리오가 오도될 수 있다. 게다가 `gross_daily_mean`은 정적 비용만 역가산한다(`backtest.py:251`).
- **제안**: 시나리오 비용을 일자별 시계열에 적용해 실제 자산곡선을 재시뮬레이션.

### P1 — 폴드 병렬화의 메모리·직렬화 비용

- **문제**: `_iter_folds`가 각 폴드의 `train_df`/`valid_df` 슬라이스를 **리스트로 모두 materialize**한 뒤(`walk_forward.py:226`) `ProcessPoolExecutor`로 넘긴다. 확장창이라 후반 폴드의 `train_df`가 매우 커서 (a) 전체 슬라이스 동시 보유로 메모리 급증, (b) 프로세스 간 대용량 피클 직렬화 오버헤드가 발생한다(Windows spawn에서 특히 큼).
- **제안**: 폴드를 인덱스 경계만 전달하고 워커 내부에서 슬라이싱, 또는 공유메모리/Arrow. 폴드 lazy 생성으로 동시 보유량 축소.
