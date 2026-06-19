# 05. Walk-Forward 검증 및 백테스트

`src/validation/`는 모델 성능 검증, 백테스트, 시그널 튜닝, 기준선 비교를 담당한다.

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

```python
@dataclass
class FoldResult:
    train_end / valid_start / valid_end: pd.Timestamp
    metrics: dict[str, float]
    train_start: pd.Timestamp | None
    fold_id: int | None

@dataclass
class WalkForwardResult:
    folds: list[FoldResult]
    oof: pd.DataFrame
    oof_diagnostics: dict

def walk_forward_validate_result(df, feature_columns, cfg) -> WalkForwardResult
```

### 폴드 생성·실행

- `_iter_fold_windows`는 **날짜 경계(train_end/valid_start/valid_end)만** 생성한다. 데이터 슬라이스를
  미리 materialize하지 않는다.
- `_execute_fold_windows`는 워커가 내부에서 슬라이싱하도록 경계만 전달하고, `ProcessPoolExecutor`의
  `initializer`로 DataFrame을 워커당 1회만 공유한다(폴드별 대용량 피클 직렬화 회피).
- 병렬 시 `replace(cfg, model_n_jobs=1, model_head_n_jobs=1)`로 모델 내부 스레드를 1로 제한해
  CPU 과점유(worker × model_n_jobs)를 방지한다.
- `purge_gap_days`(기본 1)로 검증 윈도우 근처 타깃 누수를 막고, `embargo_days`로 검증 시작을 앞으로 민다.
- `walk_forward_lookback_days > 0`이면 각 폴드를 최근 N 거래일로 제한(롤링 윈도우).

### OOF 집계 (`aggregate_oof_predictions`)

- 키 `(Date, Symbol)`로 중복 예측을 평균(`date_symbol_mean_v1`).
- 동일 키에서 `target_*`이 충돌하면 `ValueError`(데이터 무결성 보호).
- 중복 비율, 안정 컬럼 충돌 수 등 진단을 함께 반환한다.

### 폴드별 메트릭

`FoldResult.metrics`: 회귀(`mae`, `rmse`, `r2`, `ic`)와 분류(`auc`, `accuracy`, `f1` 등).

---

## OOF 처리 (`support.py`)

```python
def split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7, ...) -> OOFSplit
def fit_up_probability_calibrator(tune_df) -> calibrator
def calibrate_up_probability(oof_df, up_probs) -> pd.Series
def compute_oof_diagnostics(scored_oof) -> dict
```

OOF를 7:3으로 분할한다. tuning(70%)은 시그널 가중치 튜닝·캘리브레이터 학습에, eval(30%)은 최종 백테스트에 사용한다.

---

## 백테스트 (`backtest.py`)

```python
def run_long_only_topk_backtest(pred_df, cfg: BacktestConfig) -> dict
```

### 전략

- **Long-only Top-K**, 정렬 우선순위는 `predicted_return` 내림차순, 보조로 `signal_score` 내림차순.
- **매수/매도/관망과 순위는 익일 기대수익률 기준**이다. 뉴스·공시는 표시용 컨텍스트로 순위·추천을 바꾸지 않는다.
- **유동성/용량 필터**(`_apply_liquidity_and_capacity_filters`):
  - `up_probability >= min_up_probability`(기본 0.50)
  - `value_traded >= min_value_traded`(기본 30억원)
  - `max_capacity_notional(= value_traded × max_daily_participation) >= portfolio_value / top_k`
- **시장 유형 상한**: `max_positions_per_market_type`(기본 12) 적용.
- **턴오버 제한**: `turnover_limit < 1`이면 일별 신규 편입 비율을 제한.

### 수익률·비용 계산

- **수익률 정의 통일**: 일일 포트폴리오 수익률은 선택 종목의 `target_log_return`을 `np.expm1`로 단순수익률로
  변환한 뒤 가중 합산한다(`_weighted_simple_return`). 자산곡선 `(1+series).cumprod()`와 정합.
- **용량 가중**: 등가중이 아니라 종목별 체결 용량(`max_capacity_notional`)을 상한으로 한 비례 배분
  (`_capacity_weights`)으로 비중을 정한다. `invested_weight`(투자 비중 합)도 추적한다.
- **비용 시나리오**: `conservative`/`neutral`/`aggressive` 슬리피지 배수에 대해 **일자별** 비용 시계열을
  `gross_series`에 적용해 자산곡선을 재시뮬레이션한다(평탄 복리 근사가 아님).

### 출력(요약 키)

`days`, `cum_return`, `avg_daily_return`, `sharpe`, `max_drawdown`, `avg_turnover`, `avg_selected_count`,
`benchmark_cum_return`(KOSPI `ks11_ret_1d` 기반), `excess_cum_return`, `cost_scenarios`, `halted_days`,
`liquidity_blocked_days`, `avg_market_type_count`, `series`(일별 시계열).

### 커버리지 게이트

| 상태 | 조건 | 효과 |
|------|------|------|
| `normal` | 커버리지 정상 | 정상 실행 |
| `caution` | 외부·투자자 커버리지 중 하나라도 `max(0.7, min_*)` 미만 | 경고, 실행 계속 |
| `halt` | 최소 비율 미달 | 해당 날짜 거래 중단(수익률 0) |

---

## 기준선 평가 (`baselines.py`)

| 기준선 | 설명 |
|--------|------|
| `baseline_zero` | 항상 0 예측, 상승확률 0.5 |
| `baseline_prev_return` | 전일 `log_return`을 다음날 예측값으로 사용 |

## 시그널 가중치 튜닝 (`signal_tuning.py`)

```python
def tune_signal_weights(pred_df) -> dict
```

```
signal_score(점수 산식)
  = return_weight × norm_return + up_prob_weight × up_probability - uncertainty_penalty × uncertainty_score
```

- 튜닝셋 내부를 **시계열로 다시 분할**(`_time_split`, 70:30)해 train에서 점수를 만들고 valid 성능으로 선택한다.
- 격자: `return_weight ∈ {0.3,0.45,0.6,0.65}`, `up_prob_weight ∈ {0.20,0.35,0.50}`, `uncertainty_penalty ∈ {0.15,0.25,0.35}`.
- 목적함수는 상위 10% 평균 로그수익률에 **랭크 IC 보너스**와 **선택 종목 하방 패널티**를 더한 복합 점수다.
- train/valid 복합 점수 갭이 과도하고 기본값의 valid 점수가 충분히 근접하면
  기본값(`return_weight=0.65, up_prob_weight=0.35, uncertainty_penalty=0.25`)으로 후퇴한다.
- 동률 시 기본값에 가까운 단순한 조합을 선호.
- 결과에 `train/validation_top_decile_return`과 `top_decile_generalization_gap`을 포함해 과적합 정도를 노출한다.
  또한 `train/validation_rank_ic`, `train/validation_objective_score`, `objective_generalization_gap`,
  `overfit_fallback_applied`를 함께 노출한다.

## 백테스트 유효성 (`result_validity.py`)

```python
def evaluate_backtest_validity(backtest, tradable_count) -> dict
```

거래 가능 예측 행 존재, 평가일 존재, 전체 평가일이 커버리지 게이트로 중단되지 않음,
평균 선택 종목 수 > 0을 검사해 `backtest_valid: bool`을 반환한다. 유효하지 않으면
`pipeline_report.json`에 `status: "warning"`과 `blocking_reasons`가 추가된다. (샤프/낙폭/최소 평가일 임계값은
현재 차단 조건으로 쓰지 않는다.)

---

## 개선 및 수정 반영 현황

기존 분석의 P1 항목(시그널 튜닝 과적합/탐색 빈약, 백테스트 로그/단순수익률 혼용·등가중 가정,
비용 시나리오 평탄 복리 근사, 폴드 슬라이스 직렬화 비용)은 모두 현재 코드에 반영되었다.

P2 항목(튜닝 목적함수의 단순성)도 반영되었다. `tune_signal_weights`는 이제 상위 10% 평균 로그수익률만 보지 않고,
랭크 IC와 선택 종목 하방 리스크를 함께 평가한다. 또한 train/valid 복합 점수 갭이 큰 경우 기본 가중치로 후퇴해
검증 분할에 우연히 맞춘 조합이 운영 가중치로 반영되는 위험을 줄인다.
