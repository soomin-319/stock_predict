# Stock Predict 결과보고서 (Developer Deep Dive)

> 목적: 본 문서는 개발자 관점에서 프로젝트의 전체 실행 흐름을 먼저 설명하고, 이후 주요 파일/함수별 코드 스니펫과 역할을 상세히 설명합니다.

---

## 1) 전체 동작 흐름

진입점은 `src/pipeline.py`의 `main()`/`run_pipeline()` 입니다.

```mermaid
flowchart TD
    A[CLI 인자 파싱] --> B[run_pipeline]
    B --> C[설정 로드(load_app_config)]
    C --> D[입력 로드/정제(load_ohlcv_csv, clean_ohlcv)]
    D --> E[유니버스/실데이터 수집]
    E --> F[피처 생성(build_features)]
    F --> G[외부/투자자 컨텍스트 결합]
    G --> H[시장 국면 주석]
    H --> I[워크포워드 + OOF]
    I --> J[신호 튜닝 + 백테스트]
    J --> K[최종 학습/최신 추론]
    K --> L[정책/PM 리포트/CSV/그래프 저장]
```

### 파이프라인 핵심 단계
1. AppConfig 로드 + CLI override 반영
2. OHLCV 입력 검증/정제
3. 유니버스 필터 또는 실데이터 fetch
4. 가격/모멘텀/거래량 피처 생성
5. 외부지표/투자자 컨텍스트(옵션) 병합
6. Walk-forward 검증 및 OOF 생성
7. 신호 가중치 튜닝 + Top-K 백테스트
8. 최종 모델 추론 + 확률 보정(calibration)
9. 정책 컬럼 생성(추천/리스크/사유)
10. 결과 CSV/JSON/그래프/PM 리포트 저장

---

## 2) 파일별 상세 설명 (코드 스니펫 + 함수 역할)

## 2-1. `src/pipeline.py` (오케스트레이션)

### `run_pipeline(...)`
```python
cfg = load_app_config(config_json, overrides=cfg_overrides or None)
raw = load_ohlcv_csv(input_csv)
cleaned = clean_ohlcv(raw)
```
- 전체 실행 제어 함수입니다.
- 설정 → 데이터 → 피처 → 검증/튜닝 → 백테스트 → 최종추론 → 산출물 저장 순서로 호출합니다.

### `resolve_output_path(...)` / `resolve_output_dir(...)`
```python
result_dir = _project_result_dir()
output_path = result_dir / requested.name
```
- 출력물을 `result/` 하위로 강제 정규화합니다.
- 실행환경(로컬/서버/OS) 차이로 경로가 흔들리는 문제를 방지합니다.

### `_feature_columns(df)`
```python
return [c for c in df.columns if c.startswith(("ret_", "ma_", ...)) or c in base]
```
- 학습에 들어갈 입력 피처를 선별합니다.
- 예측 타깃/메타 컬럼이 모델 입력으로 섞이는 사고를 줄입니다.

### `_adaptive_training_cfg(cfg, feat)`
```python
uniq = len(feat["Date"].unique())
tuned.min_train_size = min(tuned.min_train_size, max(60, int(uniq * 0.6)))
```
- 데이터 길이에 따라 워크포워드 파라미터를 동적으로 보정합니다.
- 샘플 수가 작아도 fold 생성 실패를 완화합니다.

### `_split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7)`
```python
split_idx = int(len(dates) * tune_ratio)
tune_df = scored_oof[scored_oof["Date"].isin(tune_dates)]
```
- 시계열 OOF를 튜닝 구간/평가 구간으로 분리합니다.
- 룩어헤드 없이 전략 튜닝 성능을 점검합니다.

### `_calibrate_up_probability(oof_df, up_probs)`
```python
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(cal["up_probability"], y)
if raw_unique >= 4 and calibrated_unique <= 2:
    return (0.3 * calibrated + 0.7 * raw_probs).clip(0.0, 1.0)
```
- OOF 기반 확률 보정입니다.
- 보정 후 확률이 과도하게 뭉개지면(raw 대비 unique 급감) raw와 혼합하여 순위 분해능을 유지합니다.

### `_coverage_gate_status(cfg, external_coverage_ratio, investor_coverage_ratio)`
```python
if external_coverage_ratio < cfg.backtest.min_external_coverage_ratio:
    return "halt"
```
- 데이터 커버리지 부족 시 `halt/caution/normal` 게이트 상태를 계산합니다.
- 품질이 낮은 날의 과도한 액션을 억제합니다.

### `_drop_empty_detail_columns(detail_df)`
```python
if series.notna().sum() == 0:
    drop_cols.append(col)
```
- 전부 비어 있는 optional 컬럼을 결과 상세 CSV에서 제거합니다.
- 전달물 가독성을 개선합니다.

### `_safe_to_csv(df, path)`
```python
try:
    df.to_csv(path, index=False, encoding="utf-8-sig")
except PermissionError:
    fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
```
- 파일 잠금(PermissionError) 시 fallback 파일명으로 재저장합니다.
- 윈도우/엑셀 실무 환경에서 배치 실패를 줄이는 안전장치입니다.

---

## 2-2. `src/pipeline_support.py` (예측 프레임 후처리)

### `PredictionFrameContext`
```python
@dataclass(slots=True)
class PredictionFrameContext:
    external_coverage_ratio: float = 1.0
    investor_coverage_ratio: float = 1.0
```
- 후처리 단계에 전달할 런타임 컨텍스트(커버리지/유동성 기준)를 묶는 객체입니다.

### `build_scored_prediction_frame(...)`
```python
scored = build_prediction_frame(latest_df, pred, signal_cfg)
scored = vectorized_event_signal_boost(scored, cfg=investment_criteria)
```
- 모델 추론 결과를 신호 점수 기반 의사결정 프레임으로 확장합니다.
- 이벤트 기반 가중 보정, 커버리지 값 주입 등을 수행합니다.

### `build_symbol_history_accuracy(scored_oof)`
```python
tmp["history_direction_accuracy"] = (...)
return tmp.groupby("Symbol", as_index=False)["history_direction_accuracy"].mean()
```
- 종목별 과거 방향 적중률을 계산해 신뢰도 보조지표로 활용합니다.

### `finalize_latest_prediction_frame(...)`
```python
out["confidence_score"] = (1 - out["uncertainty_score"]).clip(lower=0, upper=1)
out = build_prediction_policy_frame(out, cfg=investment_criteria)
```
- 심볼명 매핑/신뢰도 라벨링/정책 컬럼 생성까지 수행하는 최종 정리 함수입니다.

---

## 2-3. `src/data/*` (입력 계층)

### `loaders.py::load_ohlcv_csv(...)`
```python
# 필수 컬럼 체크(Date/Open/High/Low/Close/Volume)
# Date 파싱 및 Symbol/Date 정렬
```
- 입력 스키마를 표준화하여 하위 단계 가정(타입/정렬)을 안정화합니다.

### `cleaners.py::clean_ohlcv(...)`
```python
# 결측/중복/비정상 가격 범위 정리
```
- 데이터 품질 1차 방어선 역할입니다.

### `fetch_real_data.py`
#### `normalize_user_symbols(symbol_inputs)`
```python
# 6자리 종목코드 등 사용자 입력을 yfinance 친화 심볼로 정규화
```
- 사용자 입력 다양성을 흡수합니다.

#### `fetch_real_ohlcv(symbols, start, end)`
```python
# 심볼별 시세 수집 후 공통 스키마로 병합
```
- 실데이터 수집의 중심 함수입니다.

#### `append_real_ohlcv_csv(path, symbols, start, end)`
```python
# 기존 CSV + 신규 수집분을 Date/Symbol 기준 병합
```
- 대량 재수집 없이 필요한 종목만 incremental update 가능합니다.

---

## 2-4. `src/features/*` (피처 엔지니어링)

### `price_features.py::build_features(df, cfg)`
```python
rsi = _compute_rsi(close, 14)
macd, macd_signal, macd_hist = _compute_macd(close)
df["target_log_return"] = ...
```
- 기술지표 생성 + 학습 타깃 생성(1d/5d/20d)을 한 번에 수행합니다.
- 피처/타깃 계산 기준이 일관되어 데이터 누수 가능성을 낮춥니다.

### 주요 helper 함수
- `_compute_rsi`, `_compute_macd`, `_compute_atr`, `_compute_stochastic`, `_compute_cci`, `_compute_obv`
- 계산 로직을 분리해 단위 테스트와 재사용성을 높였습니다.

### `external_features.py::add_external_market_features_with_coverage(df, symbols)`
```python
# 외부지표 시계열 병합 + coverage dict 반환
return merged_df, coverage
```
- 외부 데이터 결합과 품질 모니터링(성공/실패/fallback)을 동시에 제공합니다.

### `regime_features.py::annotate_market_regime(df)`
```python
# 변동성/추세 기반 시장 국면 라벨 생성
```
- 시장 상태를 구간화해 전략 해석에 도움을 줍니다.

### `investment_signals.py::add_investment_signal_features(df)`
```python
# 수급/이벤트/모멘텀 파생 신호 생성
```
- 정책 모듈에서 활용할 고수준 투자 신호를 제공합니다.

---

## 2-5. `src/models/lgbm_heads.py` (멀티헤드 모델)

### `MultiHeadStockModel`
```python
# fit(): 회귀/분류/분위수/다중호라이즌 헤드 학습
# predict(): MultiHeadPrediction 반환
```
- 수익률/상승확률/불확실성(분위수)을 동시에 예측합니다.
- “정확도 + 리스크”를 함께 제공하는 것이 핵심입니다.

### `MultiHeadPrediction`
```python
@dataclass
class MultiHeadPrediction:
    predicted_return
    up_probability
    quantile_low/quantile_mid/quantile_high
```
- 멀티헤드 결과의 데이터 계약(contract) 역할을 합니다.

---

## 2-6. `src/validation/*` (검증/전략평가)

### `walk_forward.py::walk_forward_validate_with_oof(...)`
```python
# fold별 검증 + OOF 예측 프레임 반환
```
- OOF는 calibration, signal tuning, 리포트 지표의 공통 입력으로 사용됩니다.

### `walk_forward.py::_iter_folds(...)`
```python
# 시간 순서 train/test 창을 이동하며 fold 생성
```
- 시계열 누수 방지를 보장하는 핵심 함수입니다.

### `signal_tuning.py::tune_signal_weights(...)`
```python
# grid-search로 신호 가중치 탐색
```
- 모델 출력값을 실전용 signal_score로 최적화합니다.

### `backtest.py::run_long_only_topk_backtest(pred_df, cfg)`
```python
# min_up_probability/min_signal_score 필터
# liquidity/capacity/turnover/market_type caps 반영
```
- 실운용 제약을 고려한 Long-only Top-K 성과를 계산합니다.

### `metrics.py::probability_calibration_metrics(...)`
```python
# ECE/Brier 계산
```
- 분류 확률의 calibration 품질을 수치로 평가합니다.

---

## 2-7. `src/domain/*`, `src/reports/*` (정책/리포팅)

### `signal_policy.py::recommendation_from_signal(...)`
```python
# signal + return + prob + uncertainty -> 추천 액션
```
- 모델 출력을 “실행 의사결정”으로 변환합니다.

### `signal_policy.py::vectorized_event_signal_boost(...)`
```python
# 이벤트 신호 기반 점수 보정
```
- 벡터화로 대량 종목 처리 성능을 확보합니다.

### `signal_policy.py::build_prediction_policy_frame(...)`
```python
# recommendation/risk_flag/position_size_hint/prediction_reason 생성
```
- CSV/챗봇/PM 요약에서 공통으로 쓰는 정책 컬럼 표준 생성기입니다.

### `result_formatter.py::build_result_simple(...)`
```python
# 사용자 전달용 핵심 컬럼만 추출
```
- 분석용 상세결과와 사용자용 결과를 분리합니다.

### `pm_report.py::build_pm_report(...)`, `save_pm_report(...)`
```python
# PM 시각의 요약 JSON 작성/저장
```
- 모델 출력을 운용 의사결정 포맷으로 변환합니다.

### `visualize.py` 계열
```python
# equity/drawdown/calibration/symbol-level 그래프 저장
```
- 지표 숫자만으로 보이지 않는 편향/분포를 시각 확인할 수 있습니다.

---

## 3) 문제 해결 과정 (코드 반영 관점)

### 3-1. 외부 데이터 실패 내성
- 문제: 외부지표/실데이터 수집은 실패 가능성이 높음.
- 대응: safe download/fallback + coverage 리포트 제공.
- 효과: 일부 소스 장애에도 파이프라인 연속성 유지.

### 3-2. 확률 보정 부작용 완화
- 문제: isotonic 보정 시 확률 분산 붕괴 가능.
- 대응: `_calibrate_up_probability()`의 raw-calibrated 혼합 가드.
- 효과: calibration 안정성 + 랭킹 분별력 동시 확보.

### 3-3. 출력 운영 안정성
- 문제: 경로 혼선, 파일 잠금으로 저장 실패.
- 대응: `resolve_output_path/dir` + `_safe_to_csv` fallback.
- 효과: 운영 환경별 저장 실패율 감소.

### 3-4. 정책 후처리 표준화
- 문제: 모델 raw 출력은 의사결정 바로 사용이 어려움.
- 대응: `build_prediction_policy_frame()`로 권고/리스크/사유 일원화.
- 효과: 챗봇/CSV/리포트 간 해석 일관성 확보.

---

## 4) 테스트 관점 요약

`tests/`는 단위/통합 관점으로 다음을 검증합니다.
- 파이프라인 스모크(E2E)
- 외부 지표 fallback
- 투자자 컨텍스트 통합
- signal policy 추천/사유
- 확률 calibration 가드
- 챗봇/시각화/유니버스 로더

즉, 성능 지표뿐 아니라 운영 안전장치(coverage/fallback/calibration guard)까지 회귀 검증하도록 구성되어 있습니다.

---

## 5) 결론

본 프로젝트는 단순 예측 스크립트가 아니라,
1) 입력 안정화,
2) 멀티헤드 예측,
3) 워크포워드 검증,
4) 실운용 제약 기반 백테스트,
5) 정책/리포트 자동화,
를 하나로 묶은 운영 지향형 아키텍처입니다.

향후 유지보수 시에는 `run_pipeline()`에 로직을 계속 누적하기보다, 각 레이어 모듈의 경계를 유지하고 테스트로 보호하는 방식이 가장 안전합니다.
