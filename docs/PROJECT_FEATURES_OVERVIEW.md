# Stock Predict 프로젝트 기능 전체 요약

## 1. 프로젝트 목적
본 프로젝트는 다음 거래일 기준으로 아래 값을 생성하는 End-to-End 파이프라인을 제공합니다.
- 다음날 로그수익률(`predicted_return`)
- 상승 확률(`up_probability`)
- 분위수 기반 불확실성 구간(`uncertainty_band`)
- 최종 신호 점수/라벨(`signal_score`, `signal_label`)

---

## 2. 전체 파이프라인 흐름
`src/pipeline.py`의 `run_pipeline()` 기준 처리 순서:
1. 앱 설정 로드
2. 입력 OHLCV CSV 로드
3. 데이터 정제 및 유니버스 필터 적용
4. 가격/기술적 피처 생성
5. 외부 시장지표 피처 병합(옵션)
6. 시장 국면 레이블 생성
7. Walk-forward 검증
8. Baseline 성능 계산
9. OOF 예측 생성
10. 신호 가중치 튜닝
11. Top-K 백테스트 및 그래프 생성
12. 최종 모델 학습 후 최신 시점 추론
13. CSV/JSON/그래프 산출물 저장

실행 시 콘솔에는 `[1/12]` 형태로 진행률이 출력됩니다.

---

## 3. 데이터 계층 기능
### 3.1 로딩 (`src/data/loaders.py`)
- 필수 컬럼 검증: `Date/Open/High/Low/Close/Volume`
- 날짜 파싱 및 `Symbol` 기본값 보정
- `Symbol, Date` 정렬

### 3.2 정제 (`src/data/cleaners.py`)
- 결측 제거, 숫자형 캐스팅
- 가격/거래량 유효성 검사
- High/Low 봉 일관성 검사
- 중복(`Date, Symbol`) 제거

### 3.3 유니버스 (`src/data/universe.py`)
- 기본 유니버스 컨셉: `KOSPI200 + KOSDAQ150`
- CSV 기반 심볼 유니버스 로딩/필터링 지원

### 3.4 실데이터 수집 (`src/data/fetch_real_data.py`)
- `yfinance` 기반 다중 심볼 OHLCV 수집
- 표준 컬럼 스키마로 통합 후 CSV 저장

---

## 4. 피처 엔지니어링
### 4.1 가격/기술적 피처 (`src/features/price_features.py`)
- 수익률/갭/일중수익률/변동성/이동평균 계열
- RSI(14), MACD/Signal/Histogram
- ATR(14), Stochastic(K/D), CCI(20), OBV
- 학습 타깃 생성:
  - `target_log_return`
  - `target_up`
  - `target_close`

### 4.2 외부 시장 피처 (`src/features/external_features.py`)
- KOSPI/KOSDAQ/S&P500/NASDAQ/SOX/VIX/USDKRW/미국10년금리 수집
- 각 외부지표의 종가, 1일/5일 수익률, 20일 변동성 생성
- 네트워크/티커 실패 시 graceful fallback/skip 처리

### 4.3 시장 국면 (`src/features/regime_features.py`)
- `close_to_ma_20`, `vol_20` 기반 단순 국면 라벨링
- 예: `uptrend_high_vol`, `sideways_low_vol`

---

## 5. 모델링/검증
### 5.1 멀티헤드 모델 (`src/models/lgbm_heads.py`)
- 회귀 헤드: 다음날 로그수익률
- 분류 헤드: 상승 확률
- 분위수 회귀 헤드: 불확실성 밴드
- 현재 구현은 sklearn GBDT 기반

### 5.2 검증 (`src/validation/walk_forward.py`)
- 시간순 Walk-forward fold 생성
- Fold별 회귀/분류 지표 집계
- OOF 예측 저장용 프레임 생성

### 5.3 지표/기준선
- 회귀: MAE, RMSE, Corr (`metrics.py`)
- 분류: Accuracy, ROC-AUC (`metrics.py`)
- Baseline: zero return / previous return (`baselines.py`)

---

## 6. 신호/백테스트/리포트
### 6.1 신호 생성 (`src/inference/predict.py`)
- `predicted_return`, `predicted_close`, `up_probability`, `uncertainty_band`
- 정규화된 return/relative strength/uncertainty 기반 `signal_score`
- 구간 라벨: strong_negative ~ strong_positive

### 6.2 신호 튜닝 (`src/validation/signal_tuning.py`)
- Grid-search로 top-decile 평균 수익률 최대화
- 튜닝된 가중치로 재스코어링

### 6.3 백테스트 (`src/validation/backtest.py`)
- Long-only Top-K
- 수수료/슬리피지 차감
- 누적수익률, Sharpe, MDD 계산

### 6.4 시각화 (`src/reports/visualize.py`)
- `equity_curve.png`
- `drawdown_curve.png`
- `signal_score_hist.png`

---

## 7. CLI 기능 (`src/pipeline.py`)
주요 옵션:
- `--input`: 입력 CSV 경로
- `--output`: 최종 예측 CSV 경로
- `--report-json`: 실행 리포트 JSON
- `--figure-dir`: 그래프 저장 경로
- `--fetch-real`: 실데이터 자동 수집
- `--real-symbols`: 수집 심볼 목록
- `--real-start`: 수집 시작일
- `--disable-external`: 외부 피처 병합 비활성화
- `--universe-csv`: 유니버스 CSV

---

## 8. 대표 산출물
- 예측 결과: `--output`
- 실행 리포트: `--report-json`
- OOF 결과: `reports/oof_predictions.csv`
- 그래프: `--figure-dir/*.png`

---

## 9. 테스트
`tests/test_pipeline_smoke.py` 기준으로 아래를 검증:
- 멀티헤드 예측 shape
- 출력 경로 생성/Windows `/tmp` 매핑
- 파이프라인 E2E 스모크 실행
- 외부 지표 다운로드 실패 시 graceful 동작
