# Stock Predict 프로젝트 기능 전체 요약

## 1. 프로젝트 목적
본 프로젝트는 다음 거래일 기준으로 아래 값을 생성하는 End-to-End 파이프라인을 제공합니다.
- 다음날 퍼센트 수익률(`predicted_return`, %)
- 원시 로그수익률(`predicted_log_return`)
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
13. CSV/JSON/그래프 산출물 저장 + PM 요약 생성

실행 시 콘솔에는 `[1/13]` 형태로 진행률이 출력됩니다.

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
  - `target_log_return_5d`, `target_up_5d`, `target_close_5d`
  - `target_log_return_20d`, `target_up_20d`, `target_close_20d`

### 4.2 외부 시장 피처 (`src/features/external_features.py`)
- KOSPI/KOSDAQ/S&P500/NASDAQ/SOX/VIX/USDKRW/미국10년금리 수집
- 각 외부지표의 종가, 1일/5일 수익률, 20일 변동성 생성
- 네트워크/티커 실패 시 graceful fallback/skip 처리
- 외부 지표 수집 커버리지(요청/성공/실패/fallback) 리포트 제공

### 4.3 시장 국면 (`src/features/regime_features.py`)
- `close_to_ma_20`, `vol_20` 기반 단순 국면 라벨링
- 예: `uptrend_high_vol`, `sideways_low_vol`

---

## 5. 모델링/검증
### 5.1 멀티헤드 모델 (`src/models/lgbm_heads.py`)
- 회귀 헤드: 다음날 로그수익률
- 분류 헤드: 상승 확률
- 분위수 회귀 헤드: 불확실성 밴드
- 추가 호라이즌 헤드: 5일/20일 로그수익률 + 상승확률
- LightGBM 사용 가능 환경에서는 LightGBM 우선 사용, 미설치 시 sklearn GBDT fallback

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
- OOF를 시간순으로 분할(기본 70:30)해 앞 구간에서 가중치 튜닝
- 튜닝된 가중치로 재스코어링 후 뒤 구간(holdout)에서 백테스트

### 6.3 백테스트 (`src/validation/backtest.py`)
- Long-only Top-K
- `min_up_probability`, `min_signal_score` 필터 적용
- 최소 거래대금 필터, 커버리지 halt, 거래대금 참여율(capacity) 체크
- `market_type` 기준 편중 cap, turnover cap, 동적 슬리피지 시나리오
- 누적수익률, Sharpe, MDD + 평균 turnover/평균 선정 종목 수 계산
- benchmark/excess return 및 비용 시나리오 비교

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
- `--config-json`: nested AppConfig 오버라이드
- `--disable-investor-flow`, `--disable-disclosure-context`, `--disable-news-context`
- `--news-scoring-mode`, `--openai-api-key`, `--openai-model`
- `--min-value-traded`, `--turnover-limit`, `--min-up-probability`, `--min-signal-score`

---

## 8. 대표 산출물
- 예측 결과: `--output`
- 실행 리포트: `--report-json` (외부지표 coverage, tuning/backtest 샘플, 백테스트 확장 지표 포함)
- PM 요약: `portfolio_action`, `trading_gate`, `risk_flag`, `confidence_label`
- PM 리포트 아티팩트: `pm_report.json`
- OOF 결과: `reports/oof_predictions.csv`
- 그래프: `--figure-dir/*.png`

---

## 9. 테스트
`tests/test_pipeline_smoke.py` 기준으로 아래를 검증:
- 멀티헤드 예측 shape
- 출력 경로 생성/Windows `/tmp` 매핑
- 파이프라인 E2E 스모크 실행
- 외부 지표 다운로드 실패 시 graceful 동작

---

## 10. 투자 전문가 관점의 "사유(Why)" 표시 가이드
사용자에게 예측 사유를 보여줄 때는 **정확성 + 실행가능성 + 리스크 인지**를 동시에 전달해야 합니다.

### 10.1 한 화면 3단 구조(결론 → 근거 → 행동)
1. **결론(한 줄)**: "오늘 신호는 매수 우위(중립 대비 +1단계)"
2. **근거(숫자 3개 이내)**: 상승확률, 기대수익률, 불확실성 밴드
3. **행동(조건부)**: "시가 -1% 이내 체결 시 유효, 이탈 시 관망"

### 10.2 사유는 "기여도" 중심으로 제시
- "무엇이" 신호를 올렸는지/내렸는지 상위 3개만 노출
- 각 항목을 **방향(+, -) + 강도(약/중/강)** 로 표시
- 예시:
  - `+ 중`: 최근 20일 상대강도 개선
  - `+ 약`: 외국인 수급 순유입
  - `- 중`: 변동성 급등으로 신뢰도 하락

### 10.3 확률/수익률은 구간으로 커뮤니케이션
- 단일 수치보다 범위를 먼저 보여 오해를 줄임
- 예: "기대수익률 +1.8% (신뢰구간 -0.9% ~ +3.2%)"
- 함께 표기:
  - 성공확률(방향 정확도 관점)
  - 손실 꼬리위험(하방 10% 구간)

### 10.4 반대 시나리오를 반드시 병기
- 투자자는 "왜 맞는가"보다 "언제 틀리는가"가 중요
- "무효화 조건" 명시:
  - 거래대금 급감
  - 변동성 임계치 초과
  - 장초반 갭 과대 발생

### 10.5 액션은 주문 가능한 문장으로 변환
- 나쁜 예: "상승 가능성이 있습니다."
- 좋은 예: "비중 20% 이내 분할매수, 손절 -2.5%, 1차 익절 +4%"
- 즉시 실행 가능하도록 **비중/진입/청산/재평가 시점** 4요소를 고정 템플릿으로 제공

### 10.6 신뢰도 라벨 표준화
- `High / Medium / Low`를 임의 문구 대신 정량 기준으로 정의
- 예:
  - High: 과거 유사 국면 hit-rate 상위 30%
  - Medium: 중위 40%
  - Low: 하위 30%
- 라벨 옆에 "최근 3개월 성과"를 같이 표시해 체감 신뢰성 확보

### 10.7 사용자 유형별 표현 차등
- **초보자 모드**: 결론 + 핵심 사유 1~2개 + 주의사항
- **숙련자 모드**: 피처 기여도, 백테스트 유사 국면, 비용 반영 기대값까지 확장
- 동일 모델이라도 설명 깊이를 분리하면 이해도와 만족도가 동시 개선
