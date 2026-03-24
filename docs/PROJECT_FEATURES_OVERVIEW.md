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

## 10. 투자 전문가 관점의 "사유(Why)" 표시 가이드 (개정)
사용자에게 사유를 보여줄 때 목표는 **"왜 이 판단인지" + "어떻게 실행할지" + "언제 철회할지"**를 30초 안에 이해시키는 것입니다.

### 10.1 표시 우선순위: 결론보다 "의사결정 가능성"
- 사유 설명의 품질 기준은 문장이 예쁜지보다 **주문/관망 결정을 바로 내릴 수 있는지**입니다.
- 한 화면에서 반드시 아래 4가지를 모두 제공합니다.
  1. 방향: 매수/중립/매도
  2. 기대값: 기대수익률, 상승확률, 하방위험
  3. 조건: 유효 구간(진입), 무효화 구간(철회)
  4. 행동: 비중/손절/익절/재평가 시점

### 10.2 사용자 노출 템플릿(권장)
- **[요약 헤더]** `매수 우위(신뢰도: Medium)`
- **[핵심 사유 3개]** `+강`, `+중`, `-약` 형태로 방향과 강도를 함께 표시
- **[리스크 박스]** 손실 가능 구간, 실패 확률, 무효화 조건
- **[실행 가이드]** 진입/청산/비중/재평가 시점

예시 문구:
- 결론: `오늘은 매수 우위. 단, 시가 대비 +1.5% 이상 갭 상승 시 추격매수 금지`
- 근거: `+강: 20일 추세 상향`, `+중: 거래대금 증가`, `-약: 단기 변동성 확대`
- 행동: `비중 15% 분할진입, 손절 -2.3%, 1차 익절 +3.8%, D+2 재평가`

### 10.3 숫자 표기 규칙(오해 방지)
- 단일 포인트 예측을 전면에 두지 않고 **범위와 확률을 우선**합니다.
- 권장 순서:
  1. 기대수익률 구간 (예: `-0.8% ~ +2.9%`)
  2. 상승확률 (예: `62%`)
  3. 하방 꼬리위험 (예: `하위 10% 손실 -3.4%`)
- 소수점은 과도하게 길게 쓰지 않고(권장 1자리), 사용자 비교가 쉽도록 전 종목 동일 포맷을 유지합니다.

### 10.4 "맞는 이유"보다 "틀리는 조건"을 먼저 명시
- 실전 투자에서 신뢰를 높이는 핵심은 반대 시나리오 공개입니다.
- 최소 3개 무효화 조건을 고정 노출합니다.
  - 변동성 급등(예: 20일 변동성 임계치 초과)
  - 거래대금 붕괴(유동성 기준 미달)
  - 이벤트 갭 발생(시가 괴리 과대)
- 무효화 조건이 충족되면 자동으로 `관망` 또는 `비중축소` 안내를 표시합니다.

### 10.5 사유는 "모델 언어"가 아닌 "투자 언어"로 번역
- 나쁜 표현: `특정 피처 값이 임계치를 상회함`
- 좋은 표현: `최근 수급 개선으로 단기 상방 우위가 강화됨`
- 원칙:
  - 전문용어는 괄호로 보조 설명
  - 한 문장 길이 최소화(핵심+영향)
  - 행동 문장에는 반드시 수치 포함(비중, 가격, 시점)

### 10.6 신뢰도 라벨 운영 기준
- `High / Medium / Low` 라벨은 정량 기준으로만 부여합니다.
- 최소 포함 요소:
  - 유사 국면 히트레이트
  - 최근 3개월 성과
  - 추정 오차(예측구간 폭)
- 라벨 옆에 "왜 이 라벨인지"를 1문장으로 병기해 블랙박스 느낌을 줄입니다.

### 10.7 사용자 수준별 정보 깊이 분리
- **기본 모드(초보자)**
  - 결론 1줄 + 사유 2개 + 주의사항 1개 + 실행문장 1개
- **전문 모드(숙련자)**
  - 피처 기여도, 유사 국면 샘플 수, 비용 반영 기대값, 백테스트 비교
- 같은 모델이라도 설명 깊이를 분리하면 이탈률을 줄이고 실행 일관성을 높일 수 있습니다.

### 10.8 운영 체크리스트(배포 전)
- [ ] 결론/근거/행동/무효화 조건이 한 화면에 동시에 보이는가
- [ ] 사유 3개 이내로 압축되었는가(정보 과밀 방지)
- [ ] 행동 문장에 비중·손절·익절·재평가가 모두 있는가
- [ ] 신뢰도 라벨 근거가 정량 지표와 함께 제시되는가
- [ ] 초보자/숙련자 모드에서 동일 결론이 일관되게 전달되는가

