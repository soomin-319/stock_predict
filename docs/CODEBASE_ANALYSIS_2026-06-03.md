# 코드베이스 분석 및 수정/개선 제안

작성일: 2026-06-03 KST  
대상 브랜치: `codex/fix-disabled-issue-summary-duplicate-columns`  
기준 커밋: `6894ad3 Ensure prewarm forwards news context credentials`  
검증 환경: `Python 3.14.5` (`C:\Python314\python.exe`)

## 1. 요약

이 저장소는 다음 흐름을 가진 주식 예측 연구/운영 보조 파이프라인이다.

1. OHLCV/시장/수급/외부 컨텍스트 로드
2. 가격·시장·투자자 이벤트 피처 생성
3. LightGBM 기반 다중 헤드 모델 학습
4. walk-forward OOF 검증
5. 장기/단기 예측, long-only top-k 백테스트
6. `result/` 아래 CSV/JSON/figure 생성
7. Kakao/Colab 봇에서 결과와 뉴스·공시 요약 표시

현재 테스트 상태는 양호하다.

```text
pytest -q
180 passed in 28.82s
```

단, 유지보수성과 사용자 출력 품질 측면에서 개선 필요 지점이 뚜렷하다. 특히 한글 문자열 깨짐, 대형 모듈, 신호 정책 경계, 라이브 연동 예외 처리, 백테스트 의사결정 기준 명확화가 우선 과제다.

## 2. 현재 구조

| 영역 | 주요 파일/패키지 | 역할 |
|---|---|---|
| 파이프라인 | `src/pipeline.py`, `src/pipeline_support.py` | 전체 실행 흐름, OOF, 백테스트, 출력 조립 |
| 설정 | `src/config/settings.py`, `configs/*.json` | 데이터클래스 기반 앱 설정 |
| 데이터 | `src/data/*` | OHLCV 로드, yfinance, KRX 이름, DART/Naver 컨텍스트 |
| 피처 | `src/features/*` | 가격/외부시장/수급 이벤트 피처 |
| 모델 | `src/models/lgbm_heads.py` | 회귀/분류/분위수/다중 horizon 헤드 |
| 검증 | `src/validation/*` | walk-forward, metrics, baselines, backtest |
| 추론/정책 | `src/inference/predict.py`, `src/domain/signal_policy.py` | 예측 프레임, 추천/위험/사유 정책 |
| 리포트 | `src/reports/*` | CSV/JSON/figure, 뉴스·공시 display context |
| 챗봇 | `src/chatbot/kakao_colab_bot.py` | Flask/Kakao/Colab/ngrok 통합 |
| 뉴스 영향 | `news_impact/*` | 독립 실행 가능한 뉴스·공시 영향 점수 모듈 |

규모:

| 대상 | 파일 수 | LOC |
|---|---:|---:|
| `src/` | 43 Python files | 8,491 |
| `tests/` | 26 Python files | 4,782 |
| `news_impact/` | 37 Python files | 5,218 |

## 3. 강점

- 테스트 커버리지 폭이 넓다. 파이프라인 smoke, Kakao bot, 외부 피처 fallback, 뉴스 영향, signal policy, walk-forward 등이 포함된다.
- 모든 산출물을 `result/`로 강제하고 CSV를 `utf-8-sig`로 저장하는 Windows/Excel 배려가 있다.
- LightGBM 미설치 시 sklearn fallback을 제공한다.
- 모델 artifact metadata/hash 저장 로직이 있어 재현성 기반이 있다.
- news/disclosure를 기대수익률 계산과 분리하려는 테스트와 설계가 이미 일부 존재한다.
- walk-forward에 `purge_gap_days`가 있어 multi-horizon 누수 방지 의식이 있다.

## 4. 주요 리스크

### R1. 한글 문자열/문서 깨짐

여러 파일에서 한글 리터럴·주석·테스트 기대 컬럼명이 깨져 있다.

예:

- `src/config/settings.py`
- `src/features/price_features.py`
- `src/models/lgbm_heads.py`
- `src/reports/output.py`
- `tests/test_pipeline_smoke.py`
- 기존 `docs/CODEBASE_IMPROVEMENTS_2026-06-02.md`

영향:

- Kakao 응답, CSV 컬럼, 콘솔 경고, 문서 가독성 저하
- 테스트가 깨진 문자열을 기대값으로 고정해 품질 문제를 통과시킬 수 있음
- 향후 한글 컬럼명 변경 시 regression 탐지 어려움

### R2. 대형 파일/대형 클래스

긴 함수/클래스 상위:

| 크기 | 위치 | 항목 |
|---:|---|---|
| 1,506 lines | `src/chatbot/kakao_colab_bot.py:155` | `KakaoColabPredictionBot` |
| 423 lines | `src/pipeline.py:260` | `run_pipeline` |
| 282 lines | `src/features/price_features.py:231` | `build_features` |
| 244 lines | `src/models/lgbm_heads.py:61` | `MultiHeadStockModel` |
| 160 lines | `src/validation/backtest.py:130` | `run_long_only_topk_backtest` |

영향:

- 작은 변경도 사이드이펙트 위험 증가
- 테스트 작성·리뷰 난이도 증가
- 챗봇/파이프라인 장애 원인 추적이 어려움

### R3. 신호 정책 경계가 더 명확해야 함

프로젝트 가드레일:

- 매수/매도/보유 결정은 next-day `predicted_return` 기반이어야 함
- 뉴스/공시는 display-only context이며 기대수익률·추천·자동 신호를 바꾸면 안 됨

현재 `recommendation_from_signal()`은 사실상 `predicted_return` 기준이라 방향은 맞다. 다만 `signal_score`는 `predicted_return`, `up_probability`, `rel_strength`, `uncertainty_score`, `event_boost_score`를 합성하고, 백테스트 top-k는 `signal_score` 정렬을 사용한다. 이 값의 의미가 “의사결정 점수”인지 “보조 랭킹/확신도 점수”인지 문서와 코드명이 혼재되어 있다.

영향:

- “추천 결정은 expected return 기준”이라는 정책과 “top-k 백테스트는 signal_score 기준” 사이 해석 충돌 가능
- 이후 뉴스/공시 피처가 signal_score에 들어가면 가드레일 위반 위험

### R4. 광범위 예외 처리

정적 검색 결과:

```text
broad Exception: 32 hits in 12 files
```

상위 파일:

- `src/chatbot/kakao_colab_bot.py`: 9
- `src/reports/issue_summary.py`: 7
- `src/data/investor_context.py`: 4
- `src/reports/output.py`: 3

영향:

- 라이브 연동 실패 원인 은폐
- 사용자에게 “데이터 없음”과 “API 실패”가 같아 보일 수 있음
- 운영 중 문제 재현/로그 분석 어려움

### R5. 라이브 연동 의존성이 많음

외부 연동:

- yfinance/pykrx
- DART
- Naver News
- OpenAI
- Flask/ngrok

테스트는 대부분 mocking으로 안정화되어 있으나, 실제 운영에서는 timeout/retry/rate-limit/credential error 구분이 중요하다.

### R6. `requirements.txt`와 `pyproject.toml` 중복

두 파일 모두 의존성을 별도 관리한다. 현재는 거의 동일하지만 버전 제약이 완전히 같지 않다.

예:

- `requirements.txt`: `numpy>=2.0,<2.3`
- `pyproject.toml`: `numpy`

영향:

- 설치 경로별 의존성 차이
- CI/로컬/Colab 재현성 차이

## 5. 우선순위별 수정/개선 제안

### P0. 즉시 권장

#### P0-1. 한글 깨짐 복구 및 인코딩 회귀 테스트

작업:

- 깨진 한글 리터럴을 실제 한글로 복구
- CSV 컬럼명 기대값도 정상 한글로 변경
- 테스트가 깨진 문자열을 통과시키지 못하도록 “mojibake 금지” 테스트 추가

권장 테스트:

- `src/`, `tests/`, `docs/` 주요 파일에서 `醫`, `怨`, `沅`, `諛`, `紐`, `�` 등 패턴 검출
- `result_simple.csv`, Kakao 응답 key가 정상 한글인지 확인

#### P0-2. 신호 정책 계약 테스트 강화

작업:

- `predicted_return` 변경 없이 news/disclosure append 후 다음 값이 불변인지 테스트
  - `predicted_return`
  - `predicted_close`
  - `recommendation`
  - buy/sell/hold 관련 컬럼
- news/disclosure 컬럼이 `signal_score` 또는 백테스트 rank에 들어가지 않는지 테스트

권장:

- `tests/test_signal_policy_contract.py` 신설
- “display-only context” 불변성 fixture 재사용

#### P0-3. 백테스트 랭킹 기준 명확화

선택지:

1. 정책 엄격 적용: top-k도 `predicted_return` 기준으로 정렬
2. 연구 점수 유지: `signal_score`를 `research_rank_score`로 rename하고 추천/의사결정과 분리

권장:

- 사용자-facing 추천/매수·매도·보유는 `predicted_return` 전용
- 백테스트 연구 랭킹은 이름을 명확히 분리하고 report에 기준을 명시

### P1. 1~2주 내 권장

#### P1-1. `KakaoColabPredictionBot` 분해

현재 1개 클래스가 라우팅, 세션, 캐시, 백그라운드 job, ngrok, live summary, 포맷팅을 모두 담당한다.

권장 분해:

- `chatbot/routes.py`: Flask route/create_app
- `chatbot/session_state.py`: 사용자 세션
- `chatbot/cache.py`: result/prewarm cache
- `chatbot/jobs.py`: background pipeline job
- `chatbot/formatters.py`: Kakao 응답 포맷
- `chatbot/live_context.py`: DART/Naver/OpenAI live summary

#### P1-2. `run_pipeline()` 오케스트레이션 분해

권장 단계 함수:

- `load_and_prepare_data()`
- `collect_context()`
- `build_feature_matrix()`
- `run_validation()`
- `score_predictions()`
- `run_backtest_and_figures()`
- `write_artifacts()`

효과:

- 단계별 단위 테스트 가능
- 실패 지점 로그 명확화
- Colab/CLI/Kakao 재사용 쉬움

#### P1-3. 외부 연동 에러 모델 표준화

작업:

- `ExternalCallResult` 또는 `CoverageResult` 데이터클래스 도입
- 실패 원인 분류:
  - credential_missing
  - timeout
  - rate_limited
  - empty_response
  - schema_changed
  - network_error
- report JSON에 원인별 count 기록

#### P1-4. 의존성 관리 단일화

권장:

- `pyproject.toml`에 버전 제약을 옮기고 `requirements.txt`는 최소화 또는 generated 파일로 운용
- Colab용 별도 `requirements-colab.txt`가 필요하면 명시

### P2. 중기 개선

#### P2-1. 피처 레지스트리 도입

현재 `FEATURE_COLUMN_BASE`가 큰 set으로 관리된다. 피처가 늘수록 의도/소스/누수 위험 추적이 어렵다.

권장 메타데이터:

- feature name
- source: price / market / investor / display-only
- allowed_for_model: bool
- allowed_for_signal: bool
- allowed_for_display: bool
- leakage_risk: low/medium/high

#### P2-2. 모델 실험/아티팩트 추적 강화

권장:

- report에 model metadata 저장
  - backend
  - feature hash
  - training window
  - horizon list
  - seed
- `result/model/` 또는 별도 artifact 경로에 모델 저장 옵션 추가

#### P2-3. 성능/확장성 개선

후보:

- external feature 다운로드 캐시
- DART/Naver raw event 캐시 TTL
- symbol-level figure 생성 limit/parallel 개선
- `build_features()` vectorization 검토

#### P2-4. 문서 정비

작업:

- 기존 깨진 분석 문서 복구 또는 archive
- `docs/ARCHITECTURE.md`에 decision boundary 명시
- `docs/OPERATIONS.md`에 라이브 연동 실패 유형과 대응 추가

## 6. 추천 실행 순서

1. 한글 깨짐 복구 + mojibake 금지 테스트
2. signal policy contract 테스트 추가
3. 백테스트 rank 기준 명명/정책 정리
4. `KakaoColabPredictionBot` 분해
5. `run_pipeline()` 단계 분해
6. 외부 연동 에러 모델/로그 표준화
7. 의존성 파일 정리

## 7. 검증 체크리스트

수정 후 최소 실행:

```powershell
pytest
pytest tests/test_pipeline_smoke.py
pytest tests/test_kakao_colab_bot.py
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json --figure-dir figures_smoke
```

추가 확인:

- `result/result_simple.csv` 컬럼명이 정상 한글인지 확인
- Kakao 응답 문구가 정상 한글인지 확인
- `pipeline_report.json`에 coverage/failure 원인이 기록되는지 확인
- news/disclosure append 전후 `predicted_return`, `recommendation` 불변 확인

## 8. 결론

코드베이스는 기능과 테스트 기반은 충분히 성숙하다. 현재 최우선 병목은 모델 성능보다 “운영 품질”과 “정책 경계 명확성”이다. 한글 깨짐 복구, display-only contract 강화, 대형 모듈 분해를 먼저 처리하면 이후 모델/피처 개선의 리스크가 크게 줄어든다.
