# 전체 코드베이스 분석 리포트 (2026-06-22)

> 분석 기준: `codex/report-metadata-hardening` 브랜치  
> 범위: `src/`, `tests/`, `configs/`, `data/`, `docs/`, `colab/`, 패키징 설정  
> 규모: `src/` Python 90개 파일 / 약 16,403 LOC, `tests/` Python 50개 파일 / 약 8,766 LOC  
> 목적: 현재 코드베이스의 구조, 데이터 흐름, 핵심 가드레일, 운영 산출물, 테스트 커버리지, 개선 포인트를 한 문서로 정리

---

## 1. 핵심 결론

이 저장소는 한국 주식의 **익일 기대수익률(`predicted_return`)** 을 중심으로 예측·검증·리포팅·챗봇 응답까지 처리하는 연구/운영 지원 파이프라인이다.

가장 중요한 설계 원칙은 다음 두 가지다.

1. **투자자문/자동매매 시스템이 아니다.**  
   산출물은 리서치와 운영 지원 자료다.
2. **매수/매도/관망 판단은 `predicted_return`만 사용한다.**  
   뉴스, 공시, 뉴스 임팩트, 이슈 요약은 사용자에게 보여주는 display-only 컨텍스트이며 기대수익률·순위·권고·시그널을 바꾸면 안 된다.

코드 레벨에서는 `features/feature_selection.py`, `domain/signal_policy.py`, `reports/news_impact_context.py`, 관련 테스트들이 이 원칙을 방어한다.

---

## 2. 저장소 구조

```text
src/
  pipeline.py                    # 메인 CLI/오케스트레이터
  pipeline_support.py            # 최신 예측 프레임 스코어링/마무리
  config/settings.py             # AppConfig dataclass 설정 및 검증
  data/                          # OHLCV 로드/정제, KRX 유니버스, 실데이터 fetch, 투자자 컨텍스트
  features/                      # 가격/기술/외부시장/레짐/투자 시그널 피처
  models/lgbm_heads.py           # 멀티헤드 LightGBM + sklearn fallback
  inference/predict.py           # 모델 예측값을 예측 프레임으로 변환
  validation/                    # walk-forward, OOF, 확률 보정, 백테스트, 시그널 튜닝
  domain/signal_policy.py        # 권고/리스크/PM 요약 정책
  reports/                       # 결과 CSV/JSON, 메타데이터, 매니페스트, PM 리포트, 이슈 요약
  recommendation/                # 실시간 종가배팅 추천 서비스
  chatbot/                       # Kakao/Colab 챗봇
  news_impact/                   # 벤더드 stock-news-impact 패키지
  utils/                         # 원자적 파일 쓰기, 시크릿 레닥션, 결과 정리

tests/                           # 50개 pytest 모듈
configs/                         # 운영/리서치/뉴스임팩트 예시 설정
data/                            # 샘플 OHLCV, KRX 맵, news_impact 예시 CSV
result/                          # 생성 산출물 위치
docs/                            # 현재/보관 문서
colab/                           # Colab 실행 보조
```

패키징 진입점은 `pyproject.toml`에 정의되어 있다.

| 명령 | 엔트리포인트 | 역할 |
|---|---|---|
| `stock-predict` | `src.pipeline:main` | 메인 예측 파이프라인 |
| `stock-predict-kakao` | `src.chatbot.kakao_colab_bot:main` | Kakao/Colab 봇 |
| `stock-news-impact` | `src.news_impact.run:main` | 독립 뉴스 임팩트 분석 |

---

## 3. 메인 파이프라인 흐름

`src/pipeline.py`가 전체 실행을 조율한다. CLI는 `build_cli_parser()`에서 구성되고, 핵심 실행은 `run_pipeline()` 계열 함수에서 진행된다.

```text
OHLCV CSV
  -> load_ohlcv_csv
  -> clean_ohlcv
  -> 유니버스 필터
  -> 선택적 투자자/공시/뉴스 컨텍스트 수집
  -> build_features
  -> 선택적 외부시장 피처 추가
  -> market regime / investment signal 피처
  -> select_feature_columns
  -> walk-forward OOF 검증
  -> 상승확률 보정
  -> signal weight tuning
  -> holdout long-only top-k backtest
  -> 최종 모델 학습
  -> 최신 행 예측
  -> display-only 이슈/뉴스임팩트 컨텍스트 부착
  -> result/ 산출물 작성
```

주요 CLI 옵션:

- `--input`: OHLCV 입력 CSV. 기본값 `data/real_ohlcv.csv`
- `--disable-external`: 외부시장 피처 다운로드 비활성화
- `--fetch-real`, `--real-symbols`, `--real-start`, `--auto-refresh-real`, `--add-symbols`: 실데이터 갱신
- `--fetch-investor-context`: 투자자/공시/뉴스 컨텍스트 활성화
- `--news-impact-report`: 독립 뉴스임팩트 JSON을 표시 컨텍스트로 부착
- `--config-json`: `AppConfig` 오버라이드
- `--report-json`: 파이프라인 요약 JSON 파일명
- `--walk-forward-n-jobs`, `--model-n-jobs`, `--model-head-n-jobs`: 병렬 처리 제어

샘플 스모크 실행:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

---

## 4. 설정 시스템

`src/config/settings.py`는 중첩 dataclass 기반 설정을 제공한다.

주요 설정 묶음:

- `UniverseConfig`: 기본 유니버스, 기대 종목 수
- `FeatureConfig`: 수익률/이동평균/변동성/RSI 등 피처 윈도
- `ExternalFeatureConfig`: KOSPI/KOSDAQ/미국지수/VIX/환율/금리 등 외부 심볼
- `TrainingConfig`: walk-forward train/test/step, purge/embargo, quantile, random seed
- `SignalConfig`: 기대수익률/상승확률/불확실성 페널티 가중치
- `InvestmentCriteriaConfig`: 거래대금, RSI, 순매수, 시장풍향 기준
- `BacktestConfig`: top-k, 포트폴리오 금액, 참여율, 수수료/슬리피지, 커버리지 게이트

프리셋:

- `configs/research_balanced.json`
- `configs/prod_conservative.json`
- `configs/news_impact.example.json`
- `configs/news_impact.gemma.example.json`

미지의 설정 키는 검증 단계에서 에러가 나며, 유사 키 추천까지 제공하는 방식으로 운영 실수를 줄인다.

---

## 5. 데이터 계층

`src/data/`는 입력 데이터와 외부 데이터를 표준화한다.

| 파일 | 역할 |
|---|---|
| `loaders.py` | OHLCV CSV 로드 |
| `cleaners.py` | 필수 컬럼 정제, 0거래량/극단 수익률 플래그 |
| `fetch_real_data.py` | yfinance 기반 실데이터 fetch/append/save |
| `cli_refresh.py` | CLI fetch 대상과 증분 시작일 해석 |
| `krx_universe.py` | KRX 심볼-종목명 매핑, 이름 검색 |
| `universe.py` | 유니버스 CSV/기본 유니버스 로드 및 필터 |
| `investor_context.py` | 투자자 수급, DART 공시, Naver 뉴스 컨텍스트 수집 |

데이터 관련 운영 원칙:

- 샘플/유니버스 입력은 `data/`에 둔다.
- 생성 CSV/JSON은 `result/` 아래에 둔다.
- CSV 산출물은 Excel/Windows 호환을 위해 `utf-8-sig`를 사용한다.
- 외부 API 키는 환경변수 또는 로컬 인자로만 전달한다.

---

## 6. 피처 계층

`src/features/`는 모델 입력 후보와 표시 전용 컨텍스트를 분리한다.

| 파일 | 역할 |
|---|---|
| `price_features.py` | 수익률, 이동평균, 변동성, 타깃, 품질 플래그 |
| `technical_indicators.py` | RSI, MACD, ATR, stochastic, CCI, OBV, z-score |
| `external_features.py` | 외부시장 다운로드 및 lag 적용 |
| `regime_features.py` | 시장 레짐 주석 |
| `investment_signals.py` | 거래대금/주도주/수급/기술 신호 |
| `feature_selection.py` | 모델 입력 컬럼 선택, display-only 컬럼 제거 |

가장 중요한 경계는 `feature_selection.py`다.

- 뉴스/공시/뉴스임팩트 관련 컬럼은 `DISPLAY_ONLY_CONTEXT_COLUMNS`로 분리된다.
- `news_impact_` 접두사 컬럼은 모델 피처에서 자동 제외된다.
- `select_feature_columns()`가 모델 입력을 확정한다.
- `tests/test_display_only_feature_guard.py`가 이 동작을 회귀 방지한다.

---

## 7. 모델 계층

`src/models/lgbm_heads.py`의 `MultiHeadStockModel`이 핵심 모델이다.

지원 헤드:

- 익일 로그수익률 회귀
- 상승/하락 방향 분류
- 분위수 회귀(기본 0.1, 0.5, 0.9)

특징:

- LightGBM 우선 사용
- LightGBM이 없으면 scikit-learn GBDT fallback
- 피처 결측은 학습 데이터 기반 중앙값/중립값으로 임퓨트
- 분위수 예측은 정렬해 crossing을 방지
- 모델 메타데이터에 backend, feature hash, random seed, artifact version을 남긴다

`src/inference/predict.py`는 모델 예측을 `predicted_return`, `up_probability`, quantile, uncertainty, signal label 등으로 변환한다.

---

## 8. 검증과 백테스트

`src/validation/`은 누설 방지와 평가 신뢰도를 담당한다.

| 파일 | 역할 |
|---|---|
| `walk_forward.py` | walk-forward fold 생성/실행, OOF 집계 |
| `support.py` | OOF tune/eval 분할, 보정, 진단 |
| `metrics.py` | 회귀/분류/확률보정 메트릭 |
| `baselines.py` | baseline 평가 |
| `backtest.py` | long-only top-k 백테스트, 거래비용/유동성 제약 |
| `signal_tuning.py` | OOF 기반 시그널 가중치 튜닝 |
| `result_validity.py` | 백테스트 결과 유효성 판정 |

중요 설계:

- `purge_gap_days`, `embargo_days`로 익일 타깃 누설을 줄인다.
- OOF를 tune/eval로 나누어 시그널 튜닝과 평가를 분리한다.
- holdout split이 부족하면 백테스트를 무리하게 수행하지 않는다.
- 커버리지 게이트, 유동성 필터, 거래비용, 포지션 수 제한을 반영한다.

---

## 9. 시그널 정책

`src/domain/signal_policy.py`는 권고, 신뢰도, 리스크, PM 요약을 만든다.

핵심 계약:

```text
predicted_return >  2.0  -> 매수
predicted_return <= -2.0 -> 매도
그 외                     -> 관망
```

`up_probability`, 뉴스, 공시, 이슈 요약, 뉴스임팩트는 권고를 바꾸면 안 된다.

부가 산출:

- `confidence_label`
- `risk_flag`
- `position_size_hint`
- `portfolio_action`
- `trading_gate`
- `prediction_reason`
- `jongbae_score`

`tests/test_signal_policy_contract.py`가 기대수익률 단독 권고 계약을 검증한다.

---

## 10. 리포트와 산출물

`src/reports/`는 사용자/운영 산출물을 만든다.

| 파일 | 역할 |
|---|---|
| `output.py` | CSV 경로 강제, simple/detail 결과 생성 |
| `result_formatter.py` | 사용자 표시용 포맷, 콘솔 요약 |
| `run_artifacts.py` | `result/runs/<run_id>/`, `result/latest/`, manifest 관리 |
| `report_metadata.py` | run_id, git commit, config hash, KRX 영업일, 캘린더 커버리지 |
| `pm_report.py` | PM 스타일 JSON 리포트 |
| `issue_summary.py` | 종목별 이슈 요약, LLM/룰 기반 fallback |
| `news_impact_context.py` | 독립/생성 뉴스임팩트 결과를 display-only 컬럼으로 부착 |
| `context_policy.py` | 컨텍스트 날짜 허용 정책 |

산출물 라이프사이클:

```text
result/runs/<run_id>/      # 실행별 원본
result/latest/             # 운영 최신본
result/result_simple.csv   # 챗봇 호환 파일
result/result_detail.csv   # 상세 결과
```

`RunArtifactManager`는 다음을 보장한다.

- 경로 탈출 방지
- 원자적 쓰기
- CSV `utf-8-sig`
- manifest에 sha256, 행수, 컬럼, schema version 기록
- sample/smoke 실행은 production latest를 덮어쓰지 않음

---

## 11. 뉴스 임팩트 패키지

`src/news_impact/`는 독립 실행 가능한 벤더드 패키지다.

주요 영역:

- Naver/DART 수집
- 기사 본문 fetch 및 중복 제거
- 이벤트 스키마와 taxonomy
- LLM 기반 영향 판정
- LLM 응답 캐시
- 안전 필터/프롬프트 안전
- semantic clustering
- 점수 집계와 랭킹
- backtest/validation
- JSON/CSV 리포트와 audit metadata

두 사용 경로:

```text
독립 실행:
  stock-news-impact -> report.json/report.csv/audit.json

메인 파이프라인 통합:
  run_pipeline --news-impact-report ... -> result_detail.csv의 news_impact_* 표시 컬럼
```

메인 파이프라인에 붙을 때도 `news_impact_*` 컬럼은 display-only이며 모델 입력과 권고 정책에서 제외된다.

---

## 12. 챗봇과 실시간 추천

`src/chatbot/kakao_colab_bot.py`는 가장 큰 단일 모듈이다. Flask 기반 Kakao 웹훅 서버이며 Colab/ngrok 운영을 염두에 둔다.

주요 기능:

- 사용자 발화 intent 판별
- 상태/도움말/최신화/종목 조회/추천 응답
- `result/result_simple.csv` 캐시 읽기
- 누락 종목에 대해 백그라운드 파이프라인 잡 실행
- prewarm prediction cache
- HMAC 서명, 허용 IP/CIDR, 시크릿 레닥션
- ngrok 터널 실행 보조

`src/recommendation/`은 챗봇의 추천 요청에서 쓰이는 실시간 종가배팅 후보를 만든다.

주의:

- 챗봇 응답에 뉴스/공시 요약이 포함되어도 이는 표시 전용이다.
- 기대수익률 기반 권고 계약을 깨면 안 된다.

---

## 13. 테스트 현황

`tests/`에는 50개 Python 테스트 모듈이 있다.

주요 검증 영역:

- 파이프라인 smoke와 CLI 옵션
- 데이터 정제/fetch fallback
- 피처 계층 hardening
- display-only 컨텍스트 가드
- 모델 persistence와 fallback
- walk-forward, OOF, 확률 보정, 백테스트
- 시그널 정책 계약/추천/이벤트 부스트/근거 문구
- 리포트 메타데이터, 산출물 manifest, result cleanup
- 뉴스임팩트 full package, LLM cache, prompt safety
- Kakao/Colab bot helper
- 시크릿 레닥션

최소 권장 검증:

```powershell
pytest tests/test_pipeline_smoke.py
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

전체 제출 전 권장:

```powershell
pytest
```

---

## 14. 강점

- **핵심 정책 경계가 명확함**: 권고는 `predicted_return`만 사용한다.
- **display-only 방어가 좋음**: 뉴스/공시/뉴스임팩트가 모델 입력으로 새지 않도록 테스트가 존재한다.
- **운영 산출물 관리가 견고함**: run_id, manifest, hash, latest 승격 조건, 원자적 쓰기.
- **검증 체계가 단순 train/test보다 강함**: walk-forward OOF, purge/embargo, tune/eval 분리, holdout 백테스트.
- **외부 통합 실패를 경고/감쇠 처리**: 외부시장/투자자/이슈요약 실패가 전체 파이프라인 중단으로 바로 이어지지 않도록 설계되어 있다.
- **뉴스임팩트가 독립 실행과 통합 표시를 모두 지원**한다.

---

## 15. 주의점과 개선 후보

### 15.1 대형 모듈 분해

다음 파일은 책임이 많고 수정 충돌 가능성이 높다.

- `src/pipeline.py`
- `src/chatbot/kakao_colab_bot.py`
- `src/reports/issue_summary.py`

후속 개선 시에는 CLI/오케스트레이션, artifact writing, validation orchestration, chatbot runtime, chatbot response formatting을 더 작게 나누는 것이 좋다.

### 15.2 row 함수와 vectorized 함수 병존

`signal_policy.py`에는 하위호환 row 단위 함수와 대량 처리용 series 함수가 함께 있다. 성능 측면에서는 vectorized 경로가 좋지만, 계약 테스트가 양쪽 동작 차이를 계속 감시해야 한다.

### 15.3 외부 API 의존성

yfinance, DART, Naver, OpenAI/llama.cpp, ngrok 등이 존재한다. 테스트에서는 mock/disable이 기본이어야 하고, 운영에서는 키/토큰을 절대 커밋하지 않아야 한다.

### 15.4 문서 인코딩 점검

일부 문서/출력에서 한글이 깨져 보이는 구간이 관찰된다. Windows/PowerShell/UTF-8/BOM 조합을 고려해 문서 저장 인코딩과 터미널 출력 인코딩을 점검할 필요가 있다.

### 15.5 `result/` 관리

`result/`는 생성물 영역이다. 커밋 대상은 의도된 작은 샘플/계약 산출물에 한정하고, 대형·오래된 산출물은 정리 대상이다.

---

## 16. 변경 시 체크리스트

코드 변경 전후로 다음 계약을 확인한다.

- [ ] `predicted_return` 외 값이 매수/매도/관망을 바꾸지 않는가?
- [ ] 뉴스/공시/뉴스임팩트 컬럼이 모델 피처에 들어가지 않는가?
- [ ] 새 CSV 산출물이 `result/` 아래에 `utf-8-sig`로 저장되는가?
- [ ] 외부 API/네트워크는 테스트에서 mock 또는 disable 되는가?
- [ ] 경로 입력이 `result/` 밖으로 탈출하지 않는가?
- [ ] 샘플 smoke 실행이 production latest를 덮어쓰지 않는가?
- [ ] 설정 키 검증과 기본값이 문서화되어 있는가?
- [ ] 관련 pytest가 추가/갱신되었는가?

---

## 17. 빠른 참조

```powershell
# 설치
python -m pip install -r requirements.txt
python -m pip install -e .

# 샘플 실행
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json

# 전체 테스트
pytest

# 파이프라인 핵심 테스트
pytest tests/test_pipeline_smoke.py

# Kakao bot entrypoint
stock-predict-kakao

# 독립 뉴스임팩트 CLI
stock-news-impact --help
```

---

## 18. 최종 평가

현재 코드베이스는 단순 예측 스크립트가 아니라 **데이터 수집, 피처 생성, walk-forward 검증, 기대수익률 기반 정책, 산출물 manifest, 챗봇 운영, 뉴스임팩트 표시 컨텍스트**까지 포함한 통합 연구/운영 시스템이다.

가장 중요한 품질 축은 예측 성능보다도 **정책 무결성**이다. 즉, 뉴스나 공시가 사용자 설명에는 등장하더라도 `predicted_return` 기반 권고를 바꾸지 않는다는 점이 프로젝트의 핵심 계약이다. 앞으로의 기능 추가도 이 경계를 먼저 확인한 뒤 진행해야 한다.
