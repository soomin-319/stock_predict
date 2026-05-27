# 전문가 관점 코드베이스 분석 및 개선/수정 제안

분석일: 2026-05-27  
대상: `C:\Users\카운\Desktop\stock_predict`

## 1. 요약

이 저장소는 단순 예측 스크립트가 아니라 **데이터 수집 → 피처 생성 → walk-forward 검증 → 모델 학습 → 백테스트 → 리포트/챗봇 제공**까지 포함한 엔드투엔드 주식 예측 플랫폼이다. 구조 자체는 `src/data`, `features`, `models`, `validation`, `reports`, `domain`, `chatbot`으로 잘 나뉘어 있으나, 현재는 다음 문제가 유지보수성과 운영 안정성을 가장 크게 제한한다.

1. 실행 환경 재현성이 낮다. `pyproject.toml`, `requirements.txt`, 실제 로컬 환경이 불일치한다.
2. 외부 의존성이 import 시점에 강하게 결합되어 테스트 수집 단계부터 실패할 수 있다.
3. `src/pipeline.py`와 `src/chatbot/kakao_colab_bot.py`가 너무 많은 책임을 가진다.
4. 일부 한글 문자열/주석이 인코딩 깨짐 상태로 남아 있어 사용자 출력과 정책 로직에 위험하다.
5. 외부 API, 캐시, 병렬 실행, 산출물 스키마에 대한 운영 방어선이 더 필요하다.

## 2. 현재 상태 관찰

### 규모

- Python 파일: 67개
- Python 코드 라인: 약 12,517줄
- 주요 대형 파일
  - `src/chatbot/kakao_colab_bot.py`: 1,860줄, 함수/메서드 93개
  - `tests/test_kakao_colab_bot.py`: 1,700줄, 테스트 95개
  - `src/pipeline.py`: 841줄
  - `src/reports/issue_summary.py`: 655줄
  - `src/features/price_features.py`: 513줄

### 강점

- 도메인별 디렉터리 분리는 양호하다.
- walk-forward, purge gap, OOF 예측, 보정, 백테스트 등 ML 검증 구조가 이미 있다.
- `result_simple.csv`, `result_detail.csv`, `pm_report.json` 등 산출물 계약이 명확하다.
- Kakao/Colab 운영 흐름과 실시간 추천 기능까지 포함되어 실사용 시나리오가 분명하다.
- 테스트가 꽤 풍부하다. 특히 챗봇 캐시/요약/실시간 요청 관련 회귀 테스트가 많다.

## 3. 테스트/환경 진단

로컬에서 `python -m pytest -q` 실행 시 수집 단계에서 실패했다.

주요 원인:

- `sklearn`, `yfinance`, `openai`, `joblib` 미설치
- 현재 로컬 Python: `3.14.5`
- 설치된 주요 패키지: `numpy 2.4.6`, `pandas 2.3.3`, `matplotlib 3.10.9`, `pykrx 1.2.8`, `pytest 9.0.3`
- `requirements.txt`는 `numpy>=2.0,<2.3`를 요구하지만 실제 환경은 `numpy 2.4.6`
- `requirements.txt`에는 `pykrx`가 없지만 `pyproject.toml`에는 있다.
- pytest cache가 `result/.pytest_cache` 아래에 쓰이도록 되어 있으나, 현재 환경에서 permission warning이 발생했다.

### 즉시 수정 권장

1. 지원 Python 버전을 현실적으로 제한한다.
   - 예: `requires-python = ">=3.10,<3.14"`
   - scikit-learn/lightgbm/yfinance 호환 확인 후 3.14 지원은 별도 작업으로 승격.
2. 의존성 원천을 하나로 정한다.
   - `pyproject.toml` 기준으로 관리하고 `requirements.txt`는 생성물로 취급하거나,
   - 반대로 `requirements*.txt`를 기준으로 하고 `pyproject.toml`을 동기화.
3. `requirements.txt`에 `pykrx` 추가 또는 `pyproject.toml`에서 optional extra로 분리.
4. `dev` extra에 최소 테스트 실행 의존성을 포함한다.
   - 현재 `dev = ["pytest"]`만으로는 테스트 수집도 불가.

권장 예시:

```toml
[project.optional-dependencies]
dev = [
  "pytest",
  "scikit-learn",
  "joblib",
  "yfinance",
  "openai",
  "pykrx",
]
bot = ["flask", "pyngrok"]
llm = ["openai"]
market = ["yfinance", "pykrx"]
```

## 4. 최우선 개선 과제

### P0-1. import-time 외부 의존성 제거

현재 여러 모듈이 import만 해도 외부 패키지를 요구한다.

예:

- `src/data/fetch_real_data.py`: `import yfinance`
- `src/data/investor_context.py`: `import yfinance`
- `src/features/external_features.py`: `import yfinance`
- `src/reports/issue_summary.py`: `from openai import OpenAI`
- `src/models/lgbm_heads.py`: `joblib`, `sklearn`
- `src/recommendation/__init__.py`: 실시간 추천 서비스를 즉시 import하여 `yfinance` 경로까지 끌어옴

개선안:

- 외부 통합은 함수 내부 lazy import로 이동.
- optional dependency가 없을 때 명확한 메시지와 fallback 제공.
- `src/recommendation/__init__.py`에서 `RealTimeCloseBettingRecommendationService` 자동 import 제거 또는 lazy accessor 사용.
- `--disable-external` 경로는 yfinance/openai/pykrx 없이도 import와 기본 테스트가 가능해야 한다.

완료 기준:

- `python -c "import src.pipeline"`가 외부 API 패키지 없이 성공.
- `pytest tests/test_signal_policy_recommendation.py tests/test_universe_loader.py`가 market/llm extra 없이 성공.

### P0-2. 인코딩 깨짐 수정

확인된 위험 예:

- `src/chatbot/kakao_colab_bot.py`
  - `row.get("?덉륫 ?댁쑀")`
- `src/recommendation/realtime_close_betting.py`
  - `"?? ?? ?? ?? ??"`
- `src/pipeline.py`, `src/config/settings.py` 일부 주석/출력 문자열에 깨진 한글이 보임

위험:

- 사용자 응답 품질 저하
- 컬럼명 참조 실패
- 테스트 fixture와 실제 산출물의 암묵적 불일치

개선안:

1. 깨진 문자열을 원래 의미로 복원.
2. 핵심 한글 컬럼명은 상수화.
3. UTF-8 회귀 테스트 추가.
4. Windows/Colab 실행에서 `PYTHONUTF8=1` 또는 명시적 `encoding="utf-8"` 일관 적용.

### P0-3. 파이프라인 책임 분리

`src/pipeline.py`는 현재 CLI, 설정 override, 데이터 적재, 외부 피처, 검증, 백테스트, 최종 학습, 리포트 저장까지 담당한다.

개선 방향:

- `src/pipeline.py`: CLI와 orchestration만 유지.
- `src/pipeline_steps/` 또는 기존 모듈로 단계 이동.
  - `data_stage.py`: load/clean/universe/real refresh
  - `feature_stage.py`: price/external/investor feature
  - `validation_stage.py`: walk-forward/OOF/diagnostics
  - `training_stage.py`: final model/prediction
  - `artifact_stage.py`: CSV/JSON/figure 저장
- 현재 compatibility wrapper는 유지하되 deprecated 표시 후 점진 제거.

완료 기준:

- `run_pipeline()` 본문 250줄 이하.
- 각 단계가 입력/출력 dataclass를 사용.
- 산출물 저장은 `src/reports/output.py` 단일 경로로 통합.

### P0-4. Kakao 봇 서비스 분리

`src/chatbot/kakao_colab_bot.py`는 1,860줄로, 웹훅 처리, 세션, 캐시, 프로세스 실행, 부트스트랩, 실시간 뉴스 요약, ngrok, 포매팅을 모두 포함한다.

권장 분리:

- `bot_app.py`: Flask endpoint
- `intent_parser.py`: 사용자 발화 해석
- `session_store.py`: 사용자 세션
- `prediction_cache.py`: result CSV/cache 읽기·쓰기
- `job_runner.py`: subprocess 실행/상태 관리
- `issue_summary_service.py`: 뉴스/공시 요약
- `message_formatter.py`: Kakao 응답 포맷
- `tunnel.py`: pyngrok/Colab tunnel

완료 기준:

- 각 모듈 300줄 이하.
- 파일 lock/atomic write 도입.
- 캐시 TTL, prediction date, runtime signature 검증을 단일 서비스에서 수행.

## 5. 모델/검증 개선

### 현재 장점

- walk-forward OOF 예측을 사용한다.
- `purge_gap_days`가 있어 다중 horizon target 누수 방지 의도가 있다.
- signal score와 정책 recommendation이 분리되어 있다.
- 백테스트에 거래대금, turnover, participation, 수수료/슬리피지 개념이 있다.

### 추가 권장

1. 데이터 누수 점검 리포트 자동화
   - feature 생성 시 미래 `Close`, `Volume`, raw event가 섞이지 않는지 검사.
   - horizon별 target 생성 후 validation 시작 전 purge가 실제 적용됐는지 리포트.
2. 모델/정책 버전 명시
   - `pm_report.json`, `pipeline_report.json`, CSV에 `model_version`, `feature_schema_hash`, `policy_version` 추가.
3. baseline 고정
   - 단순 momentum, market-neutral, equal-weight benchmark를 항상 저장.
4. calibration gate 강화
   - Brier score, ECE, calibration curve summary를 report에 포함.
   - `up_probability`가 과신 상태면 추천 강도 제한.
5. 백테스트 현실성 강화
   - 다음날 시가/종가 체결 가정 명시.
   - 거래정지/관리종목/상폐 survivorship bias 문서화.
   - KOSPI/KOSDAQ 별 benchmark 분리.

## 6. 외부 데이터/운영 안정성

외부 통합:

- yfinance
- pykrx
- DART
- Naver News
- OpenAI
- Flask/ngrok

권장:

- provider별 timeout/retry/backoff 표준화.
- API별 에러 타입 정의.
- 원천 데이터 cache metadata 저장: `source`, `fetched_at`, `reference_date`, `ttl`, `status`.
- DART/Naver 결과는 예측 피처와 표시용 컨텍스트를 명확히 분리.
- OpenAI 호출은 비용/timeout/rate limit guard 추가.
- live fetch 실패 시 `stale-but-usable` 응답과 `freshness warning` 분리.

## 7. 병렬 실행/파일 안전성

현재 병렬 관련 코드가 여러 곳에 있다.

- walk-forward 병렬 실행
- model head 병렬 실행
- yfinance/download 병렬 실행
- investor context 병렬 실행
- chatbot background thread/subprocess

위험:

- nested parallelism 과다로 Colab/로컬 CPU 과점유
- `result/*.csv`, cache registry 동시 쓰기 충돌
- background summary와 bootstrap job 경합

개선안:

- 전역 concurrency budget 도입.
- `model_n_jobs`, `model_head_n_jobs`, `walk_forward_n_jobs` 상호 제약 검증.
- CSV/JSON 쓰기는 temp file 후 atomic replace.
- cache registry는 file lock 사용.
- job state는 dataclass + enum으로 명확화.

## 8. 코드 품질 개선 포인트

### broad exception 축소

`except Exception` 사용이 다수 있다.

핫스팟:

- `src/chatbot/kakao_colab_bot.py`: 23건
- `src/reports/issue_summary.py`: 8건
- `src/data/investor_context.py`: 4건
- `src/reports/output.py`: 3건

개선안:

- 네트워크/파싱/파일/LLM 예외를 분리.
- 사용자에게 보여줄 메시지와 내부 로그를 분리.
- 최소한 `logger.exception(...)` 또는 structured error payload 저장.

### 스키마 상수화

반복되는 CSV 컬럼명, 한글 표시명, issue summary 컬럼은 상수 모듈로 분리한다.

권장 파일:

- `src/reports/schema.py`
- `src/domain/policy_schema.py`

완료 기준:

- `result_simple.csv`와 `result_detail.csv` schema test가 컬럼 순서/필수 여부를 검증.

### 설정 검증

현재 dataclass 설정 병합은 unknown key를 조용히 무시한다. 운영에서는 오타가 그대로 지나갈 수 있다.

개선안:

- strict mode 추가.
- unknown key 발견 시 warning 또는 error.
- numeric range validation 추가.
  - 예: `0 <= turnover_limit <= 1`
  - `top_k > 0`
  - `min_up_probability`는 `[0, 1]`

## 9. 보안/비밀정보

좋은 점:

- README/AGENTS에서 API key를 커밋하지 말라고 명시.
- OpenAI/DART/Naver/ngrok token을 환경변수/인자로 받을 수 있다.

추가 권장:

- CLI command/log에 secret이 출력되지 않게 masking.
- Kakao webhook 요청 body 로그 최소화.
- ngrok public URL 출력은 필요하지만 token/headers는 금지.
- `.env.example`만 제공하고 실제 `.env`는 ignore.
- OpenAI 입력에 포함되는 뉴스/공시 텍스트 길이 제한과 개인정보 필터 추가.

## 10. 권장 수정 로드맵

### 1일 내

1. `pyproject.toml`/`requirements.txt` 불일치 수정.
2. Python 지원 범위 명시.
3. import-time optional dependency 제거 1차.
4. 깨진 한글 문자열 복원.
5. `pytest` 수집 단계 통과.

### 1주 내

1. `pipeline.py` artifact 저장/validation/training 단계 분리.
2. `kakao_colab_bot.py`에서 formatter/cache/job runner 분리.
3. result schema test 추가.
4. 설정 strict validation 추가.
5. atomic write/file lock 도입.

### 1개월 내

1. provider abstraction 도입.
2. model artifact schema/versioning 정착.
3. CI 추가: Python 3.10~3.12, unit/smoke 분리.
4. 백테스트 benchmark와 누수 점검 리포트 자동화.
5. 운영 로그/메트릭/장애 대응 문서화.

## 11. 추천 PR 분해

1. `Fix dependency metadata and supported Python range`
2. `Make external integrations lazy-import safe`
3. `Repair Korean encoding and centralize output schema`
4. `Split pipeline artifact and validation stages`
5. `Extract Kakao bot cache/job/message services`
6. `Add atomic artifact writes and cache locking`
7. `Add model/report version fields and schema tests`

## 12. 성공 기준

- `python -m pip install -e ".[dev]"` 후 `pytest` 수집 성공.
- `stock-predict --input data/sample_ohlcv.csv --disable-external` 성공.
- 외부 API key 없이도 smoke test 성공.
- `result_simple.csv`, `result_detail.csv`, `pm_report.json` schema가 테스트로 고정.
- Kakao bot은 캐시 hit, stale cache, live fetch failure, bootstrap running 상태를 구분해서 응답.
- `src/pipeline.py`와 `src/chatbot/kakao_colab_bot.py`의 책임이 명확히 분리되어 신규 기능이 대형 파일에 계속 누적되지 않음.
