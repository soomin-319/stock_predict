# 프로젝트 폴더 구조 분석 및 개선 제안

분석일: 2026-06-04  
대상 경로: `C:\Users\카운\Desktop\stock_predict`

## 1. 요약

현재 저장소는 `src/` 아래에 데이터 수집, 피처 생성, 모델, 검증, 추론, 리포트, 도메인 정책, 챗봇을 분리한 계층형 구조다. 전체 방향은 좋다. 특히 예측 의사결정 기준을 `predicted_return`으로 고정하고, 뉴스/공시/뉴스임팩트는 표시용 컨텍스트로 분리한 점이 명확하다.

다만 몇몇 파일이 지나치게 커졌고, 운영 산출물(`result/`)과 개발 문서/아카이브가 늘어나면서 새 기여자가 핵심 진입점을 빠르게 파악하기 어렵다. 우선순위는 **대형 모듈 분할**, **챗봇/파이프라인 오케스트레이션 정리**, **패키징/테스트 설정 일원화**, **문서 인덱스 정리**다.

## 2. 현재 최상위 구조

```text
.
├─ src/                  # 핵심 Python 패키지
├─ tests/                # pytest 테스트
├─ data/                 # 샘플/유니버스/종목명 매핑 CSV
├─ data/news_impact/     # 뉴스임팩트 예시 CSV
├─ configs/              # 실행 설정 예시/프리셋 JSON
├─ docs/                 # 운영/아키텍처/분석 문서
├─ docs/archive/         # 과거 분석/계획 문서 보관
├─ colab/                # Colab 실행 래퍼
├─ result/               # 생성 산출물, gitignore 대상
├─ .agents/              # 에이전트/플러그인 메타데이터
├─ .codex-plugins/       # 로컬 Codex 플러그인
├─ README.md
├─ AGENTS.md
├─ pyproject.toml
└─ requirements.txt
```

## 3. 현재 `src/` 책임 경계

```text
src/
├─ pipeline.py                 # CLI + run_pipeline 오케스트레이션
├─ pipeline_support.py         # 파이프라인 보조 유틸
├─ config/                     # AppConfig, 설정 병합
├─ data/                       # CSV 로딩/정제, 실데이터, 유니버스, 투자자 컨텍스트
├─ features/                   # 가격/외부시장/국면/투자 신호 피처
├─ models/                     # LightGBM/sklearn fallback 모델 헤드
├─ validation/                 # walk-forward, OOF, 백테스트, 지표, 튜닝
├─ inference/                  # 최신 예측 프레임 생성
├─ domain/                     # 추천/리스크/정책 필드
├─ reports/                    # CSV/JSON/그림/이슈 요약 출력
├─ recommendation/             # 장마감/실시간 추천 보조 로직
├─ chatbot/                    # Kakao/Colab Flask 웹훅
└─ news_impact/                # vendored 뉴스/공시 영향도 모듈
```

## 4. 장점

- **계층 분리가 대체로 명확함**: `data → features → validation/models → inference/domain → reports` 흐름이 유지된다.
- **운영 가드레일 명확함**: 매수/매도/관망은 `predicted_return`만 사용하고, 뉴스/공시는 표시용으로 제한한다.
- **테스트 범위 넓음**: 파이프라인 스모크, Kakao bot, 신호 정책, 뉴스임팩트, 실데이터 fallback, 시각화 등 핵심 경로가 테스트된다.
- **산출물 위치 일관성**: 생성 CSV/JSON이 `result/` 아래로 모인다.
- **문서가 이미 풍부함**: `docs/ARCHITECTURE.md`, `OPERATIONS.md`, `CODEBASE_ANALYSIS.md`, `ROADMAP.md`가 있어 온보딩 기반이 있다.

## 5. 발견한 구조상 리스크

### 5.1 대형 파일 집중

아래 파일은 단일 책임보다 많은 일을 한다. 변경 리스크와 테스트 비용이 커진다.

| 파일 | 대략 라인 수 | 리스크 |
|---|---:|---|
| `src/chatbot/kakao_colab_bot.py` | 1958 | Flask 라우팅, 세션, 캐시, 작업 큐, 응답 포맷, Colab 실행이 한 파일에 집중 |
| `src/pipeline.py` | 841 | CLI, refresh, 설정, 실행 단계, 출력 연결이 한 파일에 집중 |
| `src/reports/issue_summary.py` | 662 | rule 기반 요약, OpenAI 요약, 포맷/예외 처리가 혼재 |
| `src/news_impact/pipeline.py` | 612 | vendored 패키지 내부 오케스트레이션 비대 |
| `src/features/price_features.py` | 541 | 가격 지표, 타깃, 이벤트/레거시 컬럼 처리가 혼재 |

### 5.2 테스트 파일도 일부 비대

- `tests/test_kakao_colab_bot.py`: 약 1836라인
- `tests/test_pipeline_smoke.py`: 약 503라인
- `tests/test_issue_summary.py`: 약 346라인

테스트가 기능별로 분리되지 않으면 실패 원인 파악이 느려진다.

### 5.3 패키징 설정 수동 관리

`pyproject.toml`의 `[tool.setuptools] packages`가 수동 리스트다. 새 하위 패키지를 만들면 누락될 수 있다. `setuptools.find` 기반으로 바꾸는 편이 안전하다.

### 5.4 의존성 정의 중복

- `requirements.txt`에 `pytest` 포함
- `pyproject.toml`에는 `dev = ["pytest"]`

런타임/개발 의존성 경계가 약하다. 운영 설치에서 테스트 도구가 같이 설치될 수 있다.

### 5.5 문서가 많지만 진입점이 분산됨

`docs/README.md`가 문서 인덱스 역할을 하지만, 분석/아카이브 문서가 많아 현재 기준 문서와 과거 문서 경계가 더 명확해야 한다.

### 5.6 `result/` 로컬 산출물 과다

`result/`는 ignore 대상이지만 로컬에 많은 JSON/CSV와 pytest 임시물이 쌓여 있다. 기능 문제는 아니나, 폴더 탐색과 백업/동기화 비용이 커진다.

### 5.7 `src` 패키지명

현재 패키지명이 실제 도메인명 대신 `src`다. console script는 동작하지만, 외부 사용자는 `import src...` 형태가 직관적이지 않다. 장기적으로 `stock_predict/` 패키지명 전환을 검토할 수 있다.

## 6. 권장 개선안

### P0: 정책/예측 가드레일 유지

구조 개선 중에도 아래 원칙은 건드리지 않는다.

- `predicted_return`만 매수/매도/관망 결정에 사용
- 뉴스/공시/뉴스임팩트는 표시/리뷰 컨텍스트만 담당
- 생성 산출물은 `result/` 아래 저장
- CSV는 Windows/Excel 호환을 위해 `utf-8-sig` 유지
- 테스트는 live network 없이 deterministic fixture 우선

### P1: `kakao_colab_bot.py` 분할

제안 구조:

```text
src/chatbot/
├─ kakao_colab_bot.py      # main, Flask app wiring만 유지
├─ app.py                  # Flask route/create_app
├─ state.py                # 세션/잡 상태 모델
├─ cache.py                # result_simple/detail/news 로딩과 캐시
├─ jobs.py                 # 백그라운드 예측 job 시작/상태 관리
├─ responses.py            # Kakao 응답 포맷/메시지 빌더
├─ symbol_resolver.py      # 종목명/코드 검색
└─ colab_runtime.py        # ngrok/Colab 특화 실행 보조
```

효과:

- 챗봇 응답 변경과 job/cache 변경을 독립 테스트 가능
- monkeypatch/레거시 호환 코드 위치가 명확해짐
- `tests/test_kakao_colab_bot.py`를 기능별 테스트로 분리 가능

### P1: `pipeline.py` 오케스트레이션 분리

제안 구조:

```text
src/pipeline.py             # CLI parser, main, run_pipeline 공개 API 유지
src/pipeline_steps/
├─ load_context.py          # 설정/입력/유니버스/컨텍스트
├─ build_features.py        # price/external/regime/investor feature 단계
├─ validate.py              # walk-forward/OOF/baseline/calibration
├─ train_predict.py         # 최종 모델 학습/최신 예측
└─ write_outputs.py         # reports/json 저장
```

주의:

- 기존 안정 인터페이스 `src.pipeline.run_pipeline(...)`, `build_cli_parser()` 유지
- 테스트 호환 wrapper는 남기되 신규 로직은 step 모듈로 이동
- 단계별 입력/출력 dataclass를 두면 회귀 테스트가 쉬워짐

### P1: `issue_summary.py` 분할

제안 구조:

```text
src/reports/issue_summary/
├─ __init__.py
├─ rules.py          # rule-based summary
├─ llm.py            # OpenAI 호출/timeout/fallback
├─ formatting.py     # 사용자 표시 문자열
└─ schema.py         # 입력/출력 컬럼 정의
```

효과:

- OpenAI 없는 테스트와 rule-only 테스트가 분리됨
- API 장애 fallback 검증이 쉬워짐

### P2: `price_features.py` 분할

제안 구조:

```text
src/features/
├─ price_features.py        # 공개 build_features/select_feature_columns 유지
├─ technical_indicators.py  # RSI, MACD, ATR, OBV 등
├─ targets.py               # 1d/5d/20d target 생성
├─ event_features.py        # 수급/거래대금/52주 신고가 등
└─ feature_selection.py     # 모델 입력 컬럼 선택/제외 정책
```

주의:

- display-only 컬럼이 feature list에 들어가지 않는 guard test 유지/강화
- 기존 import 경로 호환을 위해 wrapper 유지

### P2: 테스트 파일 분할

제안:

```text
tests/chatbot/
├─ test_cache.py
├─ test_jobs.py
├─ test_responses.py
├─ test_symbol_resolver.py
└─ test_webhook.py

tests/pipeline/
├─ test_cli.py
├─ test_refresh.py
├─ test_outputs.py
└─ test_smoke.py

tests/reports/
├─ test_issue_summary_rules.py
├─ test_issue_summary_llm_fallback.py
└─ test_result_formatter.py
```

효과:

- 실패 범위 축소
- 기능별 fixture 재사용 증가
- CI 로그 탐색 쉬움

### P2: 패키징/의존성 정리

권장 변경:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

또는 장기 전환 시:

```text
stock_predict/
└─ ...
```

의존성 권장:

- `requirements.txt`: 운영 실행용 최소 의존성
- `requirements-dev.txt` 또는 `.[dev]`: pytest/개발 도구
- 선택 의존성 분리 검토
  - `.[news]`: OpenAI/Naver/DART 관련
  - `.[chatbot]`: Flask/pyngrok
  - `.[model]`: lightgbm

### P2: 문서 인덱스 정리

권장:

- `docs/README.md`에 “현재 기준 문서”와 “과거 아카이브” 구분 강화
- `docs/CODEBASE_ANALYSIS.md`는 상세 분석 유지
- 이 문서는 구조 개선 체크리스트로 연결
- 과거 문서는 `docs/archive/analysis-reports/`에 계속 보관

예시:

```text
docs/
├─ README.md                         # 문서 지도
├─ ARCHITECTURE.md                   # 현재 아키텍처
├─ OPERATIONS.md                     # 운영 절차
├─ PROJECT_STRUCTURE_REVIEW_2026-06-04.md
└─ archive/
```

### P3: 로컬 산출물 관리

권장:

- 주기적으로 `result/` 정리 스크립트 추가 검토
- 보존할 artifact는 `result/keep/` 또는 날짜별 폴더로 분리
- pytest tmp는 `result/.pytest_tmp` 아래 유지하되 정리 명령 문서화

예시 명령:

```powershell
Remove-Item -Recurse -Force result/.pytest_tmp
```

주의: 위 명령은 로컬 생성물 삭제이므로 실행 전 경로 확인 필요.

## 7. 목표 구조 예시

단기적으로는 패키지명 전환 없이 다음 형태가 현실적이다.

```text
src/
├─ pipeline.py
├─ pipeline_steps/
├─ chatbot/
│  ├─ kakao_colab_bot.py
│  ├─ app.py
│  ├─ cache.py
│  ├─ jobs.py
│  ├─ responses.py
│  └─ symbol_resolver.py
├─ data/
├─ features/
│  ├─ price_features.py
│  ├─ technical_indicators.py
│  ├─ targets.py
│  └─ feature_selection.py
├─ models/
├─ validation/
├─ inference/
├─ domain/
├─ reports/
│  └─ issue_summary/
├─ recommendation/
└─ news_impact/
```

장기적으로는 다음 이름 전환을 검토한다.

```text
stock_predict/
├─ pipeline.py
├─ chatbot/
├─ data/
├─ features/
└─ ...
```

단, 이 전환은 import 경로와 console script, 테스트 전체에 영향이 크므로 별도 PR로 진행하는 것이 좋다.

## 8. 권장 작업 순서

1. **문서/테스트 기준 고정**
   - 현재 smoke/Kakao/signal-policy 테스트 통과 확인
   - display-only guard 테스트 유지
2. **챗봇 모듈 분할**
   - cache/responses/jobs부터 추출
   - 기존 public 함수/API 유지
3. **파이프라인 단계 분할**
   - 13단계 로그와 산출물 이름 유지
   - step별 dataclass/context 도입 검토
4. **리포트/피처 대형 파일 분할**
    - `issue_summary`, `price_features` 순서 권장
5. **테스트 디렉터리 재배치**
   - 기능별 테스트 파일로 나누되 fixture 중복 제거
6. **패키징/의존성 정리**
   - `setuptools.find` 도입
   - dev/runtime dependency 경계 정리
7. **장기 패키지명 변경 검토**
   - `src` → `stock_predict`는 큰 변경이므로 마지막에 별도 진행

## 9. 빠른 체크리스트

- [ ] `src/chatbot/kakao_colab_bot.py`를 cache/jobs/responses/app 단위로 분할
- [ ] `tests/test_kakao_colab_bot.py`를 기능별 테스트로 분할
- [ ] `src/pipeline.py`의 13단계를 `src/pipeline_steps/`로 이동
- [ ] `src/reports/issue_summary.py`를 rules/llm/formatting/schema로 분리
- [ ] `src/features/price_features.py`를 technical/targets/selection으로 분리
- [ ] `pyproject.toml` 패키지 검색 자동화
- [ ] runtime/dev 의존성 분리
- [ ] `docs/README.md`에 이 문서 링크 추가
- [ ] `result/` 정리 운영 절차 문서화

## 10. 결론

현재 구조는 연구용 주식 예측 파이프라인으로서 큰 책임 경계가 잘 잡혀 있다. 가장 큰 개선 포인트는 폴더 추가보다 **비대한 파일을 기존 경계에 맞춰 작게 나누는 것**이다. 우선 `chatbot`, `pipeline`, `issue_summary`, `price_features`를 순서대로 분할하면 유지보수성과 테스트 속도가 가장 크게 좋아질 것이다.
