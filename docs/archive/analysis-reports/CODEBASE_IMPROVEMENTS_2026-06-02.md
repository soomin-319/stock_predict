# 코드베이스 분석 및 개선/수정 제안

작성일: 2026-06-02 KST  
대상 브랜치: `codex/kospi200-high-score-recommendations`  
범위: `src/`, `news_impact/`, `tests/`, 설정/데이터/문서 일부

## 요약

- 현재 구조는 가격/시장/수급 피처, walk-forward 검증, long-only top-k 백테스트, Kakao/Colab 챗봇, `news_impact` 표시 컨텍스트가 분리되어 있다.
- 핵심 가드레일은 유지해야 한다: 매수/매도/보류 판단은 `predicted_return` 기반이어야 하며, 뉴스/공시는 표시 컨텍스트로만 사용해야 한다.
- 즉시 수정 우선순위는 테스트 실행 안정화, KRX 종목명 캐시 오염, 대형 모듈 분해, 의존성/옵션 import 정리다.

## 현재 상태 지표

| 항목 | 관찰 |
|---|---:|
| Python 파일 | 106개 |
| Python 라인 수 | 약 17,893줄 |
| 테스트 파일 | 24~25개 |
| 가장 큰 소스 | `src/chatbot/kakao_colab_bot.py` 약 1,859줄 |
| 가장 큰 함수/클래스 | `KakaoColabPredictionBot` 1,422줄, `run_pipeline` 420줄, `build_features` 282줄 |
| tracked `result/` 파일 | 없음 |

## 검증 결과

### 1) 기본 `pytest -q`

결과:

```text
82 passed, 82 warnings, 80 errors in 6.25s
```

주요 원인:

```text
PermissionError: [WinError 5] 액세스가 거부되었습니다:
...\stock_predict\result\.pytest_tmp
```

`pyproject.toml`에서 pytest 임시/캐시 경로를 `result/` 아래로 강제한다.

```toml
[tool.pytest.ini_options]
cache_dir = "result/.pytest_cache"
addopts = "--basetemp=result/.pytest_tmp"
```

현재 로컬 `result/.pytest_tmp`, `result/.pytest_cache`는 ACL 접근 문제가 있어 테스트 setup 단계에서 대량 error가 난다.

### 2) 임시 경로 우회 실행

명령:

```powershell
pytest -q --basetemp=C:\tmp\stock_predict_pytest_tmp -o cache_dir=C:\tmp\stock_predict_pytest_cache
```

결과:

```text
161 passed, 1 failed
```

실제 남은 실패:

```text
FAILED tests/test_realtime_close_betting.py::test_default_realtime_service_uses_bundled_universe_without_pykrx
```

원인 추정:

- `tests/test_krx_symbol_names.py`가 `krx_universe.KRX_SYMBOL_NAME_CSV`를 monkeypatch한다.
- `_load_krx_symbol_name_df()`는 `@lru_cache(maxsize=1)`라 monkeypatch된 CSV 결과가 다음 테스트까지 남을 수 있다.
- 이후 `RealTimeCloseBettingRecommendationService` 기본 universe 로드 시 `000660.KS` 이름이 `SK하이닉스`가 아니라 `000660.KS` fallback으로 남는다.

## P0: 즉시 수정 권장

### P0-1. pytest 임시/캐시 경로를 `result/` 밖으로 이동

문제:

- `result/`는 생성 산출물 디렉터리다.
- 테스트 framework 내부 상태인 `.pytest_tmp`, `.pytest_cache`를 `result/`에 두면 권한/정리/산출물 관리가 섞인다.

권장 수정:

```toml
[tool.pytest.ini_options]
cache_dir = ".pytest_cache"
# addopts의 --basetemp 제거
```

필요하면 `.gitignore`에 추가:

```gitignore
.pytest_tmp/
```

대안:

- CI 전용으로만 `--basetemp` 지정.
- 로컬은 pytest 기본 임시 디렉터리 사용.

### P0-2. KRX 종목명 CSV loader 캐시 오염 수정

문제:

- `_load_krx_symbol_name_df()`가 path를 인자로 받지 않는 전역 `lru_cache`다.
- 테스트/런타임에서 `KRX_SYMBOL_NAME_CSV`가 바뀌면 캐시가 실제 path와 불일치한다.

권장 수정:

```python
@lru_cache(maxsize=4)
def _load_krx_symbol_name_df_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    ...

def _load_krx_symbol_name_df() -> pd.DataFrame:
    return _load_krx_symbol_name_df_cached(str(KRX_SYMBOL_NAME_CSV))
```

테스트 보강:

- monkeypatch 후뿐 아니라 teardown에서도 cache clear.
- 같은 프로세스에서 기본 CSV 복귀 후 `000660.KS -> SK하이닉스`가 복구되는 회귀 테스트 추가.

### P0-3. 뉴스/공시 display-only 불변성 테스트 강화

이미 `append_news_impact_context` 테스트가 있으나, 핵심 정책은 더 넓게 고정해야 한다.

추가 권장:

- `predicted_return` 값이 뉴스/공시 merge 전후 동일.
- expected-return ranking 동일.
- recommendation/signal decision이 뉴스/공시 컬럼 추가만으로 바뀌지 않음.
- 챗봇 추천 키워드도 뉴스 impact score를 의사결정 score로 쓰지 않음.

## P1: 구조 개선

### P1-1. `src/chatbot/kakao_colab_bot.py` 분해

현재:

- 파일 약 1,859줄.
- `KakaoColabPredictionBot` 클래스 약 1,422줄.
- 캐시, 세션, Flask handler, Kakao response, pipeline job, live issue summary, ngrok/Colab 실행이 한 클래스에 결합.

권장 분리:

| 신규 모듈 | 책임 |
|---|---|
| `src/chatbot/state_store.py` | job/session JSON read/write, stale state 처리 |
| `src/chatbot/kakao_response.py` | Kakao simpleText/quick replies formatting |
| `src/chatbot/pipeline_jobs.py` | subprocess 실행, prewarm, progress log |
| `src/chatbot/issue_summary_service.py` | cached/live issue summary 생성 |
| `src/chatbot/app.py` | Flask route/create_app |

기대효과:

- 테스트가 작은 단위로 분리.
- timeout/live fetch/mock 범위 축소.
- Kakao payload 제한/format 회귀 테스트가 더 쉬워짐.

### P1-2. `src/pipeline.py`의 orchestration 분리

현재:

- `run_pipeline()` 약 420줄.
- fetch, clean, feature, train, validation, prediction, report, figure, issue/news context가 한 함수에 이어져 있다.
- `src.reports.output` 함수 위임 wrapper가 많다.

권장 분리:

| 단계 | 후보 함수/모듈 |
|---|---|
| 데이터 준비 | `pipeline_steps/prepare_data.py` |
| 피처 생성 | `pipeline_steps/build_features.py` |
| 학습/OOF | `pipeline_steps/train_validate.py` |
| 최신 예측 | `pipeline_steps/predict_latest.py` |
| 리포트/figure | `pipeline_steps/write_artifacts.py` |
| 외부 context | `pipeline_steps/context.py` |

추가:

- backward-compatible wrapper는 테스트/importer 이전 기간만 유지하고 deprecation 주석 추가.

### P1-3. `build_features()` 분해

현재:

- `src/features/price_features.py::build_features()` 약 282줄.

권장:

- 가격/수익률 피처
- rolling/volatility 피처
- technical indicator 피처
- event/risk flag 피처
- final column cleanup

각 단계별 입력/출력 컬럼 contract를 테스트로 고정.

## P2: 품질/운영 개선

### P2-1. 의존성 정리

현재:

- `pyproject.toml`은 `numpy`, `pandas`, `scikit-learn`, `yfinance`, `matplotlib`, `lightgbm`, `openai`, `flask`, `pyngrok`를 모두 runtime dependency로 둔다.
- `tests/test_p0_import_and_encoding.py`는 기본 import path가 market/LLM extras 없이 import되어야 한다고 검증한다.

권장:

- core dependency와 optional extras 분리.

예:

```toml
[project.optional-dependencies]
ml = ["scikit-learn", "lightgbm"]
live = ["yfinance", "pykrx"]
llm = ["openai"]
chatbot = ["flask", "pyngrok"]
dev = ["pytest"]
```

### P2-2. `requirements.txt`와 `pyproject.toml` 동기화

현재:

- `requirements.txt`는 `numpy>=2.0,<2.3`.
- `pyproject.toml`은 `numpy`만 명시.

권장:

- 버전 제약을 한 곳에서 관리하거나 둘을 동일하게 유지.
- 특히 `pandas`, `scikit-learn`, `lightgbm`, `openai`는 major update에 민감하므로 최소/상한 검토.

### P2-3. 인코딩/콘솔 출력 가이드 보강

관찰:

- CSV와 소스는 UTF-8 계열로 읽히지만 Windows PowerShell 출력에서 한글이 깨져 보일 수 있다.
- 산출 CSV는 Excel/Windows 호환을 위해 `utf-8-sig` 유지가 맞다.

권장:

- 운영 문서에 다음을 추가:

```powershell
$env:PYTHONIOENCODING="utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### P2-4. live network 테스트 차단 기본값

권장:

- pytest fixture로 외부 HTTP/yfinance/pykrx 호출을 기본 차단.
- live path 테스트는 명시 marker 필요.

예:

```python
@pytest.mark.live
def test_live_fetch_smoke(...):
    ...
```

### P2-5. result 산출물 청소 스크립트

현재:

- `result/`에 로컬 산출물이 많다.
- tracked는 아니지만 테스트와 운영 산출물이 섞인다.

권장:

- `scripts/clean_results.py` 또는 `python -m src.reports.clean_outputs`.
- 삭제 대상은 `result/` 내부로만 제한하고 dry-run 지원.

## 제안 작업 순서

1. `pyproject.toml` pytest 경로 수정.
2. KRX symbol-name loader 캐시를 path-keyed cache로 변경.
3. 위 두 수정 후 `pytest -q` 재실행.
4. 뉴스/공시 display-only 불변성 테스트 추가.
5. 챗봇/파이프라인 대형 모듈 단계적 분해.
6. optional dependencies 정리.

## 확인한 주요 명령

```powershell
pytest -q
pytest -q --basetemp=C:\tmp\stock_predict_pytest_tmp -o cache_dir=C:\tmp\stock_predict_pytest_cache
pytest -q tests/test_p0_import_and_encoding.py --basetemp=C:\tmp\stock_predict_pytest_tmp -o cache_dir=C:\tmp\stock_predict_pytest_cache
```

## 주의

이 프로젝트의 출력은 리서치/운영 보조용이다. 자동매매 또는 투자 조언으로 취급하면 안 된다. 신호 결정은 반드시 다음 날 기대수익률(`predicted_return`) 기반 정책을 따라야 하며, 뉴스/공시는 표시/검토 컨텍스트로만 유지해야 한다.
