# 코드베이스 분석 및 개선 제안

- 분석일: 2026-06-06
- 대상: `stock_predict`
- 범위: 구조, 테스트, 파이프라인 실행, 모델·설정 검증, 외부 연동, 챗봇 동시성, 유지보수성

## 1. 요약

현재 코드베이스는 핵심 기능과 회귀 테스트가 잘 갖춰져 있다. 쓰기 가능한 임시 경로를 사용하면 전체 테스트 **206개가 통과**하며, 샘플 데이터 파이프라인도 정상 완료된다. 뉴스·공시 데이터가 모델 예측과 추천 정책을 변경하지 않도록 하는 테스트도 존재한다.

다만 운영 안정성 측면에서 다음 문제를 우선 해결해야 한다.

1. 챗봇 타임아웃이 실행 중 작업을 실제로 중단하지 못한다.
2. 테스트 임시 디렉터리를 저장소 내부 고정 경로로 강제해 권한·소유자 변경 시 테스트가 대량 실패한다.
3. 외부 뉴스·공시 수집 실패를 광범위하게 숨겨 장애 원인과 데이터 누락을 파악하기 어렵다.
4. 잘못된 설정 키와 분위수 설정을 늦게 또는 전혀 검증하지 않는다.

## 2. 검증 결과

### 성공

| 검증 | 결과 |
|---|---|
| 전체 테스트, 쓰기 가능한 별도 basetemp 사용 | `206 passed in 20.78s` |
| 샘플 파이프라인 | 정상 완료, 예측 결과 출력 |
| Python 문법·모듈 import | 전체 테스트 실행으로 확인 |
| 뉴스·공시 display-only 경계 | 관련 테스트 통과 |
| 결과 CSV 핵심 경로 | `utf-8-sig`, 임시 파일 후 replace 방식 사용 |

실행 명령:

```powershell
python -m pytest -q --basetemp result\analysis_pytest_tmp\base
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_analysis.json
```

### 최초 전체 테스트 실패

기본 `pytest` 실행은 `result/.pytest_tmp/pytest-of-카운` 접근 거부로 **109 passed, 97 errors**가 발생했다. 코드 로직 실패가 아니라 테스트 임시 경로 권한 문제였다.

## 3. 구조 평가

### 장점

- `src/data`, `features`, `models`, `validation`, `inference`, `reports`, `domain`으로 책임이 분리되어 있다.
- walk-forward OOF, 백테스트, 확률 보정, 신호 정책에 대한 테스트가 존재한다.
- `src/features/feature_selection.py`가 뉴스·공시 및 파생 컨텍스트 열을 모델 입력에서 제외한다.
- `tests/test_display_only_feature_guard.py`와 `tests/test_news_impact_context.py`가 display-only 정책을 회귀 검증한다.
- 핵심 결과 CSV는 atomic replace와 `utf-8-sig`를 사용한다.

### 규모

- 소스: 84개 Python 파일, 약 13,425줄
- 테스트: 31개 Python 파일, 206개 테스트, 약 5,258줄
- 큰 모듈:
  - `src/chatbot/kakao_colab_bot.py`: 1,979줄
  - `src/pipeline.py`: 1,007줄
  - `src/reports/issue_summary.py`: 661줄

## 4. 발견 문제와 해결책

## P1. 타임아웃 후 작업이 계속 실행됨

**위치:** `src/chatbot/kakao_colab_bot.py:1607-1617`

`Future.cancel()`은 이미 실행 중인 스레드를 중단하지 못한다. 현재 구현은 타임아웃 응답을 반환한 뒤에도 뉴스·공시 수집 또는 요약 작업이 계속 실행될 수 있다.

재현 결과:

```text
returned None elapsed 0.021 event_immediate False
event_later True
```

**영향**

- 요청이 반복되면 백그라운드 스레드와 외부 API 호출이 누적될 수 있다.
- 타임아웃 후 완료된 작업이 캐시·상태를 뒤늦게 변경할 수 있다.
- 중복 비용과 예측 불가능한 상태 변경이 발생한다.

**해결책**

1. 외부 호출 자체에 연결·읽기 타임아웃을 전달한다.
2. 작업 함수에 `threading.Event` 기반 취소 토큰을 전달하고 단계 사이에서 확인한다.
3. 장시간·강제 종료 필요 작업은 스레드 대신 별도 프로세스 또는 작업 큐로 격리한다.
4. 동일 종목 작업을 단일-flight 방식으로 중복 방지한다.
5. “응답 타임아웃”과 “작업 취소 완료”를 별도 상태로 기록하는 테스트를 추가한다.

## P1. 테스트 임시 경로가 권한 변화에 취약함

**위치:** `tests/conftest.py:9-14`

테스트가 `result/.pytest_tmp`를 고정 생성하고 `tempfile.tempdir`까지 전역 변경한다. 이 디렉터리가 다른 사용자·샌드박스 소유가 되면 `tmp_path`를 사용하는 다수 테스트가 setup 단계에서 실패한다.

**영향**

- 실제 코드 결함 없이 테스트 97개가 동시에 오류 처리된다.
- 개발 환경, CI, 샌드박스 사이에서 재현성이 떨어진다.
- 전역 `tempfile.tempdir` 변경이 테스트 격리를 약화한다.

**해결책**

1. `tempfile.tempdir` 전역 변경을 제거하고 pytest의 기본 `tmp_path`를 사용한다.
2. 저장소 내부 경로가 꼭 필요하면 세션·프로세스별 고유 경로를 만든다.
3. 시작 시 쓰기 가능 여부를 검사하고 명확한 오류를 출력한다.
4. CI에서는 `--basetemp`를 명시하되 실행별 고유 경로를 사용한다.

## P1. 외부 컨텍스트 수집 실패가 조용히 숨겨짐

**위치**

- `src/pipeline.py:380-383`
- `src/data/investor_context.py:145`, `192`, `325`
- `src/reports/issue_summary.py`의 다수 광범위 예외 처리

뉴스·공시 원시 이벤트 수집 중 예외가 발생하면 빈 DataFrame 또는 빈 결과로 대체되며, 일부 경로에서는 원인·심볼·공급자 정보가 기록되지 않는다.

뉴스·공시는 display-only이므로 `predicted_return`을 변경하지 않는 현재 정책은 적절하다. 그러나 운영자는 “이슈 없음”과 “수집 장애”를 구분하기 어렵다.

**해결책**

1. `except Exception`에서 최소한 공급자, 심볼, 날짜 범위, 예외 타입을 구조화 로그로 남긴다.
2. pipeline report에 `context_collection_status`, `failed_symbols`, `error_types`를 기록한다.
3. 빈 결과에 `no_events`와 `collection_failed` 상태를 구분한다.
4. 네트워크·파싱·인증 오류를 타입별로 처리한다.
5. 실패 상태가 예측·추천을 변경하지 않는 회귀 테스트를 유지한다.

## P2. 잘못된 설정 키가 조용히 무시됨

**위치:** `src/config/settings.py:104-115`

`_merge_dataclass_config()`는 알 수 없는 키를 `continue`로 무시한다. 예를 들어 `min_trian_size` 오타를 넣어도 오류 없이 기본값 `756`이 유지된다.

**영향**

- 운영 설정이 적용되지 않았는데 정상 적용된 것으로 오인할 수 있다.
- 백테스트 조건과 모델 학습 조건의 재현성이 훼손된다.

**해결책**

- 기본은 unknown key에 `ValueError`를 발생시킨다.
- 이전 설정 호환이 필요하면 `strict=False` 모드에서 경고와 무시된 키 목록을 반환한다.
- 숫자 범위와 리스트 길이도 로드 시 검증한다.

## P2. 분위수 설정 검증이 너무 늦음

**위치:** `src/models/lgbm_heads.py:125-187`

모델은 분위수 1개만 전달해도 학습을 완료하지만, 예측 시점에야 “최소 3개 필요” 오류를 발생시킨다.

재현 결과:

```text
fit_quantile_heads [0.5]
RuntimeError MultiHeadPrediction requires at least 3 quantile heads, got 1: [0.5]
```

또한 독립 분위수 모델은 분위수 교차(`low > high`)가 가능하지만 예측 프레임 생성 시 이를 명시적으로 방어하지 않는다.

**해결책**

1. `fit()` 시작 시 분위수 개수, 정렬, 중복, `0 < q < 1`을 검증한다.
2. low/median/high 의미를 명시적으로 매핑한다.
3. 예측 후 분위수 교차율을 측정하고 report에 기록한다.
4. 필요하면 행별 정렬 또는 monotonic quantile 모델을 적용한다.

## P2. 대형 함수와 광범위 예외 처리로 변경 위험 증가

**근거**

- `src/features/price_features.py:81-356` `build_features`: 276줄
- `src/validation/backtest.py:130-293` `run_long_only_topk_backtest`: 164줄
- `src/pipeline.py:551-710` `_write_pipeline_artifacts`: 160줄
- `src/chatbot/kakao_colab_bot.py`: 광범위 `except Exception` 25개

**영향**

- 기능 변경 시 부작용 범위가 넓다.
- 예외 원인과 책임 경계가 불명확하다.
- 단위 테스트보다 통합 테스트 의존도가 높아진다.

**해결책**

- 특징 생성은 기술 지표, 투자자 흐름, 이벤트 파생값, 타깃 생성으로 분리한다.
- 백테스트는 eligibility, ranking, position sizing, cost, metrics 단계로 분리한다.
- artifact 저장은 detail/simple/context/report writer로 분리한다.
- 예외는 경계 계층에서만 변환하고 내부 로직은 구체적 예외를 유지한다.

## P2. 의존성 버전 재현성이 낮음

**위치:** `requirements.txt`, `pyproject.toml`

NumPy 외 대부분의 핵심 의존성에 상한 또는 고정 버전이 없다.

**영향**

- pandas, scikit-learn, LightGBM, yfinance 변경으로 동일 코드의 결과·API가 달라질 수 있다.
- 연구 결과와 운영 장애 재현이 어려워진다.

**해결책**

- 직접 의존성에 검증된 호환 범위를 지정한다.
- 배포·CI용 lock 파일 또는 constraints 파일을 생성한다.
- Python 3.10~3.14 지원 매트릭스에서 테스트한다.
- 모델 artifact metadata에 라이브러리 버전을 기록한다.

## P3. 정적 품질 검증 단계 부족

**위치:** `requirements-dev.txt`, `pyproject.toml`

개발 의존성은 `pytest`만 포함하며 lint, import 검사, 타입 검사 설정이 없다.

**해결책**

- 최소 단계: Ruff 또는 Flake8, `python -m compileall`, import smoke.
- 핵심 경계부터 mypy/pyright를 점진 적용한다.
- CI에서 테스트와 정적 검사를 별도 job으로 실행한다.

## P3. 모델 artifact 로드는 신뢰 경계가 필요함

**위치:** `src/models/lgbm_heads.py:236`

`joblib.load()`는 pickle 기반이므로 신뢰할 수 없는 파일을 로드하면 임의 코드 실행 위험이 있다.

**해결책**

- 사용자 업로드 또는 외부 다운로드 artifact를 직접 로드하지 않는다.
- 허용 디렉터리, 파일 해시, 서명 또는 manifest 검증을 적용한다.
- 운영 문서에 “신뢰된 내부 artifact만 로드” 제약을 명시한다.

## 5. 권장 실행 순서

1. 테스트 임시 경로 고정 제거 및 CI 재현성 복구.
2. 챗봇 타임아웃 작업의 실제 취소·중복 방지 설계 적용.
3. 외부 컨텍스트 실패 상태와 구조화 로그 추가.
4. 설정·분위수 fail-fast 검증 추가.
5. 대형 함수 단계적 분리.
6. dependency constraints와 정적 검사 CI 추가.

## 6. 제안 테스트

- 타임아웃 이후 작업이 상태·캐시를 변경하지 않는지 검증.
- 동일 종목 동시 요청이 외부 호출 1회만 생성하는지 검증.
- 컨텍스트 수집 실패와 “이벤트 없음”이 report에서 구분되는지 검증.
- 잘못된 설정 키와 범위가 로드 단계에서 실패하는지 검증.
- 분위수 3개 미만, 중복, 범위 밖 값이 학습 전에 실패하는지 검증.
- 분위수 교차율이 허용 기준을 넘으면 경고 또는 실패하는지 검증.

## 7. 결론

핵심 예측 파이프라인과 정책 경계는 현재 테스트 기준으로 정상이다. 가장 큰 위험은 모델 로직 자체보다 **운영 동시성, 실패 관측성, 환경 재현성**이다. P1 항목을 먼저 해결하면 실제 장애 탐지와 챗봇 안정성이 크게 개선된다.
