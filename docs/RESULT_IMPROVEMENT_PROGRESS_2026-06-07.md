# Result 개선 작업 진행 상황

기준일: 2026-06-07

관련 문서:

- [`RESULT_ANALYSIS_AND_IMPROVEMENTS.md`](RESULT_ANALYSIS_AND_IMPROVEMENTS.md)
- [`superpowers/specs/2026-06-07-result-artifact-hardening-design.md`](superpowers/specs/2026-06-07-result-artifact-hardening-design.md)
- [`superpowers/plans/2026-06-07-result-artifact-hardening.md`](superpowers/plans/2026-06-07-result-artifact-hardening.md)

## 이번 작업 범위

상세 구현 계획의 Task 1~9를 구현하고 검증했다.

## 완료된 작업

### Task 1. 비밀정보 마스킹 및 runtime 상태 보안

구현 파일:

- `src/utils/secrets.py`
- `src/chatbot/kakao_colab_bot.py`
- `tests/test_secret_redaction.py`
- `tests/test_kakao_colab_bot.py`

완료 내용:

- API 키, token, secret, password 성격의 명령 인자를 `[REDACTED]`로 마스킹한다.
- subprocess에는 실제 비밀값을 환경변수로 전달하고, 상태 JSON에는 마스킹된 명령만 기록한다.
- 콘솔 메시지와 subprocess 파일 로그에 등록된 실제 비밀값이 기록되지 않도록 한다.
- runtime 상태 JSON 저장 전에 중첩 데이터를 재귀적으로 마스킹한다.
- 완료·실패 작업 상태에서 상세 명령과 PID를 제거한다.
- runtime JSON은 UTF-8 원자 저장한다.

운영자 수동 조치:

- 기존 `result/chatbot_jobs.json`, 로그, 백업에 노출된 API 키는 코드 수정만으로 안전해지지 않는다.
- 노출된 OpenAI, DART, Naver 자격증명을 반드시 폐기하고 재발급해야 한다.

### Task 2. 공통 실행 메타데이터 및 UTF-8 JSON

구현 파일:

- `src/reports/report_metadata.py`
- `src/reports/pm_report.py`
- `src/pipeline.py`
- `tests/test_report_metadata.py`
- `tests/test_pipeline_smoke.py`

완료 내용:

- 실행마다 UTC timestamp와 임의 식별자를 조합한 `run_id`를 생성한다.
- pipeline report와 PM report에 다음 공통 필드를 포함한다.
  - `schema_version`
  - `run_id`
  - `environment`
  - `data_mode`
  - `generated_at`
  - `input_as_of_date`
  - `prediction_for_date`
  - `context_as_of_date`
  - `git_commit`
  - `config_hash`
  - `status`
  - `blocking_reasons`
- 샘플 입력 파일은 현재 `environment=smoke`, `data_mode=sample`로 분류한다.
- pipeline/PM JSON을 UTF-8 원자 저장한다.
- `run_pipeline()`은 생성된 report dict를 반환한다.

### Task 3. 실행별 산출물, manifest, latest 승격

구현 파일:

- `src/reports/run_artifacts.py`
- `src/pipeline.py`
- `tests/test_run_artifacts.py`
- `tests/test_pipeline_smoke.py`

완료 내용:

- 핵심 산출물을 `result/runs/<run_id>/`에 기록한다.
- 실행별 구조:

```text
result/runs/<run_id>/
  manifest.json
  pipeline_report.json
  pm_report.json
  csv/
    result_simple.csv
    result_detail.csv
    result_news.csv
    result_disclosure.csv
  figures/
```

- manifest에 상대 경로, SHA-256, 크기, 생성 시각, CSV 행 수를 기록한다.
- 필수 산출물 누락, JSON 파싱 실패, report `run_id` 불일치 시 실행을 `fail`로 처리한다.
- `environment=production`, `data_mode=real`, `status!=fail` 실행만 `result/latest/`로 원자 승격한다.
- 승격 후에만 기존 최상위 CSV와 `pm_report.json` 호환 복사본을 갱신한다.
- smoke/sample 실행과 부분 실패는 기존 `latest/` 및 호환 복사본을 변경하지 않는다.
- 사용자 지정 `--report-json`은 실행 디렉터리 내부 추가 alias로 저장하며, canonical report는 `pipeline_report.json`이다.

## 검증 결과

구현 전 기준선:

```text
205 passed
```

Task 1 집중 검증:

```text
pytest tests/test_secret_redaction.py tests/test_kakao_colab_bot.py -q
65 passed
```

Task 2~3 집중 검증:

```text
pytest tests/test_run_artifacts.py tests/test_pipeline_smoke.py tests/test_report_metadata.py -q
23 passed
```

전체 회귀 검증:

```text
pytest -q
256 passed
```

## 후속 운영 작업

코드 구현과 자동 검증은 완료했다. 운영자는 과거 로그·백업에 노출된 OpenAI, DART,
Naver 자격증명을 직접 폐기하고 재발급해야 한다.

## 구현 완료 작업

### Task 4. 날짜/context 정책 및 sample-safe 챗봇 읽기

완료 항목:

- `src/reports/context_policy.py` 추가
- 가격 기준일과 context 기준일 허용 차이 검증
- 오래된 context 결합 제외 및 제외 사유 기록
- 핵심 CSV에 실행 환경과 기준일 필드 추가
- 챗봇이 검증된 `result/latest/manifest.json`을 우선 읽고, 명시적으로 전달된 호환 경로 또는 운영 메타데이터가 있는 기존 최상위 CSV만 fallback으로 사용
- sample/smoke 결과를 실제 운영 추천으로 제공하지 않도록 차단

주의:

- latest manifest는 `production`/`real`, `status=pass|warning`, `promoted=true` 조건을 모두 만족해야 한다.
- 메타데이터 없는 오래된 최상위 CSV는 기본 챗봇 경로에서 운영 결과로 사용하지 않는다.

### Task 5. 백테스트 및 calibration 유효성

완료 항목:

- `src/validation/result_validity.py` 추가
- 거래 가능 예측 0건, 전체 halt, 평균 선택 0건, 평가일 없음의 blocking reason 기록
- `backtest_valid` 필드 추가
- calibration 표본 부족 시 `ece=null`과 이유 기록
- calibration bin별 표본 수, confidence, accuracy 기록

### Task 6. 뉴스·공시 record type

완료 항목:

- 뉴스·공시 CSV에 `record_type=event|summary|no_data` 추가
- `collection_status`, `no_data_reason`, `collection_error` 추가
- display-only context가 `predicted_return`, 순위, 추천, 신호에 영향을 주지 않는 회귀 테스트 강화

### Task 7. runtime TTL, retention, 안전한 cleanup

완료 항목:

- `src/utils/result_cleanup.py` 추가
- runtime 기본 경로를 `result/runtime/`으로 이동하고 기존 경로 fallback 제공
- 작업·세션 TTL 적용
- 성공 run, 실패 run, 로그, 테스트 임시 산출물 보존 정책 적용
- `latest/`, runtime 상태 JSON, 프로젝트 외부 경로 삭제 방지
- 성공 pytest 임시 산출물 자동 정리 및 `KEEP_TEST_ARTIFACTS=1` 보존 옵션
- `result/test/`의 만료된 실행별 테스트 산출물 정리

### 추가 표적 하드닝

- `RunArtifactManager`의 unsafe `run_id`, 절대 경로, `..` 경로 탈출 차단
- `pass|warning` 외 상태의 latest 승격 차단
- failed/unpromoted latest manifest의 챗봇 사용 차단
- Colab runner가 현재 실행의 run/latest artifact 경로를 반환하도록 수정

### Task 8. 사용자 문서 갱신

완료 항목:

- `RESULT_FILES_GUIDE.md`를 신규 run/latest/runtime/test 구조에 맞게 갱신
- `RESULT_ANALYSIS_AND_IMPROVEMENTS.md`에 항목별 구현 상태 추가
- `RESULT_CLEANUP.md`에 안전한 cleanup 명령과 보존 정책 추가
- README에 manifest 확인 및 smoke 승격 금지 동작 설명

### Task 9. 최종 검증 및 PR

완료 항목:

- Task 4~8 집중 테스트 및 전체 `pytest`
- bundled sample smoke pipeline 실행 검증
- `result/**/*.json`, `result/**/*.log` 실제 환경 비밀값 감사
- 코드 리뷰
- 최종 PR 갱신

## 다음 작업 재개 절차

다음 작업 시작 시:

1. 현재 브랜치와 작업 트리를 확인한다.

```powershell
git status --short --branch
git log --oneline -10
```

2. 기준선 테스트를 실행한다.

```powershell
pytest -q
```

예상 기준선: `256 passed`

3. 추가 변경 시 표적 하드닝 계획을 기준으로 TDD를 유지한다.

```text
docs/superpowers/plans/2026-06-07-result-targeted-hardening.md
```

4. 우선 확인할 호환 지점:

- `src/chatbot/kakao_colab_bot.py`의 기존 최상위 CSV 경로
- `colab/stock_predict_colab.py`의 현재 실행 artifact 반환
- `src/pipeline.py`의 공통 report metadata와 `RunArtifactManager`

5. 남은 작업이 완료되면 이 문서를 갱신하거나 완료 문서로 대체한다.

## 관련 커밋

```text
bf632e1 Protect chatbot runtime secrets
80fd480 Add common report metadata
02da3bf Isolate and promote result runs
c7c6d0f Minimize completed chatbot job state
abbb77b Pin git metadata decoding
```
