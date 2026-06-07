# `result/` 결과물 분석 및 개선 제안

분석 기준일: 2026-06-07

기준 문서: [`RESULT_FILES_GUIDE.md`](RESULT_FILES_GUIDE.md)

## 결론

현재 `result/`는 예측 결과, 실행 리포트, 챗봇 상태, 테스트 임시 파일이 한 디렉터리에 섞여 있다.

파일은 정상 생성되고 있으나 **API 키 평문 노출**, **서로 다른 실행 결과 혼합**, **결과 시점 불일치** 문제가 있어 현재 상태를 운영 결과로 그대로 신뢰하면 안 된다.

가장 먼저 API 키를 폐기·재발급하고, 이후 실행 단위 디렉터리와 결과 검증 절차를 도입해야 한다.

## 현재 현황

| 항목 | 확인 결과 |
|---|---:|
| 전체 파일 수 | 1,989개 |
| 전체 크기 | 약 60.96MB |
| CSV | 723개 |
| JSON | 611개 |
| PNG | 496개 |
| LOG | 122개 |
| 테스트 임시 파일 | 1,709개, 전체 파일의 85.9% |
| 테스트 임시 파일 크기 | 약 46.18MB, 전체 크기의 75.7% |
| 완전 중복 파일 | 1,039개 |
| 중복으로 낭비되는 크기 | 약 11.65MB |

현재 최상위 예측 CSV는 샘플 심볼 `AAA`, `BBB`, `CCC` 3개를 포함한다. 따라서 실제 종목 운영 결과가 아니라 smoke/test 성격의 결과다.

## 우선순위별 문제와 조치

### P0. `chatbot_jobs.json`에 API 키 평문 노출

#### 확인 내용

과거 legacy `result/chatbot_jobs.json`의 `command` 배열에 다음 민감정보가 평문으로 저장되어 있었다.

- DART API 키
- Naver Client ID와 Client Secret
- OpenAI API 키

동일 명령은 콘솔 출력 또는 로그로도 노출될 가능성이 있다. `result/`가 Git에서 제외되어 있어도 로컬 백업, 화면 공유, 압축 파일, 로그 수집 과정에서 유출될 수 있다.

#### 즉시 조치

1. 노출된 OpenAI, DART, Naver 자격증명을 즉시 폐기하고 재발급한다.
2. 기존 `chatbot_jobs.json`, 관련 로그, 외부 백업에서 민감정보를 제거한다.
3. 삭제만으로 끝내지 않는다. 이미 노출된 키는 반드시 교체한다.

#### 코드 수정 제안

- subprocess 인자에 키 값을 넣지 말고 환경변수로 전달한다.
- 작업 상태 저장 전 명령을 마스킹한다.
- `--openai-api-key`, `--dart-api-key`, `--naver-client-secret`, 토큰류의 다음 값을 `[REDACTED]`로 치환한다.
- `_console_log()`에 전체 명령을 출력하지 않거나 마스킹된 명령만 출력한다.
- 비밀값이 JSON과 로그에 없는지 검사하는 회귀 테스트를 추가한다.

#### 완료 기준

- `result/**/*.json`, `result/**/*.log` 비밀 패턴 검사 결과 0건
- 챗봇 작업 실행 기능 정상 동작
- 노출된 기존 키 전부 폐기 완료

---

### P0. 서로 다른 실행의 산출물이 최상위에서 혼합됨

#### 확인 내용

최상위 결과 파일은 항상 같은 이름으로 덮어쓴다.

- `result_detail.csv`
- `result_simple.csv`
- `result_news.csv`
- `result_disclosure.csv`
- `pm_report.json`

반면 파이프라인 리포트와 그래프는 여러 이름으로 누적된다. 현재 최상위 CSV와 `pm_report.json`의 수정 시각은 **2026-06-06 21:57:32**지만, 가장 최신 파이프라인 리포트는 `pipeline_report_smoke.json`의 **2026-06-06 21:48:03**이다. 즉, 현재 CSV와 직접 대응하는 실행 리포트가 없다.

#### 위험

- 사용자가 오래된 리포트와 최신 CSV를 같은 실행 결과로 오해할 수 있다.
- 백테스트, coverage, 예측 결과의 연결 관계를 검증할 수 없다.
- 테스트 실행이 운영 결과를 덮어쓸 수 있다.

#### 수정 제안

실행별 불변 디렉터리를 생성한다.

```text
result/
  latest/
    manifest.json
    result_simple.csv
    result_detail.csv
    result_news.csv
    result_disclosure.csv
    pm_report.json
  runs/
    20260607T120000Z_<run_id>/
      manifest.json
      pipeline_report.json
      pm_report.json
      csv/
      figures/
  runtime/
    chatbot_jobs.json
    chatbot_sessions.json
    prewarm_cache_meta.json
    logs/
  test/
```

- 한 실행의 파일은 같은 `run_id`를 공유한다.
- 모든 파일 생성이 성공한 후에만 `latest/`를 원자적으로 교체한다.
- 테스트는 `result/test/` 또는 OS 임시 디렉터리만 사용한다.
- `manifest.json`에 각 파일의 상대 경로, SHA-256, 행 수, 생성 시각을 기록한다.

#### 완료 기준

- `latest/manifest.json`의 `run_id`가 모든 핵심 산출물과 일치
- 부분 실패 시 기존 `latest/` 유지
- smoke/test 실행이 운영 `latest/`를 변경하지 않음

---

### P0. 가격 데이터와 뉴스·공시 기준일 불일치

#### 확인 내용

- `result_detail.csv`의 가격·예측 기준일: **2023-08-10**
- `result_news.csv`, `result_disclosure.csv`의 기준일: **2026-06-06**

약 3년 차이의 정보를 한 결과 묶음처럼 보여준다. 뉴스와 공시는 표시용이지만, 사용자에게 현재 예측의 관련 정보처럼 오해될 수 있다.

#### 수정 제안

- `input_as_of_date`, `prediction_for_date`, `context_as_of_date`, `generated_at`을 모든 핵심 결과에 명시한다.
- 가격 기준일과 컨텍스트 기준일 차이가 허용 범위를 넘으면 뉴스·공시 요약을 결과에 결합하지 않는다.
- 샘플 데이터 실행은 결과 상단과 파일명에 `environment=smoke` 또는 `data_mode=sample`을 표시한다.
- 챗봇은 샘플 결과를 실제 종목 답변으로 제공하지 못하게 차단한다.

#### 완료 기준

- 운영 결과에서 가격·컨텍스트 기준일 차이가 정책 범위 이내
- 샘플 결과에는 명확한 `sample/smoke` 표시 존재

---

### P1. 파이프라인 리포트 JSON 인코딩 불일치

#### 확인 내용

`pm_report.json`, 챗봇 상태 JSON은 UTF-8이지만 `pipeline_report_*.json`과 `report.json`은 현재 Windows 환경에서 CP949로 기록되어 있다. 원인은 `src/pipeline.py`의 리포트 저장 시 명시적 인코딩이 없는 `write_text()` 사용이다.

#### 위험

- UTF-8을 전제로 하는 CI, Linux, API 소비자가 파일을 읽지 못할 수 있다.
- 동일한 `result/` 내 JSON 파일의 처리 규칙이 달라진다.

#### 수정 제안

- 모든 JSON을 `encoding="utf-8"`로 저장한다.
- 가능하면 기존 `atomic_write_text()` 유틸리티를 사용해 원자적으로 저장한다.
- JSON 읽기 테스트에서 UTF-8 디코딩 성공을 검증한다.

#### 완료 기준

- 모든 `result/**/*.json`이 UTF-8로 디코딩 가능
- 한글 경로와 한글 값 왕복 테스트 통과

---

### P1. 실행 리포트에 식별·시점 메타데이터 부족

#### 확인 내용

현재 실행 리포트와 `pm_report.json`에는 다음 필드가 없다.

- `schema_version`
- `run_id`
- `generated_at`
- `input_as_of_date`

또한 오래된 리포트와 최신 리포트 간 키 구성이 다르다. 일부 파일에는 `config`, `coverage_gate`, `diagnostics`, `pm_summary`가 없다.

#### 수정 제안

공통 메타데이터를 추가한다.

```json
{
  "schema_version": "1.0",
  "run_id": "20260607T120000Z_ab12cd34",
  "environment": "smoke",
  "generated_at": "2026-06-07T12:00:00+09:00",
  "input_as_of_date": "2023-08-10",
  "prediction_for_date": "2023-08-11",
  "git_commit": "abc1234",
  "config_hash": "...",
  "status": "warning"
}
```

- 필수 키를 스키마로 정의한다.
- 과거 스키마를 읽어야 한다면 명시적 migration 또는 호환 로더를 둔다.
- `report.json`처럼 의미가 불명확한 이름은 제거하고 `pipeline_report.json`으로 통일한다.

---

### P1. 현재 백테스트 결과의 해석 가능성 부족

#### 확인 내용

현재 smoke/analysis 리포트는 다음 상태다.

- `tradable_prediction_count`: 0
- 백테스트 누적수익률: 0
- 평균 선택 종목 수: 0
- 유동성 차단 일수: 50일

`pipeline_report_with_context.json`은 투자자 coverage가 0이며 `coverage_gate.status=halt`, 전체 378일이 중단 상태다.

이 결과는 파이프라인 실행 성공 여부는 보여주지만 전략 성과를 평가하는 유효한 백테스트로 보기 어렵다.

#### 수정 제안

- 리포트 최상위에 `status: pass|warning|fail`과 `blocking_reasons`를 추가한다.
- `avg_selected_count == 0`, 전 기간 halt, `tradable_prediction_count == 0`이면 성과 수치를 정상 성과처럼 표시하지 않는다.
- `backtest_valid=false`와 원인을 명시한다.
- smoke 데이터용 유동성 임계값을 별도 설정하거나, smoke 테스트에서는 성과 검증과 산출물 생성 검증을 분리한다.

---

### P1. 확률 calibration 지표 검증 필요

#### 확인 내용

여러 리포트에서 Brier score는 약 `0.247~0.249`인데 ECE는 모두 `0.0`이다. 실제로 완전 보정된 모델일 수도 있지만, 여러 실행에서 반복적으로 정확히 0인 값은 계산 또는 반올림 방식 확인이 필요하다.

#### 수정 제안

- ECE 계산 전 표본 수와 bin별 집계를 리포트에 포함한다.
- 빈 bin 처리와 반올림 전 원본 값을 테스트한다.
- calibration 표본이 부족하면 `null`과 사유를 기록한다.

---

### P2. 테스트 임시 파일과 그래프 중복 누적

#### 확인 내용

테스트 임시 디렉터리 3개가 전체 파일의 85.9%, 전체 크기의 75.7%를 차지한다.

- `result/.pytest_tmp/`
- `result/analysis_pytest_tmp/`
- `result/pr_pytest_tmp/`

완전 중복 파일도 1,039개이며 약 11.65MB가 중복 저장되어 있다.

#### 수정 제안

- 테스트 종료 후 임시 디렉터리를 자동 삭제한다.
- 디버깅이 필요할 때만 `--keep-test-artifacts` 옵션으로 보존한다.
- CI 테스트 산출물은 실패 시에만 업로드한다.
- 운영 그래프는 실행별 보존 기간과 최대 실행 수를 설정한다.
- [`RESULT_CLEANUP.md`](RESULT_CLEANUP.md)에 세 테스트 임시 폴더를 모두 포함한다.

권장 기본 보존 정책:

| 유형 | 보존 정책 |
|---|---|
| `latest/` | 항상 1개 |
| 성공 실행 | 최근 10개 또는 30일 |
| 실패 실행 | 최근 30일 |
| 테스트 임시 파일 | 성공 시 즉시 삭제 |
| 챗봇 로그 | 14일, 비밀값 마스킹 필수 |

---

### P2. 챗봇 상태·로그의 개인정보 및 운영정보 관리

#### 확인 내용

- `chatbot_sessions.json`은 사용자 식별 해시와 마지막 조회 정보를 저장한다.
- `chatbot_jobs.json`은 실행 명령, 로컬 Python 경로, PID, 작업 이력을 저장한다.
- 실패 작업과 오래된 완료 작업이 계속 남아 있다.

#### 수정 제안

- 세션과 작업 레지스트리에 TTL을 적용한다.
- 사용자 해시도 개인정보성 운영 데이터로 취급한다.
- 완료 작업은 필요한 최소 필드만 남기고 상세 명령은 제거한다.
- 실패 로그에는 오류 요약과 추적 ID만 상태 파일에 기록한다.
- 로컬 절대 경로 대신 프로젝트 상대 경로를 사용한다.

---

### P2. 뉴스·공시 placeholder 행 구분 부족

#### 확인 내용

현재 뉴스와 공시 CSV는 실제 원문 이벤트가 아니라 “수집된 뉴스가 없음” 등의 요약 placeholder 행 3개를 포함한다. `provider=summary`이지만 실제 이벤트와 같은 CSV 구조에 저장되어 있어 집계 시 기사 또는 공시 건수로 오인할 수 있다.

#### 수정 제안

- `record_type=event|summary|no_data` 필드를 추가한다.
- 실제 이벤트 수와 placeholder 수를 분리한다.
- `no_data_reason`, `collection_status`, `collection_error`를 기록한다.
- 뉴스·공시는 계속 표시·검토용으로만 유지하며 `predicted_return`, 순위, 권고에 영향을 주지 않도록 회귀 테스트를 유지한다.

## 권장 구현 순서

### 1단계: 즉시 보안 조치

1. 노출된 API 키 전부 폐기·재발급
2. 명령 저장 및 로그 출력 시 비밀값 마스킹
3. 비밀정보 회귀 테스트 추가

### 2단계: 결과 신뢰성 확보

1. `run_id`, `generated_at`, 기준일, 환경 구분 추가
2. 실행별 디렉터리와 `manifest.json` 도입
3. 동일 실행 결과만 `latest/`로 원자적 승격
4. 가격·뉴스·공시 기준일 검증 추가

### 3단계: 포맷 표준화

1. 모든 JSON UTF-8 및 원자적 저장
2. 실행 리포트 스키마 버전 도입
3. 유효하지 않은 백테스트 상태와 blocking reason 표시

### 4단계: 정리 자동화

1. 테스트 임시 파일 자동 삭제
2. 실행·로그 보존 정책 적용
3. placeholder 이벤트 명시

## 권장 테스트 추가

| 테스트 | 검증 내용 |
|---|---|
| `test_job_registry_redacts_secrets` | 작업 상태 JSON과 로그에 비밀값 미포함 |
| `test_pipeline_report_is_utf8` | 실행 리포트 UTF-8 디코딩 |
| `test_artifacts_share_run_id` | 핵심 결과의 `run_id` 일치 |
| `test_latest_promoted_only_after_success` | 부분 실패 시 이전 latest 보존 |
| `test_context_date_matches_prediction_policy` | 가격·뉴스·공시 기준일 정책 검증 |
| `test_smoke_output_cannot_replace_production_latest` | 테스트 결과와 운영 결과 격리 |
| `test_invalid_backtest_reports_blocking_reason` | 거래 불가 백테스트 명시 |
| `test_news_context_never_changes_signal` | 뉴스·공시가 예상수익률·권고에 영향 없음 |

## 최종 권장 상태

운영자가 `result/latest/manifest.json` 하나만 열어도 다음을 판단할 수 있어야 한다.

- 어떤 입력과 설정으로 실행했는가
- 언제 생성했고 어느 날짜를 예측하는가
- 샘플 결과인가 실제 운영 결과인가
- coverage와 백테스트가 유효한가
- 모든 CSV, JSON, 그래프가 같은 실행에서 생성되었는가
- 민감정보가 포함되지 않았는가

현재 구현은 위 조건을 코드와 회귀 테스트로 검증한다.

## 구현 상태 (2026-06-07)

| 우선순위 | 상태 | 구현·검증 |
|---|---|---|
| P0 비밀정보 마스킹 | 완료 | `src/utils/secrets.py`, `tests/test_secret_redaction.py` |
| P0 실행별 산출물/latest 격리 | 완료 | `src/reports/run_artifacts.py`, `tests/test_run_artifacts.py` |
| P0 sample/smoke 운영 추천 차단 | 완료 | `src/chatbot/kakao_colab_bot.py`, `tests/test_kakao_colab_bot.py` |
| P1 날짜/context 정책 | 완료 | `src/reports/context_policy.py`, `tests/test_context_policy.py` |
| P1 백테스트/calibration 유효성 | 완료 | `src/validation/result_validity.py`, `tests/test_backtest_and_calibration.py` |
| P1 뉴스·공시 명시적 record type | 완료 | `src/pipeline.py`, `tests/test_news_impact_context.py` |
| P2 runtime TTL/보존/안전 정리 | 완료 | `src/utils/result_cleanup.py`, `tests/test_result_cleanup.py` |

이미 로그나 백업에 노출된 API 키는 코드 변경으로 폐기되지 않는다. 운영자가 OpenAI,
DART, Naver 자격증명을 직접 폐기하고 재발급해야 한다.
