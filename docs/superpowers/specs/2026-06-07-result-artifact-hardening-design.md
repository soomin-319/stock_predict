# Result Artifact Hardening Design

## Goal

`result/` 산출물을 실행 단위로 격리하고 검증된 운영 결과만 최신 결과로 승격한다. 동시에 비밀정보 노출, 기준일 불일치, 불명확한 백테스트 상태, 테스트 산출물 누적 문제를 해결한다.

뉴스와 공시는 표시·검토용 컨텍스트로만 유지하며 `predicted_return`, 기대수익률 순위, 추천, 자동 신호 결정에 영향을 주지 않는다.

## Scope

이 설계는 `RESULT_ANALYSIS_AND_IMPROVEMENTS.md`의 P0, P1, P2 개선안을 모두 포함한다.

- 작업 명령과 로그의 비밀정보 제거
- 실행별 불변 산출물과 원자적 최신 결과 승격
- 기존 최상위 결과 경로의 단계적 호환 유지
- 가격·컨텍스트 기준일 정책
- 공통 리포트 메타데이터와 UTF-8 원자 저장
- 백테스트·calibration 유효성 표시
- 테스트 산출물, runtime 상태, 운영 run 보존 정책
- 뉴스·공시 placeholder의 명시적 구분

실제 노출된 API 키의 폐기·재발급은 코드로 수행할 수 없으므로 운영자가 별도로 수행한다.

## Result Directory Layout

```text
result/
  latest/
    manifest.json
    pipeline_report.json
    pm_report.json
    csv/
      result_simple.csv
      result_detail.csv
      result_news.csv
      result_disclosure.csv
    figures/
  runs/
    <run_id>/
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
  result_simple.csv
  result_detail.csv
  result_news.csv
  result_disclosure.csv
  pm_report.json
```

`runs/<run_id>/`는 한 실행의 불변 산출물이다. `latest/`는 검증에 통과한 최신 운영 실행의 완전한 복사본이다. 최상위 CSV와 `pm_report.json`은 단계적 전환을 위한 호환 복사본이다.

## Run Lifecycle

### Run identity

파이프라인 시작 시 한 번만 `run_id`와 공통 메타데이터를 생성한다.

```text
run_id = <UTC timestamp>_<short random identifier>
```

예: `20260607T120000Z_ab12cd34`

모든 리포트와 manifest는 동일한 `run_id`를 공유한다.

### Staging and finalization

1. 실행 산출물을 `runs/<run_id>/` 내부에 기록한다.
2. 필수 파일, JSON 스키마, 기준일 정책, 파일 해시를 검증한다.
3. 검증 결과를 `manifest.json`에 기록한다.
4. 운영 실행이며 필수 검증에 통과한 경우에만 `latest/`를 원자적으로 교체한다.
5. `latest/` 승격 성공 후 최상위 호환 복사본을 갱신한다.

부분 실패, 예외, 검증 실패 시 기존 `latest/`와 최상위 호환 복사본을 보존한다.

### Environment isolation

실행은 `production`, `smoke`, `test` 환경으로 구분한다.

- `production`: 검증 통과 시 `latest/`와 호환 복사본 승격 가능
- `smoke`: `runs/` 또는 `test/`에 기록하지만 운영 최신 결과 승격 금지
- `test`: `result/test/` 또는 pytest 임시 디렉터리만 사용

샘플 심볼이나 샘플 입력을 사용하면 `data_mode=sample`로 기록한다. `sample` 결과는 환경값과 무관하게 운영 최신 결과로 승격하거나 실제 종목 챗봇 추천에 사용할 수 없다.

## Manifest

`manifest.json`은 다음 필드를 포함한다.

```json
{
  "schema_version": "1.0",
  "run_id": "20260607T120000Z_ab12cd34",
  "environment": "production",
  "data_mode": "real",
  "generated_at": "2026-06-07T12:00:00+09:00",
  "input_as_of_date": "2026-06-05",
  "prediction_for_date": "2026-06-08",
  "context_as_of_date": "2026-06-07",
  "git_commit": "abc1234",
  "config_hash": "<sha256>",
  "status": "pass",
  "blocking_reasons": [],
  "artifacts": []
}
```

각 artifact 항목은 실행 디렉터리 기준 상대 경로, SHA-256, 바이트 크기, 생성 시각을 포함한다. CSV는 행 수도 포함한다.

`status`는 `pass`, `warning`, `fail` 중 하나다. 최신 결과 승격은 `pass` 또는 운영 정책상 허용된 `warning`에만 가능하다. `fail`은 승격할 수 없다.

## Common Report Metadata

`pipeline_report.json`과 `pm_report.json`은 manifest와 동일한 공통 메타데이터를 가진다.

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

모든 JSON은 UTF-8로 원자 저장한다. 의미가 불명확한 신규 `report.json`은 생성하지 않고 `pipeline_report.json`을 사용한다. 기존 사용자 지정 `--report-json` 이름은 호환성을 위해 계속 허용한다.

## Compatibility Strategy

챗봇과 소비 코드는 다음 순서로 결과를 찾는다.

1. 검증된 `result/latest/manifest.json`과 그 artifact 경로
2. 기존 최상위 `result_simple.csv`, `result_detail.csv`, `pm_report.json`

최상위 호환 복사본은 `latest/` 승격 후에만 갱신한다. 이 규칙으로 기존 스크립트를 깨뜨리지 않으면서 서로 다른 실행의 산출물이 혼합되는 것을 방지한다.

## Secret Handling

### Command execution

API 키와 토큰은 subprocess 명령 인자에 넣지 않고 자식 프로세스 환경변수로 전달한다. 명령 인자 호환이 반드시 필요한 외부 호출도 저장 또는 출력 전 공통 redactor를 거친다.

공통 redactor는 최소한 다음을 처리한다.

- `--openai-api-key`
- `--dart-api-key`
- `--naver-client-id`
- `--naver-client-secret`
- 이름에 `token`, `secret`, `password`, `api-key`, `api_key`가 포함된 인자
- 등록된 실제 비밀값의 문자열 출현

마스킹 값은 `[REDACTED]`로 통일한다.

### Runtime persistence and logs

`chatbot_jobs.json`에는 마스킹된 명령만 저장한다. 콘솔과 파일 로그에도 마스킹된 문자열만 기록한다. 완료 작업은 상세 명령 대신 작업 ID, 상태, 시간, 종료 코드, 오류 요약만 보존한다.

기존 상태 파일을 읽을 때는 저장 전에 마스킹하여 마이그레이션한다. 실제 노출 자격증명은 운영자가 폐기·재발급해야 한다.

## Date and Context Policy

핵심 결과는 다음 시점을 명시한다.

- `input_as_of_date`: 가격·특징 입력의 최종 기준일
- `prediction_for_date`: 예측 대상 거래일
- `context_as_of_date`: 뉴스·공시 컨텍스트 기준일
- `generated_at`: 산출물 생성 시각

컨텍스트 기준일과 가격 기준일의 차이가 설정된 허용 범위를 초과하면 뉴스·공시를 해당 예측 결과에 결합하지 않는다. 대신 수집 상태와 제외 사유를 기록한다.

샘플 또는 smoke 결과는 CSV와 JSON에 명확히 표시한다. 챗봇은 이를 실제 종목 운영 추천으로 제공하지 않는다.

## Backtest and Calibration Validity

파이프라인 리포트는 백테스트 수치와 별도로 유효성을 기록한다.

```json
{
  "backtest_valid": false,
  "blocking_reasons": ["tradable_prediction_count_zero"]
}
```

다음 조건은 최소한 유효하지 않은 백테스트로 처리한다.

- `tradable_prediction_count == 0`
- 평가 기간 전체가 coverage halt
- `avg_selected_count == 0`

유효하지 않은 백테스트의 0 수익률을 정상 성과처럼 표시하지 않는다.

Calibration은 표본 수와 bin별 집계를 기록한다. 표본이 정책상 부족하면 ECE를 `null`로 기록하고 이유를 포함한다. 충분한 표본에서는 빈 bin 처리와 반올림 전 원본 값이 테스트 가능해야 한다.

## News and Disclosure Records

뉴스와 공시 출력은 `record_type`을 포함한다.

- `event`: 실제 수집 이벤트
- `summary`: 여러 이벤트 또는 수집 결과의 요약
- `no_data`: 이벤트 없음 또는 수집 불가

추가 필드:

- `collection_status`
- `no_data_reason`
- `collection_error`

뉴스와 공시는 계속 표시·검토 전용이다. 해당 필드를 추가하거나 컨텍스트를 제외해도 `predicted_return`, 순위, 추천, 자동 신호가 바뀌지 않아야 한다.

## Runtime State and Retention

챗봇 상태와 로그는 `result/runtime/`에 저장한다. 기존 최상위 runtime 파일은 읽기 fallback을 제공하고, 다음 저장 시 새 위치로 이동한다.

작업과 세션에는 TTL을 적용한다. 만료된 상태는 안전한 저장 시점에 제거한다. 완료 작업은 최소 메타데이터만 남긴다.

기본 보존 정책:

- `latest/`: 항상 보존
- 성공 운영 run: 최근 10개 또는 최근 30일
- 실패 run: 최근 30일
- 성공한 테스트 임시 산출물: 테스트 종료 시 제거
- runtime 로그: 14일, 항상 비밀값 마스킹

cleanup은 `result/runs/`, `result/test/`, `result/runtime/logs/`의 허용된 하위 경로만 삭제할 수 있다. `latest/`, runtime 상태 JSON, 프로젝트 외부 경로는 삭제하지 않는다. 테스트 산출물은 명시적 보존 설정이 있을 때만 남긴다.

## Error Handling

- 산출물 쓰기는 UTF-8 원자 저장 또는 임시 디렉터리 후 rename을 사용한다.
- artifact 생성 실패는 run을 `fail`로 마감하고 승격하지 않는다.
- manifest 생성 또는 해시 검증 실패도 승격을 차단한다.
- `latest/` 교체 실패 시 기존 최신 결과를 복구 또는 유지한다.
- 호환 복사본 갱신 실패는 경고로 기록하되 검증된 `latest/`는 유지한다.
- runtime 상태 정리 실패는 챗봇 요청을 중단하지 않고 마스킹된 경고를 남긴다.

## Components

### `src/reports/run_artifacts.py`

실행 ID, 디렉터리 레이아웃, 공통 메타데이터, manifest 생성, 해시 계산, 검증, latest 승격, 호환 복사를 담당한다.

### `src/utils/secrets.py`

명령 인자, 환경값, 로그 문자열의 공통 비밀정보 마스킹을 담당한다.

### `src/reports/report_metadata.py`

공통 리포트 메타데이터와 상태·blocking reason 구성을 담당한다.

### `src/reports/context_policy.py`

가격과 컨텍스트 기준일의 허용 여부를 판단한다.

### `src/validation/result_validity.py`

백테스트와 calibration 결과의 유효성 상태를 계산한다.

### `src/utils/result_cleanup.py`

허용 경로와 보존 정책에 따라 run, 테스트 산출물, 로그를 정리한다.

기존 pipeline, reports, chatbot 모듈은 이 컴포넌트를 호출하고 도메인별 계산 책임은 유지한다.

## Testing Strategy

모든 동작 변경은 실패하는 테스트를 먼저 작성한 뒤 최소 구현으로 통과시킨다.

필수 회귀 테스트:

- 작업 상태 JSON과 로그가 비밀값을 포함하지 않는다.
- pipeline과 PM 리포트가 UTF-8로 한글을 왕복한다.
- 한 실행의 manifest와 핵심 리포트가 동일한 `run_id`를 가진다.
- 부분 실패 시 기존 `latest/`가 유지된다.
- smoke, test, sample 실행은 운영 `latest/`를 변경하지 않는다.
- latest 승격 후에만 최상위 호환 복사본이 갱신된다.
- 날짜 정책 초과 컨텍스트는 결합되지 않는다.
- sample 결과는 챗봇 운영 추천에 사용되지 않는다.
- 유효하지 않은 백테스트는 blocking reason을 기록한다.
- calibration 표본 부족 시 ECE가 `null`과 사유를 가진다.
- 뉴스·공시 컨텍스트는 신호, 순위, `predicted_return`을 바꾸지 않는다.
- runtime TTL과 cleanup은 허용 경로 밖을 삭제하지 않는다.
- 뉴스·공시 placeholder는 `record_type`으로 구분된다.

구현 완료 시 영향 테스트, 전체 `pytest`, bundled sample smoke pipeline을 실행한다.

## Delivery

전체 개선안은 하나의 PR로 전달하되 내부 구현은 독립적인 작은 커밋으로 나눈다.

1. 비밀정보 마스킹과 runtime 상태 보안
2. 공통 메타데이터와 UTF-8 저장
3. 실행별 산출물, manifest, latest 승격, 호환 복사
4. 날짜·sample 정책과 챗봇 차단
5. 백테스트·calibration 유효성
6. 뉴스·공시 record type
7. TTL, retention, cleanup
8. 전체 검증과 문서 갱신

