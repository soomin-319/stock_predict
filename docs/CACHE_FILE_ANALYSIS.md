# 캐시 파일 필요성 분석

기준 문서: `docs/CACHE_FILE_CREATION.md`  
분석 기준: 현재 repository 코드 검색 및 실제 생성/사용 흐름

## 결론

대부분의 캐시는 현재 구조에서 필요하다. 다만 일부는 미사용 코드이거나, 무효화 조건이 과하고, 장기 보관 정책이 없어 정리가 필요하다.

## 필요한 캐시

| 항목 | 판단 | 이유 |
| --- | --- | --- |
| `result/result_simple.csv` | 필요 | Kakao 챗봇이 바로 읽는 핵심 예측 결과 디스크 캐시다. 없으면 종목 조회마다 파이프라인 재실행 위험이 있다. |
| `result/result_detail.csv` | 필요 | 상세 예측 결과와 피처/정책 컨텍스트 저장용이다. 챗봇은 여기서 최신 예측 날짜도 보정한다. |
| `_result_simple_cache` | 필요 | 챗봇 프로세스 내부 메모리 캐시다. `mtime_ns`로 CSV 변경을 감지하므로 반복 `read_csv()` 비용을 줄인다. |
| `result/prewarm_cache_meta.json` | 필요 | prewarm 결과 재사용 여부를 signature hash로 판단한다. 비싼 기본 예측 재실행을 줄인다. |
| `_live_events_cache` | 필요 | Naver/DART 라이브 원문 조회 결과를 15분 TTL로 저장한다. API 호출 비용과 timeout 위험을 줄인다. |
| `_issue_summary_cache` | 조건부 필요 | 종목별 뉴스/공시 요약 결과를 메모리에 저장한다. OpenAI 요약 timeout 이후 재조회 UX를 개선한다. 프로세스 종료 시 사라지는 현재 방식은 적절하다. |
| `result/chatbot_jobs.json` | 필요 | 백그라운드 예측/부트스트랩 작업 상태 추적에 필요하다. |
| `result/chatbot_sessions.json` | 조건부 필요 | 사용자가 `결과`, `최신화`처럼 종목을 생략했을 때 마지막 종목을 찾는 데 필요하다. |
| `result/.pytest_cache` | 개발용 필요 | 런타임에는 필요 없지만 pytest 속도 개선용으로 적절하다. `result/` 아래라 git ignore 정책과 맞다. |

## 불필요하거나 정리 후보인 부분

| 항목 | 판단 | 이유 |
| --- | --- | --- |
| `src/news_impact/data_cache.py::DataCache` | 현재 미사용 | 클래스와 메서드는 있지만 `src/`, `tests/`에서 직접 호출 지점이 없다. 실제 사용할 계획이 없으면 제거 후보이고, 사용할 계획이 있으면 문서와 테스트를 보강해야 한다. |
| `prewarm signature`의 `report_json`, `figure_dir` | 과한 무효화 가능 | 챗봇 응답 캐시 내용에는 직접 영향이 작다. 파일명이나 그림 폴더만 바뀌어도 prewarm 캐시가 무효화될 수 있다. |
| 오래된 `chatbot_jobs.json` 완료 이력 | pruning 필요 | 완료/실패 이력이 계속 누적될 수 있다. 최근 N개 또는 최근 N일만 보관하는 정책이 필요하다. |
| `result/chatbot_logs/*.log` | 보존 | 예측 subprocess, bootstrap, 실시간 추천 실행 이력 확인에 필요하므로 별도 pruning 대상에서 제외한다. |
| `chatbot_sessions.json` 영구 저장 | privacy 정책 필요 | 사용자 ID와 마지막 종목이 저장된다. TTL, 삭제, 익명화 정책을 정하는 것이 좋다. |

## 문서 누락 사항

`CACHE_FILE_CREATION.md`는 `DataCache`는 언급하지만, 실제 사용 중인 LLM 응답 파일 캐시를 빠뜨리고 있다.

실제 캐시:

- `src/news_impact/llm_client.py`
  - `FileLLMResponseCache`
- `src/news_impact/pipeline.py`
  - `{output_dir}/llm_cache/impact_judgments/*.json`
  - `{output_dir}/llm_cache/semantic_clusters/*.json`

이 캐시는 실제로 `run_daily_pipeline()`에서 생성될 수 있다. 따라서 `CACHE_FILE_CREATION.md`에 별도 섹션으로 추가하는 것이 좋다.

## 위험 및 개선 포인트

### 1. `result_simple.csv` fallback 문제

`safe_to_csv()`는 `PermissionError`가 나면 다음처럼 fallback 파일을 만든다.

```text
result/result_simple.csv
-> result/result_simple_fallback.csv
```

하지만 챗봇은 기본적으로 `result/result_simple.csv`만 읽는다.  
따라서 fallback이 생성되면 챗봇은 stale cache를 계속 볼 수 있다.

권장 개선:

- 챗봇용 핵심 파일은 fallback 저장 대신 실패로 처리한다.
- 또는 챗봇이 fallback/latest artifact를 인식하도록 한다.
- 또는 atomic write 후 replace 방식으로 파일 잠금 문제를 줄인다.

### 2. schema 불일치 CSV도 캐시에 저장됨

`_load_cached_result_simple()`은 필수 컬럼이 누락되어도 로그만 남기고 loaded DataFrame을 메모리에 저장한 뒤 반환한다.

권장 개선:

- 필수 schema 불일치 시 빈 DataFrame 반환
- 또는 재생성 job 유도
- 또는 사용자 응답에 “캐시 손상/스키마 불일치” 메시지 표시

### 3. `prewarm_cache_meta.json` write가 atomic이 아님

`chatbot_jobs.json`, `chatbot_sessions.json`은 임시 파일 후 `replace()` 방식이다.  
반면 `prewarm_cache_meta.json`은 직접 `write_text()`로 저장한다.

권장 개선:

- `prewarm_cache_meta.json.tmp`에 먼저 쓴 뒤 `replace()`로 교체한다.

### 4. LLM 캐시에 retention 정책 없음

`FileLLMResponseCache`는 output directory 아래 JSON을 계속 쌓는다.  
`operations.py`에는 `cache_retention_days` 개념이 있으나, 이 캐시에 직접 적용되는 정리 로직은 확인되지 않는다.

권장 개선:

- output directory별 캐시 보관 기간 설정
- 오래된 `llm_cache` JSON 삭제 도구 또는 명령 추가

## 권장 우선순위

1. `CACHE_FILE_CREATION.md`에 `FileLLMResponseCache`/`llm_cache` 설명 추가.
2. `DataCache`를 실제로 사용할지 제거할지 결정.
3. `chatbot_jobs`, `chatbot_sessions`에 TTL/pruning 정책 추가. `chatbot_logs`는 실행 기록 보존 목적으로 유지한다.
4. `result_simple_fallback.csv`가 챗봇 stale cache를 만들 수 있는 문제 해결.
5. `prewarm_cache_meta.json` 저장을 atomic write로 변경.
6. `prewarm signature`에서 `report_json`, `figure_dir`를 제외할지 검토.

## 요약

- 예측 결과 캐시와 챗봇 메모리 캐시는 필요하다.
- 라이브 뉴스/공시 캐시는 API 비용과 timeout 방지를 위해 필요하다.
- 미사용 `DataCache`는 정리 후보이다.
- 실제 LLM 캐시가 문서에 누락되어 있다.
- 장기적으로는 캐시 보관 기간, fallback 처리, atomic write 정책을 정리하는 것이 좋다.
