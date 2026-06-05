# 캐시 파일 필요성 재분석

분석일: 2026-06-05  
분석 범위: 현재 저장소 코드(`src/`, `tests/`, `docs/`) 기준  
주의: 이전 문서가 기준으로 삼은 `docs/CACHE_FILE_CREATION.md`는 현재 저장소에 없다. 따라서 이 문서는 실제 코드 검색 결과를 기준으로 다시 정리한다.

## 결론

현재 캐시는 대부분 성능·운영 안정성 때문에 필요하다. 특히 Kakao/Colab 봇은 `result/result_simple.csv`를 즉시 응답용 캐시로 사용하므로, 캐시가 없으면 매 요청마다 예측 파이프라인을 다시 실행할 수 있다.

정리가 필요한 지점은 세 가지다.

1. `result_simple.csv` 저장 실패 시 `_fallback` 파일로 저장되지만 챗봇은 기본 파일만 읽어 stale 결과를 볼 수 있다.
2. `prewarm_cache_meta.json`, LLM 응답 캐시, `DataCache`는 atomic write가 아니다.
3. `chatbot_jobs.json`, `chatbot_sessions.json`, `llm_cache`, `chatbot_logs`는 보관 기간과 정리 정책이 명확하지 않다.

뉴스·공시·LLM 요약 캐시는 표시/검토용 컨텍스트다. `predicted_return`, 기대수익률 순위, 매수/매도/관망 결정에는 쓰면 안 된다.

## 캐시·산출물 현황

| 항목 | 위치/구현 | 현재 사용 | 판단 | 근거 |
| --- | --- | --- | --- | --- |
| 예측 요약 CSV | `result/result_simple.csv` | 파이프라인 저장, Kakao 봇 즉시 응답 | 필수 | `_load_cached_result_simple()`가 읽고 `_find_cached_prediction()`이 종목별 캐시 hit에 사용한다. |
| 예측 상세 CSV | `result/result_detail.csv` | 파이프라인 저장, 날짜 보정/상세 컨텍스트 | 필요 | 챗봇이 종목별 최신 예측 기준일을 보정할 때 사용한다. |
| 뉴스/공시 CSV | `result/result_news.csv`, `result/result_disclosure.csv` | 이슈 요약 원문 재사용 | 필요 | `_load_result_news()`가 기존 원문을 읽고 부족할 때만 live fetch를 시도한다. 표시용 컨텍스트다. |
| 파이프라인 리포트/그림 | `pipeline_report.json`, `pm_report.json`, figure dirs | 실행 산출물 | 필요 | 연구/운영 검토용 artifact. 캐시라기보다 결과물이다. |
| 메모리 예측 캐시 | `KakaoColabPredictionBot._result_simple_cache` | 프로세스 내 CSV 재사용 | 필요 | `mtime_ns`가 같으면 `read_csv()` 반복 비용을 줄인다. |
| 메모리 이슈 요약 캐시 | `_issue_summary_cache` | 프로세스 내 요약 재사용 | 필요 | timeout 뒤 백그라운드 요약 결과를 이후 응답에 붙인다. `result_simple.csv` mtime 변경 시 초기화된다. |
| live events 메모리 캐시 | `_live_events_cache` | Naver/DART 원문 조회 15분 TTL | 필요 | 같은 종목/날짜 반복 API 호출과 timeout을 줄인다. |
| prewarm 메타 | `result/prewarm_cache_meta.json` | prewarm 재사용 signature | 필요 | 입력 파일, 기본 유니버스, 주요 runtime 옵션, KST 날짜가 같으면 기본 예측 재실행을 건너뛴다. |
| 작업 상태 | `result/chatbot_jobs.json` | 예측/부트스트랩 job 상태 | 필수 | running/completed/failed, log path, pid 추적에 필요하다. |
| 사용자 세션 | `result/chatbot_sessions.json` | 마지막 종목/intent 저장 | 조건부 필요 | 사용자가 `결과`, `최신화`처럼 종목을 생략할 때 필요하다. 단 privacy 정책 필요. |
| 챗봇 로그 | `result/chatbot_logs/*.log` | subprocess/bootstrap/recommendation 기록 | 필요 | 장애 분석용. 최근 추가된 `recommendation_*.log`도 성공/실패와 응답 텍스트를 남긴다. |
| LLM 응답 파일 캐시 | `{output_dir}/llm_cache/impact_judgments/*.json`, `{output_dir}/llm_cache/semantic_clusters/*.json` | news-impact LLM 호출 재사용 | 필요 | `FileLLMResponseCache`가 prompt payload hash로 응답을 저장한다. 비용·시간 절감 효과가 크다. |
| vendored 데이터 캐시 | `src/news_impact/data_cache.py::DataCache` | 직접 호출 없음 | 정리 후보 | 클래스는 있으나 현재 `src/`, `tests/`에서 사용처가 확인되지 않는다. |
| pytest 캐시 | `result/.pytest_cache` | 테스트 개발 캐시 | 개발용 필요 | `pyproject.toml`에 명시. 런타임 캐시는 아니다. |

## 핵심 흐름

### 1. 예측 결과 캐시

`src/pipeline.py`는 실행 후 `result_detail.csv`, `result_simple.csv`, `result_news.csv`, `result_disclosure.csv`를 `utf-8-sig`로 저장한다. `src/reports/output.py::safe_to_csv()`는 Windows에서 파일이 Excel 등에 열려 있으면 `_fallback` 파일명으로 저장한다.

Kakao 봇은 기본적으로 `result/result_simple.csv`만 읽는다. 파일이 있으면 즉시 응답하고, 없거나 요청 종목이 없으면 예측 job을 시작한다.

### 2. prewarm 캐시

`prewarm_prediction_cache()`는 `result/result_simple.csv`와 `result/prewarm_cache_meta.json`의 `signature_hash`를 함께 본다. signature에는 KST 날짜, 입력 CSV stat, 기본 유니버스 stat, report/figure 경로, 외부 데이터/투자자 컨텍스트/OpenAI/Naver/DART 설정 여부가 들어간다.

`report_json`, `figure_dir`가 signature에 들어가는 것은 보수적이다. 이 둘을 제외하면 다른 artifact 경로를 요청해도 cache hit로 인해 새 report/figure가 생성되지 않을 수 있다. 따라서 단순히 “과한 무효화”로 보기 어렵다.

### 3. 이슈 요약과 live events 캐시

챗봇은 먼저 `result_news.csv`/`result_disclosure.csv`를 읽고, 당일 원문이 없으면 Naver/DART live fetch를 시도한다. live fetch 결과는 15분 TTL 메모리 캐시에 저장된다. 요약이 오래 걸리면 현재 응답은 기존 row로 보내고, 백그라운드에서 요약 후 `_issue_summary_cache`에 저장한다.

이 캐시는 사용자 표시 품질을 높이기 위한 것이다. 추천/순위/기대수익률 산정에는 영향을 주면 안 된다.

### 4. news-impact LLM 캐시

`FileLLMResponseCache`는 LLM 요청 payload와 required key를 hash해 JSON 응답을 저장한다. 실제 생성 위치는 다음이다.

- `output_dir/llm_cache/impact_judgments/*.json`
- `output_dir/llm_cache/semantic_clusters/*.json`

현재 구현은 파일을 직접 `write_text()`로 저장한다. crash 중 partial write 가능성이 있고, 보관 기간 정리 로직도 없다.

## 문제점과 개선안

### P0. `result_simple_fallback.csv` stale 문제

문제:

```text
result/result_simple.csv -> result/result_simple_fallback.csv
```

`safe_to_csv()`가 fallback에 저장하면 파이프라인 report에는 fallback 경로가 기록된다. 하지만 Kakao 봇은 기본 경로인 `result/result_simple.csv`만 읽는다. 사용자는 새 예측이 끝났다고 생각해도 봇은 이전 CSV를 계속 볼 수 있다.

개선안:

- 챗봇 핵심 artifact(`result_simple.csv`, 가능하면 `result_detail.csv`)는 fallback 대신 실패로 처리한다.
- 또는 `latest_artifacts.json` 같은 manifest를 만들고 봇이 실제 저장 경로를 읽게 한다.
- 또는 atomic temp write 후 replace를 기본으로 바꿔 파일 잠금 문제를 줄인다.

### P0. 스키마 불일치 CSV 처리

문제:

`_load_cached_result_simple()`은 필수 컬럼 누락 시 로그를 남긴 뒤에도 DataFrame을 메모리 캐시에 저장하고 반환한다. 이후 경로에서 빈 결과/포맷 오류로 우회될 수 있지만, “손상된 캐시”로 명확히 다루지는 않는다.

개선안:

- 필수 컬럼 누락 시 `pd.DataFrame()` 반환.
- job 재생성 또는 사용자 안내 메시지로 “예측 캐시 스키마 불일치” 표시.
- 스키마 버전 컬럼 또는 sidecar metadata 추가 검토.

### P1. atomic write 통일

이미 atomic에 가까운 구현:

- `chatbot_jobs.json`
- `chatbot_sessions.json`

직접 write라 개선 필요한 구현:

- `prewarm_cache_meta.json`
- `FileLLMResponseCache` JSON
- `DataCache` JSON
- `result/*.csv` 전체

개선안:

- 같은 directory에 `.tmp`로 쓰고 `replace()`.
- CSV는 temp 파일도 `utf-8-sig` 유지.
- 동시 실행 가능성이 있는 job/meta 파일은 file lock까지 검토.

### P1. 보관 기간과 pruning 정책

현재 누적 가능 항목:

- `result/chatbot_jobs.json` 완료/실패 이력
- `result/chatbot_sessions.json` 사용자별 마지막 종목
- `result/chatbot_logs/*.log`
- `result/**/llm_cache/**/*.json`
- figure/report 산출물

권장 기본 정책:

| 항목 | 권장 보관 | 이유 |
| --- | --- | --- |
| `chatbot_jobs.json` | 최근 100개 또는 30일 | 운영 상태만 필요. 무기한 보관 불필요. |
| `chatbot_sessions.json` | 30~90일 TTL | 사용자 ID/마지막 종목 저장. privacy 고려 필요. |
| `chatbot_logs/*.log` | 30일 또는 용량 기준 | 장애 분석용. 무기한 누적 방지. |
| `llm_cache/**/*.json` | provider policy의 `cache_retention_days`와 연결 | 비용 절감과 약관/저장정책 균형. |
| `result_simple/detail/news/disclosure.csv` | 최신 운영본 유지 | 봇 즉시 응답에 필요. 백업은 별도 경로. |

### P1. 없는 기준 문서 정리

현재 `CACHE_FILE_ANALYSIS.md`만 있고 `CACHE_FILE_CREATION.md`는 없다. 문서 참조가 깨져 있었다.

개선안:

- 이 문서를 기준 문서로 삼거나,
- 별도 `CACHE_FILE_CREATION.md`를 새로 만들고 캐시 생성 위치/형식/보관정책을 분리한다.

### P2. `DataCache` 사용 여부 결정

`src/news_impact/data_cache.py::DataCache`는 news/disclosure/market cache와 manifest 기능을 제공하지만 현재 호출 지점이 없다.

선택지:

1. 사용할 계획이면 collectors/pipeline에 연결하고 테스트 추가.
2. vendored 호환용으로 보존할 계획이면 “현재 미사용, 호환 목적” 주석/문서 추가.
3. 계획이 없으면 제거 후보로 둔다.

## 권장 실행 순서

1. `result_simple.csv` fallback stale 문제 해결.
2. `_load_cached_result_simple()` 스키마 불일치 시 빈 캐시/재생성 경로로 변경.
3. `prewarm_cache_meta.json`과 LLM cache write를 atomic으로 변경.
4. `chatbot_jobs.json`, `chatbot_sessions.json`, `chatbot_logs`, `llm_cache` retention 정책 추가.
5. `DataCache`를 실제 사용/호환 보존/제거 중 하나로 결정.
6. 캐시 생성·무효화·보관 정책을 `docs/OPERATIONS.md` 또는 별도 cache 문서에 연결.

## 최종 판단

- 삭제하면 안 되는 핵심 캐시: `result_simple.csv`, `result_detail.csv`, `prewarm_cache_meta.json`, `chatbot_jobs.json`, LLM cache.
- 조건부로 필요한 캐시: `chatbot_sessions.json`, `_issue_summary_cache`, `_live_events_cache`, `chatbot_logs`.
- 정리 후보: 미사용 `DataCache`, 오래된 job/session/log/LLM cache 누적분.
- 가장 큰 운영 리스크: fallback CSV와 기본 CSV 경로 불일치로 인한 stale 챗봇 응답.
