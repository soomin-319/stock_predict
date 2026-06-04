# 캐시 파일 생성 파트 설명

이 문서는 현재 코드 기준으로 캐시성 파일이 어디서, 어떤 조건으로 만들어지는지 정리한다.

## 범위

주요 캐시/상태 파일은 `result/` 아래에 생성된다.

| 파일 | 생성 주체 | 용도 |
| --- | --- | --- |
| `result/result_simple.csv` | `src/pipeline.py` | Kakao 챗봇이 바로 읽는 사용자용 예측 결과 캐시 |
| `result/result_detail.csv` | `src/pipeline.py` | 최신 예측 상세 결과와 피처/정책 컨텍스트 |
| `result/prewarm_cache_meta.json` | `src/chatbot/kakao_colab_bot.py` | prewarm 캐시 재사용 여부 판단용 서명 메타데이터 |
| `result/chatbot_jobs.json` | `src/chatbot/kakao_colab_bot.py` | 챗봇 백그라운드 예측 작업 상태 |
| `result/chatbot_sessions.json` | `src/chatbot/kakao_colab_bot.py` | 사용자별 마지막 종목/의도 세션 상태 |
| `result/chatbot_logs/*.log` | `src/chatbot/kakao_colab_bot.py` | 예측 subprocess 및 bootstrap 로그 |
| `result/.pytest_cache` | `pyproject.toml` | pytest 내부 캐시 |

별도 모듈로 `src/news_impact/data_cache.py`의 `DataCache`도 JSON 캐시 작성 기능을 제공하지만, 현재 repository 내에서는 직접 호출 지점이 없다.

## 예측 결과 캐시 생성 흐름

### 1. 파이프라인 실행

`colab/stock_predict_colab.py`의 `run_colab_pipeline()` 또는 CLI가 `src.pipeline.run_pipeline()`을 호출한다.

Colab 경로:

```text
prewarm_prediction_cache()
  -> run_colab_pipeline()
    -> run_pipeline()
```

일반 챗봇 종목 요청 경로:

```text
KakaoColabPredictionBot._start_prediction_job()
  -> PipelineRuntimeConfig.build_command()
  -> python src/pipeline.py ...
  -> run_pipeline()
```

### 2. `run_pipeline()`이 결과 DataFrame 생성

`src/pipeline.py`는 예측/정책/리포트 컬럼을 만든 뒤:

- `detail_df`를 만든다.
- `_build_result_simple(detail_df)`로 사용자용 `simple_df`를 만든다.
- `_safe_to_csv()`로 두 CSV를 저장한다.

관련 코드 위치:

- `src/pipeline.py`
  - `_build_result_simple()`
  - `resolve_output_path()`
  - `_safe_to_csv()`
  - `run_pipeline()`
- `src/reports/output.py`
  - `project_result_dir()`
  - `resolve_output_path()`
  - `safe_to_csv()`
  - `build_pipeline_result_simple()`
- `src/reports/result_formatter.py`
  - `build_result_simple()`
  - `validate_result_simple_schema()`

### 3. 저장 위치 강제

`src/reports/output.py`의 `resolve_output_path()`는 모든 CSV/JSON 산출물을 project-local `result/` 아래로 보낸다.

예:

```text
result_detail.csv -> result/result_detail.csv
result_simple.csv -> result/result_simple.csv
```

절대경로가 들어와도 `result/` 밖이면 파일명만 가져와 `result/` 아래에 저장한다.

### 4. CSV 인코딩

`safe_to_csv()`는 `encoding="utf-8-sig"`로 저장한다. Windows Excel 호환을 위한 정책이다.

파일이 열려 있어 `PermissionError`가 나면:

```text
result/result_simple.csv
-> result/result_simple_fallback.csv
```

처럼 fallback 파일명으로 저장한다.

## `result_simple.csv` 스키마

`src/reports/result_formatter.py`가 사용자용 캐시 컬럼을 만든다.

필수 컬럼:

- `종목코드`
- `종목명`
- `권고`
- `내일 예상 종가`
- `내일 예상 수익률(%)`
- `상승확률(%)`
- `예측 신뢰도`

선택 컬럼:

- `5일 예상 수익률(%)`
- `20일 예상 수익률(%)`
- `5일 상승확률(%)`
- `20일 상승확률(%)`
- `예측 이유`
- `공시 요약`
- `뉴스 요약`
- `뉴스/공시 영향 점수`
- `뉴스/공시 영향 요약`
- `뉴스/공시 영향 참고`

정렬은 신뢰도와 예상 수익률 기준 내림차순이다.

주의: 뉴스/공시 컬럼은 표시용이다. 매수/매도/보유 판단은 `predicted_return` 기반 정책 결과를 사용한다.

## 챗봇의 캐시 읽기

`KakaoColabPredictionBot._find_cached_prediction()`은 `result_simple.csv`에서 요청 종목을 찾는다.

흐름:

```text
_handle_symbol_request()
  -> _find_cached_prediction(symbol)
    -> _load_cached_result_simple()
      -> pandas.read_csv(result/result_simple.csv, encoding="utf-8-sig")
      -> validate_result_simple_schema()
```

`_load_cached_result_simple()`은 파일 `mtime_ns`를 저장한다.

- 파일 mtime이 같으면 메모리의 `_result_simple_cache`를 복사해 반환한다.
- mtime이 바뀌면 CSV를 다시 읽는다.
- 파일이 없거나 읽기 실패면 빈 DataFrame을 반환한다.

즉, `result_simple.csv` 자체는 디스크 캐시이고, `_result_simple_cache`는 프로세스 내부 메모리 캐시다.

## prewarm 캐시 생성

`src/chatbot/kakao_colab_bot.py`의 `prewarm_prediction_cache()`가 기본 예측 캐시를 미리 만든다.

주요 경로:

```text
prewarm_prediction_cache(runtime_config, force=False)
  -> result/result_simple.csv 존재 확인
  -> result/prewarm_cache_meta.json 로드
  -> 현재 runtime signature 생성
  -> signature_hash 일치하면 기존 result_simple.csv 재사용
  -> 불일치/없음/force=True면 run_colab_pipeline() 실행
  -> result_simple.csv/result_detail.csv 생성
  -> prewarm_cache_meta.json 기록
```

### prewarm 재사용 조건

다음이 모두 만족되어야 재사용한다.

1. `force=False`
2. `result/result_simple.csv` 존재
3. CSV가 비어 있지 않음
4. `result/prewarm_cache_meta.json`의 `signature_hash`가 현재 hash와 같음

### signature에 포함되는 값

`_runtime_cache_signature()`는 다음 정보를 해시에 넣는다.

- KST 기준 캐시 날짜 `cache_date_kst`
- 입력 CSV 경로, mtime, 크기
- 기본 universe CSV mtime, 크기
- 리포트 JSON 이름
- figure directory
- 외부 피처 사용 여부
- 투자자/공시 context 설정
- 기본 심볼 bootstrap 여부
- real data start date
- DART/OpenAI/Naver 설정 활성 여부
- OpenAI model 이름

그래서 날짜가 바뀌거나 입력 CSV가 바뀌거나 외부 context 설정이 바뀌면 prewarm 캐시는 무효화된다.

### prewarm 메타 파일

`_write_prewarm_meta()`는 다음 구조의 JSON을 쓴다.

```json
{
  "signature": {
    "cache_date_kst": "YYYY-MM-DD",
    "input_csv": "data/real_ohlcv.csv"
  },
  "signature_hash": "sha256..."
}
```

실제 `signature`에는 더 많은 필드가 들어간다.

## bootstrap 작업 상태 캐시

챗봇 시작 또는 최초 요청 시 전체 기본 종목 예측을 백그라운드로 돌릴 수 있다.

관련 코드:

- `_start_bootstrap_job()`
- `_run_bootstrap_prewarm_worker()`
- `_save_registry()`

흐름:

```text
_start_bootstrap_job()
  -> result/chatbot_jobs.json에 BOOTSTRAP running 기록
  -> thread 시작
  -> _run_bootstrap_prewarm_worker()
    -> prewarm_prediction_cache()
    -> 완료/실패 상태를 chatbot_jobs.json에 기록
```

`chatbot_jobs.json`에는 작업별로 다음 정보가 들어간다.

- `symbol`
- `display_code`
- `command`
- `log_path`
- `submitted_at`
- `status`
- `pid`
- `exit_code`
- `completed_at`

`_save_registry()`는 임시 파일에 먼저 쓰고 `replace()`로 교체한다.

```text
chatbot_jobs.json.tmp -> chatbot_jobs.json
```

이 방식은 작업 상태 JSON이 반쯤 쓰이는 위험을 줄인다.

## 종목별 예측 작업 상태 캐시

사용자가 캐시에 없는 종목을 요청하면 `_start_prediction_job()`이 subprocess를 시작한다.

흐름:

```text
_handle_symbol_request()
  -> 캐시 미존재
  -> _start_job_response()
    -> _start_prediction_job()
      -> subprocess.Popen(command)
      -> result/chatbot_jobs.json에 running 기록
      -> result/chatbot_logs/{종목}_{시각}.log 생성
```

subprocess 완료 후 `_finalize_process()`가:

- 작업 상태를 `completed` 또는 `failed`로 변경
- `exit_code`, `completed_at` 기록
- 이후 사용자가 다시 조회하면 `result_simple.csv`를 다시 읽어 응답

중요: `_finalize_process()`가 직접 `result_simple.csv`를 만들지는 않는다. CSV 생성은 subprocess 안의 `run_pipeline()`이 한다.

## 사용자 세션 캐시

`_update_session()`은 사용자별 마지막 종목과 intent를 `result/chatbot_sessions.json`에 저장한다.

용도:

- 사용자가 `결과`라고만 입력해도 마지막 종목을 다시 조회
- `최신화` 입력 시 마지막 종목 재예측

저장도 `_save_registry()`를 사용하므로 임시 파일 후 replace 방식이다.

## 뉴스/공시 요약 캐시

챗봇에는 디스크 파일이 아닌 메모리 캐시가 있다.

| 변수 | 용도 |
| --- | --- |
| `_issue_summary_cache` | 종목별 생성된 공시/뉴스 요약 row 캐시 |
| `_issue_summary_jobs` | 요약 생성 백그라운드 job 상태 |
| `_live_events_cache` | 라이브 뉴스/공시 원문 수집 결과 |

`_live_events_cache`는 TTL이 15분이다.

이 캐시는 챗봇 프로세스가 종료되면 사라진다. `result_simple.csv`와 달리 디스크 영속 캐시가 아니다.

## `src.news_impact.data_cache.DataCache`

`DataCache`는 다음 JSON 캐시를 쓸 수 있는 유틸리티다.

| 메서드 | 경로 패턴 | 내용 |
| --- | --- | --- |
| `write_news_search()` | `{root}/news/{source}/{date}/{query_hash}.json` | 뉴스 검색 결과 |
| `write_disclosure()` | `{root}/dart/{receipt_no}.json` | DART 공시 단건 |
| `write_market_price()` | `{root}/market/{vendor}/{ticker}/{date}.json` | 시장 가격 payload |
| `write_snapshot_manifest()` | `{root}/snapshots/{snapshot_id}.manifest.json` | 캐시 파일 sha256 목록 |

JSON은 `ensure_ascii=False`, `indent=2`, `sort_keys=True`로 저장한다.

현재 코드 검색 기준, 이 클래스는 정의되어 있지만 `src/`, `tests/` 안에서 실제 호출되는 곳은 없다.

## 전체 요약

현재 핵심 캐시 파일 생성 책임은 나뉘어 있다.

1. **예측 CSV 생성**: `src/pipeline.py`가 `result_detail.csv`, `result_simple.csv` 생성.
2. **캐시 재사용 판단**: `prewarm_prediction_cache()`가 `prewarm_cache_meta.json`의 signature hash로 판단.
3. **챗봇 상태 저장**: `KakaoColabPredictionBot`이 `chatbot_jobs.json`, `chatbot_sessions.json` 저장.
4. **챗봇 응답 캐시 읽기**: `_load_cached_result_simple()`이 mtime 기반으로 `result_simple.csv`를 메모리에 캐시.
5. **뉴스/공시 data cache 유틸**: `DataCache`는 JSON 캐시 기능을 제공하지만 현재 직접 사용되지는 않는다.
