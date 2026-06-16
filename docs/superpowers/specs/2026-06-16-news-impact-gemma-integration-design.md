# News-Impact gemma 통합 설계 (on-demand)

- 날짜: 2026-06-16
- 상태: 승인됨 (설계) — 스펙 리뷰 대기
- 작성 맥락: `stock-news-impact` 독립 프로젝트를 `stock_predict`에 `src/news_impact/`로 합쳤으나, gemma 기반 임팩트 판정 파이프라인이 챗봇 실행 경로에 연결되지 않아 사실상 죽은 코드로 남아 있음. 이를 의도대로 살린다.

## 1. 목표

카카오 챗봇이 **요청받은 종목**에 대해 예측을 돌릴 때, 뉴스/공시 임팩트 점수를 현재의 규칙 기반(heuristic) 대신 **로컬 gemma(`gemma-4-26b-a4b`) LLM 판정**으로 산출해 예측 리포트에 붙인다. 단, 이 점수는 기존과 동일하게 **표시용(display-only)** 이며 예측 모델 입력/산출에는 반영하지 않는다.

## 2. 배경 — 현재 무엇이 끊겨 있나 (근거)

- gemma를 쓰는 진짜 파이프라인은 독립 CLI(`stock-news-impact daily-run`, `src/news_impact/run.py:130`)로만 실행되며, `--llm-config`로 gemma 설정을 줄 때만 gemma가 켜진다 (`src/news_impact/pipeline.py:89-93`). 미지정 시 `LLMConfig.default()` = OpenAI.
- 챗봇은 예측을 `src/pipeline.py` subprocess로 돌리는데(`src/chatbot/kakao_colab_bot.py:1121` `build_command`), 그 명령에 `--news-impact-report`도 `--llm-config`도 넘기지 않는다 (`kakao_colab_bot.py:102-125`).
- 그 결과 `src/pipeline.py:738-741`의 분기에서 **항상 else** → `append_generated_news_impact_context`(규칙 기반, `risk_flags=["heuristic_display_only"]`)만 실행된다.
- repo 전체에서 `run_daily_pipeline` / `--news-impact-report` / gemma config를 실제 호출·전달하는 곳은 `run.py`(CLI)와 테스트뿐. → gemma는 챗봇 흐름에서 한 번도 호출되지 않음.

## 3. 확정된 결정

| 항목 | 결정 |
| --- | --- |
| 모델 | `gemma-4-26b-a4b`, provider `llama_cpp` |
| 엔드포인트 | `http://localhost:8001/v1` (`configs/news_impact.gemma.example.json` 그대로) |
| 실행 위치 | 챗봇·gemma 서버 모두 동일 로컬 PC |
| 적용 범위 | **요청 종목만 gemma 판정** |
| 부트스트랩(전 종목) | 기존 규칙 기반 유지 (26B로 전 종목은 비현실적) |
| 폴백 | gemma 서버 무응답/예외/모델 alias 불일치 시 규칙 기반으로 자동 복귀 |
| 점수 성격 | 표시용 유지, 예측 미반영 |

## 4. 아키텍처 & 데이터 흐름

```
카톡 요청(종목) → 챗봇(_start_prediction_job)
  └ subprocess: python src/pipeline.py … [--news-impact-llm-config configs/news_impact.gemma.example.json]
       ├ 뉴스/공시 수집 → context_raw_df  (기존)
       ├ append_issue_summary_columns(...)  (기존, OpenAI — 이번 범위 밖)
       └ news_impact 분기 (src/pipeline.py:738-741)
            ├ [신규] llm_config 지정 & 대상 종목 →
            │     build_news_impact_fixture(context_raw_df, 종목, name_map)
            │       → 임시 fixture.json / watchlist.csv / company_master.csv
            │     → run_daily_pipeline(DailyPipelineInputs(llm_config_path=gemma))
            │       → result.report_rows
            │     → append_news_impact_rows(pred_df, rows)
            │     (예외 발생 시 ↓ 규칙 기반으로 폴백)
            └ else → append_generated_news_impact_context  (기존)
```

- 핵심 변경점은 단 한 곳의 분기(`src/pipeline.py:738-741`)에 gemma 경로를 추가하는 것. 기존 규칙 기반은 **삭제하지 않고 폴백 경로로 유지**한다.
- gemma 호출은 챗봇 예측 subprocess 내부에서 인-프로세스로 일어나며, 챗봇은 이미 이 작업을 백그라운드로 비동기 실행("분석 중" 응답)하므로 26B 지연이 사용자 응답을 막지 않는다.

## 5. 컴포넌트

### 5.1 fixture 빌더 (신규)
- 위치(안): `src/reports/news_impact_fixture.py` (신규 모듈)
- 시그니처(안): `build_news_impact_fixture(context_raw_df, symbols, symbol_name_map, run_date) -> NewsImpactFixtureBundle`
  - 반환: 임시 디렉터리에 쓴 `fixture_path`, `watchlist_path`, `company_master_path` 경로 묶음
- 책임: `context_raw_df`(컬럼 `Date, Symbol, source_type, title, published_at, provider, url, raw_id`)를 `run_daily_pipeline`이 읽는 fixture JSON(`{"news":[…], "disclosures":[…]}`) + watchlist/company-master CSV로 변환.

### 5.2 gemma 분기 함수 (신규)
- 위치(안): `src/reports/news_impact_context.py`에 `append_llm_news_impact_context(pred_df, context_raw_df, *, llm_config_path, symbols, symbol_name_map)` 추가
- 책임: fixture 빌드 → `run_daily_pipeline(llm_config_path=…)` 호출 → `report_rows`를 dict로 변환 → 기존 `append_news_impact_rows(pred_df, rows)` 재사용.
- 예외(`LLMResponseError`, `URLError`, `LLMModelAliasError`, 타임아웃 등) → 잡아서 `None`/원본 반환하고 호출부가 규칙 기반으로 폴백하게 함. 폴백 사유는 로그로 남김.

### 5.3 `src/pipeline.py`
- CLI 인자 추가: `--news-impact-llm-config <path>` (기본 None). `run_pipeline(...)`까지 전달.
- 분기 수정(738-741):
  ```python
  if news_impact_report:
      pred_df = append_news_impact_context(pred_df, news_impact_report)
  elif news_impact_llm_config:
      pred_df = append_llm_news_impact_context(
          pred_df, context_raw_df,
          llm_config_path=news_impact_llm_config,
          symbols=issue_summary_symbols,   # 요청 종목과 동일 집합
          symbol_name_map=symbol_name_map,
      )  # 내부 실패 시 규칙 기반 폴백
  else:
      pred_df = append_generated_news_impact_context(pred_df, context_raw_df)
  ```

### 5.4 `src/chatbot/kakao_colab_bot.py`
- `PipelineRuntimeConfig`에 `news_impact_llm_config: str | None = None` 필드 추가.
- `build_command`: **부트스트랩이 아닌 온디맨드 작업**에서만 `--news-impact-llm-config <path>` 추가. (부트스트랩은 add_symbols가 전 종목이므로 미전달.)
- 노트북에서 `news_impact_llm_config="configs/news_impact.gemma.example.json"`로 설정.

## 6. fixture 스키마 매핑 (가장 큰 작업/리스크)

`context_raw_df`는 8개 컬럼뿐이라 `NewsItem`/`DisclosureItem`의 나머지 필수 필드를 **기본값으로 합성**해야 한다. 검증 규칙(`src/news_impact/schema.py`)을 통과해야 함.

### NewsItem (source_type == "news")
| 필드 | 출처/기본값 |
| --- | --- |
| source | `provider` 또는 `"naver"` |
| title | `title` |
| summary | `""` (본문 미수집) |
| url | `url` |
| original_url | `url` 또는 None |
| publisher_domain / _source | None / None |
| publisher_confidence | `0.0` |
| published_at | `published_at` 파싱, 없으면 `run_date` 장마감(KST) 합성 — **tz-aware 필수** |
| timestamp_source | `"naver_pubDate"` 있으면, 합성 시 `"manual"` |
| collected_at | `now(KST)` 또는 published_at — tz-aware |
| signal_at | **반드시 `max(published_at, collected_at)`** (검증됨) |
| market_session | `published_at` 기준 `market_clock`로 산출 (불명 시 `"regular"`) |
| raw_text | None |
| storage_policy | `"metadata_only"` |
| quality_flags | `["title_only"]` 등 |

### DisclosureItem (source_type == "disclosure")
| 필드 | 출처/기본값 |
| --- | --- |
| source | `provider` 또는 `"dart"` |
| receipt_no | `raw_id` 또는 합성 ID |
| corp_code | name_map/dart_corp_map 조회, 없으면 `""` |
| ticker | `Symbol` → 6자리 (예: `005930.KS` → `005930`) — **6자 검증** |
| disclosure_title | `title` |
| disclosure_at | `published_at`/합성 — tz-aware |
| collected_at / signal_at | NewsItem과 동일 규칙 |
| is_correction | 제목에 "정정" 포함 여부 |
| original_receipt_no | None |
| url | `url` |
| quality_flags | `[]` |

> 주의: `published_at`이 빈 문자열인 행이 존재(`src/pipeline.py:186`). datetime 합성 + `signal_at == max(...)` 불변식을 빌더에서 보장해야 한다.

## 7. 뉴스→티커 매핑 결정

- gemma 파이프라인의 `_build_llm_judged_events`는 watchlist/company-master를 받아 **뉴스를 내부 mapper로 티커에 매핑**한다(뉴스에는 ticker 필드가 없음).
- 우리는 종목 1개 단위로 돌리므로 **watchlist = [요청 티커]**, **company-master = {요청 티커: {ticker, name, …}}** 로 구성한다.
- 매핑 실패로 이벤트가 0건이 되면 → 해당 종목은 규칙 기반으로 폴백한다(품질 저하 방지). 테스트로 단일 종목 매핑이 동작함을 고정한다.

## 8. 에러 처리 / 폴백

- gemma 서버 무응답·alias 불일치·타임아웃·스키마 검증 실패 → 모두 잡아 규칙 기반(`append_generated_news_impact_context`)으로 복귀. 챗봇 응답 흐름이 절대 깨지지 않게 한다.
- gemma 결과의 `risk_flags`는 LLM 판정 표식을 유지하여 휴리스틱 결과와 구분 가능하게 한다.
- 폴백 발생 시 로그 1줄(사유 포함).

## 9. 범위 밖 (YAGNI)

- 부트스트랩 전 종목 gemma 판정.
- 배치/스케줄 사전계산 + 캐시 리포트 방식(대안 B).
- `issue_summary`(공시·뉴스 요약, OpenAI)의 gemma 전환 — **별도 후속 작업**.
- 뉴스 본문(raw_text) 수집 확장.
- 원격 gemma 엔드포인트/터널링.

## 10. 테스트

- fixture 빌더 단위 테스트: context_raw_df → fixture/watchlist/company-master, datetime tz-aware + `signal_at` 불변식, ticker 6자 변환, 빈 published_at 합성.
- 분기 선택 테스트: `news_impact_llm_config` 지정 시 `run_daily_pipeline` 호출(임팩트 판정 LLM은 `impact_judge_llm` 주입 mock으로 대체), 미지정 시 휴리스틱.
- 폴백 테스트: LLM mock이 예외 → 규칙 기반 결과로 복귀.
- `build_command` 테스트: 온디맨드엔 `--news-impact-llm-config` 포함, 부트스트랩엔 미포함.
- 기존 `tests/test_news_impact_context.py`, `test_news_impact_full_package.py` 패턴 재사용.

## 11. 선행 조건 (코드 외)

- 로컬에 gemma 서버 구동: `/v1/models`가 `gemma-4-26b-a4b` 서빙.
- 검증: `python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json` 가 `status: ok`.
- 구현 1단계는 이 연결 검증으로 시작한다.

## 12. 리스크 & 열린 질문

- **fixture 스키마 갭(§6)** 이 본 작업의 난이도 핵심. 합성 기본값이 gemma 판정 품질에 영향(특히 본문 부재로 제목 기반 판정).
- 26B 로컬 지연: 단일 종목·소수 클러스터라 허용 범위로 가정. 타임아웃은 config의 `timeout_seconds`로 관리.
- `report_rows`(ReportRow) → `append_news_impact_rows`가 기대하는 dict 키 정합성은 구현 계획에서 `report.py`로 최종 확인.

## 13. 단계 (요약)

1. gemma 서버 연결 검증(llm-smoke).
2. fixture 빌더 + 테스트.
3. gemma 분기 함수 + 폴백 + 테스트.
4. `src/pipeline.py` 인자/분기 연결 + 테스트.
5. `kakao_colab_bot.build_command` 연결 + 테스트.
6. 노트북 config 설정 + 단일 종목 e2e 확인.
