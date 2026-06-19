# 08. 뉴스 임팩트 모듈

`src/news_impact/`는 vendored `stock-news-impact` 패키지를 통합한 선택 기능이다. 단독 CLI
(`stock-news-impact`)로 리포트를 만들 수 있고, 메인 파이프라인은 생성된 값을 **표시 전용 컨텍스트**로만 붙인다.

> 핵심 정책: 매수/매도/보유 결정과 순위는 `predicted_return`만 사용한다. 뉴스/공시/LLM 결과는 화면 표시와
> 검토 보조용이며 기대수익률, 추천, 신호를 바꾸지 않는다.

---

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `pipeline.py` / `run.py` | 뉴스 임팩트 파이프라인 진입점과 CLI 래퍼 |
| `collectors.py` / `article_fetcher.py` | 뉴스/공시 수집, 기사 본문 가져오기 |
| `deduper.py` / `mapper.py` | 중복 제거, 뉴스-종목 매핑 |
| `news_filter.py` / `safety_filter.py` | 관련성·안전 필터 |
| `impact_judge.py` | LLM 기반 임팩트 판정, 프롬프트 안전 검사 |
| `llm_client.py` / `llm_config.py` / `env_config.py` | LLM 클라이언트, 설정, 환경변수 설정 |
| `llm_smoke.py` | LLM 연결 스모크 테스트 |
| `scorer.py` / `ranking.py` / `weight_tuning.py` | 임팩트 점수 집계, 순위, 가중치 튜닝 |
| `semantic_clusterer.py` | 의미 기반 클러스터링 |
| `event_taxonomy.py` / `sector_keywords.py` | 이벤트 분류 체계, 섹터 키워드 |
| `market_clock.py` | 한국 시장 개장/마감 시각 처리 |
| `global_market_collector.py` / `global_proxy_loader.py` / `global_proxy_adjuster.py` | 글로벌 프록시 수집·로드·보정 |
| `data_cache.py` | 수집/응답 데이터 캐시 |
| `report.py` / `schema.py` | CSV/JSON 리포트 출력, 데이터 클래스 정의 |
| `backtester.py` / `backtest_snapshots.py` / `performance_validation.py` | 뉴스 점수 검증·백테스트 유틸리티 |
| `operations.py` | 운영 보조 |
| `stock_factors/` | 뉴스-주식 팩터 분류 서브패키지(classifier, factor_taxonomy, impact_rules, freshness, output_schema) |

---

## 전체 흐름

```text
watchlist.csv + company_master.csv
  -> collectors.py -> article_fetcher.py -> deduper.py -> mapper.py
  -> news_filter.py / safety_filter.py -> impact_judge.py
  -> scorer.py -> semantic_clusterer.py -> ranking.py -> report.py
```

한국 종목에는 한국 뉴스/공시를 우선 사용하고, 해외 매체/비한국어 소스는 명시적으로 필요한 경우에만 보조로 쓴다.

---

## LLM 판정 (`impact_judge.py`)

공개 API: `build_system_prompt()`, `build_news_user_prompt()`, `detect_prompt_injection()`, `judgment_to_impact_event()`.

`LLM_REQUIRED_KEYS`:

| 키 | 설명 |
|---|---|
| `event_type` | 이벤트 분류 |
| `direction` | 주가 영향 방향 |
| `impact_score` | -100 ~ +100 임팩트 점수 |
| `impact_strength` | 0.0 ~ 1.0 이벤트 강도 |
| `confidence` | 0.0 ~ 1.0 판정 신뢰도 |
| `time_horizon` | 영향 시간 범위 |
| `reason` / `why_may_be_wrong` | 근거 / 반대 시나리오 |
| `risk_flags` | 위험 플래그 목록 |

누락 키가 있으면 `judgment_to_impact_event()`는 `ValueError`를 발생시킨다.

시스템 프롬프트 본문은 `docs/NEWS_IMPACT_LLM_PROMPT.md`에서 읽는다. `build_system_prompt()`는 이 파일 내용에
안전 가드(`PROMPT_SOURCE`, JSON-only, 매수/매도 추천 금지)를 덧붙인다. 프롬프트 수정 시 이 파일을 편집한다.

### 프롬프트 안전

- 기사 본문은 `build_news_user_prompt()`에서 `<untrusted_article_text>` 블록 안에 넣는다.
- 시스템 프롬프트는 기사/공시 내부 지시를 따르지 말고 JSON만 반환하도록 요구한다.
- `detect_prompt_injection()`은 영/한 인젝션 문구(이전 지시 무시, 시스템 프롬프트, 매수/매도 추천 등)를
  감지하면 `prompt_injection_risk`를 붙인다. 이 플래그는 참고·감사용이며 예측값/추천을 바꾸지 않는다.

---

## LLM 클라이언트와 재현성 (`llm_client.py`)

| 구성 | 설명 |
|---|---|
| `LlamaCppClient` | OpenAI 호환 `/chat/completions` 호출 클라이언트 |
| `FileLLMResponseCache` | 파일 기반 JSON 응답 캐시 |
| `LLMResponseError` / `LLMModelAliasError` | 응답 오류 / 설정 모델 부재 오류 |

| Provider | 설정 |
|---|---|
| 로컬 Gemma/Llama | `llm_provider: "llama_cpp"` |
| OpenAI | `llm_provider: "openai"` + `OPENAI_API_KEY` |

캐시 키는 요청 payload + required key 목록의 sha256으로 생성한다. 캐시 파일은
`stock-news-impact.llm_cache.v1` 봉투 포맷으로 저장되며 재현성 메타데이터를 함께 남긴다.

```json
{
  "schema": "stock-news-impact.llm_cache.v1",
  "metadata": {
    "model": "...", "temperature": 0.1,
    "prompt_hash": "<시스템 프롬프트 sha256>",
    "article_hash": "<유저 프롬프트 sha256>",
    "required_keys": ["confidence", "direction", "..."]
  },
  "response": { "...LLM JSON..." }
}
```

런 단위 재현성은 `audit.json`에 남는다. `RunAudit`는 `llm_model_requested`/`llm_model_returned`에 더해
`llm_temperature`, `llm_prompt_hash`를 기록하고 `replay` 블록에도 노출한다. 봉투가 아닌 레거시 캐시 파일도 하위호환으로 읽는다.

---

## 메인 파이프라인 통합과 보호 장치

```python
if news_impact_report:
    pred_df = append_news_impact_context(pred_df, news_impact_report)
elif news_impact_llm_config:
    pred_df = append_llm_news_impact_context(...)
```

`--news-impact-report` 또는 `--news-impact-llm-config`를 쓰면 `result_detail.csv`에 `news_impact_*` 표시 컬럼이 추가된다.

- `feature_selection.DISPLAY_ONLY_CONTEXT_PREFIXES = ("news_impact_",)`로 모든 `news_impact_*` 컬럼이
  모델 입력에서 제외된다. `_missing` 파생 컬럼이 생겨도 피처로 들어가지 않는다.
- 정책 테스트가 추천, 순위, `predicted_return`이 뉴스 문맥으로 바뀌지 않는지 확인한다.

---

## 실행 예시

```powershell
Copy-Item configs/news_impact.example.json configs/news_impact.json
stock-news-impact --help
stock-news-impact `
  --watchlist data/news_impact/watchlist.csv `
  --company-master data/news_impact/company_master.csv `
  --output result/news_impact_report.json
```

관련 샘플: `configs/news_impact.example.json`(OpenAI 기본), `configs/news_impact.gemma.example.json`(로컬 LLM),
`data/news_impact/*.example.csv`(watchlist, company_master, aliases, sector_keywords, relationships, global_market_proxy).

---

## Improvement and Fix Proposals

> Priority: **P2 (operations)**.

### P2 - LLM response cache lifecycle management
- **Problem**: `FileLLMResponseCache` previously had no expiration or invalidation policy, so cache directories could grow indefinitely and stale responses could remain after prompt/model changes.
- **Implemented**:
  - `ttl_seconds` expires enveloped cache entries from `metadata.cached_at`.
  - `expected_metadata` treats `prompt_hash`, `article_hash`, `model`, `temperature`, and `required_keys` mismatches as cache misses.
  - `max_entries` prunes the oldest JSON files when the cache exceeds the configured entry limit.
  - Legacy bare dict cache entries remain readable when callers do not request `expected_metadata` validation.
