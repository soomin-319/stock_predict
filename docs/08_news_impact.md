# 08. 뉴스 임팩트 모듈

`src/news_impact/`는 vendored `stock-news-impact` 패키지를 현재 저장소에 통합한 선택 기능이다. 단독 CLI(`stock-news-impact`)로 리포트를 만들 수 있고, 메인 파이프라인은 생성된 값을 **표시 전용 문맥**으로만 붙인다.

> 핵심 정책: 매수/매도/보유 결정과 순위는 `predicted_return`만 사용한다. 뉴스/공시/LLM 결과는 화면 표시와 검토 보조용이며 기대수익률, 추천, 신호를 바꾸면 안 된다.

---

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `pipeline.py` | 뉴스 임팩트 파이프라인 진입점 |
| `run.py` | CLI 실행 래퍼 |
| `collectors.py` | 뉴스/공시 수집 |
| `article_fetcher.py` | 기사 본문 가져오기 |
| `deduper.py` | 중복 제거 |
| `mapper.py` | 뉴스-종목 매핑 |
| `news_filter.py` / `safety_filter.py` | 관련성·안전 필터 |
| `impact_judge.py` | LLM 기반 임팩트 판정, 프롬프트 안전 검사 |
| `llm_client.py` / `llm_config.py` | LLM API 클라이언트와 설정 |
| `scorer.py` | 임팩트 점수 집계 |
| `semantic_clusterer.py` | 의미 기반 클러스터링 |
| `ranking.py` | 리포트용 순위 계산 |
| `report.py` | CSV/JSON 리포트 출력 |
| `schema.py` | 데이터 클래스 정의 |
| `backtester.py` / `backtest_snapshots.py` | 뉴스 점수 검증·백테스트 유틸리티 |
| `stock_factors/` | 뉴스-주식 팩터 분류 서브패키지 |

---

## 전체 흐름

```text
watchlist.csv + company_master.csv
  -> collectors.py
  -> article_fetcher.py
  -> deduper.py
  -> mapper.py
  -> news_filter.py / safety_filter.py
  -> impact_judge.py
  -> scorer.py
  -> semantic_clusterer.py
  -> ranking.py
  -> report.py
```

한국 종목에는 한국 뉴스와 공시를 우선 사용한다. 해외 매체나 비한국어 소스는 명시적으로 필요한 경우에만 보조로 쓴다.

---

## LLM 판정 (`impact_judge.py`)

공개 API:

```python
def build_system_prompt() -> str
def build_news_user_prompt(analysis_input: NewsAnalysisInput, summary: str | None = None) -> str
def detect_prompt_injection(text: str) -> tuple[str, ...]
def judgment_to_impact_event(...) -> ImpactEvent
```

`LLM_REQUIRED_KEYS`:

| 키 | 설명 |
|---|---|
| `event_type` | 이벤트 분류 |
| `direction` | 주가 영향 방향 |
| `impact_score` | -100 ~ +100 임팩트 점수 |
| `impact_strength` | 0.0 ~ 1.0 이벤트 강도 |
| `confidence` | 0.0 ~ 1.0 판정 신뢰도 |
| `time_horizon` | 영향 시간 범위 |
| `reason` | 근거 |
| `why_may_be_wrong` | 반대 시나리오 |
| `risk_flags` | 위험 플래그 목록 |

누락 키가 있으면 `judgment_to_impact_event()`는 `ValueError`를 발생시킨다.

### 프롬프트 안전

- 기사 본문은 `build_news_user_prompt()`에서 `<untrusted_article_text>` 블록 안에 넣는다.
- 시스템 프롬프트는 기사/공시 내부 지시를 따르지 말고 JSON만 반환하도록 요구한다.
- `detect_prompt_injection()`은 영어와 한국어 인젝션 문구(예: 이전 지시 무시, 시스템 프롬프트, 매수/매도 추천)를 감지하면 `prompt_injection_risk`를 붙인다.
- 이 플래그는 LLM 판단의 참고·감사용이다. 뉴스 문맥이 예측값이나 추천을 바꾸면 안 된다.

---

## LLM 클라이언트 (`llm_client.py`)

| 구성 | 설명 |
|---|---|
| `LlamaCppClient` | OpenAI 호환 `/chat/completions` 호출 클라이언트 |
| `FileLLMResponseCache` | 파일 기반 JSON 응답 캐시 |
| `LLMResponseError` | 응답 shape/JSON 오류 |
| `LLMModelAliasError` | 설정 모델이 런타임에 없을 때 |

지원 설정:

| Provider | 설정 |
|---|---|
| 로컬 Gemma/Llama | `llm_provider: "llama_cpp"` |
| OpenAI | `llm_provider: "openai"` + `OPENAI_API_KEY` |

예시:

```json
{
  "llm_provider": "llama_cpp",
  "llm_base_url": "http://localhost:8001/v1",
  "llm_model": "gemma-4-26b-a4b",
  "temperature": 0.1,
  "max_retries": 2,
  "json_schema_required": true,
  "timeout_seconds": 60
}
```

캐시 키는 요청 payload와 required key 목록에서 생성한다. 재현성 검토에는 모델명, 프롬프트 버전, 온도, 기사 해시, required key 목록을 함께 기록하는 방식이 적합하다.

---

## 메인 파이프라인 통합

```python
if news_impact_report:
    pred_df = append_news_impact_context(pred_df, news_impact_report)
elif news_impact_llm_config:
    pred_df = append_llm_news_impact_context(...)
```

`--news-impact-report result/news_impact_report.json` 또는 `--news-impact-llm-config ...`를 쓰면 `result_detail.csv`에 `news_impact_*` 표시 컬럼이 추가된다.

보호 장치:

- `src.features.feature_selection.DISPLAY_ONLY_CONTEXT_COLUMNS`는 기존 뉴스/공시 컬럼을 모델 입력에서 제외한다.
- `select_feature_columns()`는 `news_impact_` 접두어 컬럼을 전부 제외한다. 새 `news_impact_*` 컬럼이나 `_missing` 파생 컬럼이 생겨도 모델 feature로 들어가지 않는다.
- 정책 테스트는 추천, 순위, `predicted_return`이 뉴스 문맥으로 바뀌지 않는지 확인한다.

---

## 백테스트 유틸리티

`backtester.py` 공개 함수:

```python
def match_signal_returns(...)
def calculate_metrics(...)
def summarize_by_bucket(...)
def compare_score_variants(...)
```

`BacktestSignal`과 `PriceBar`를 매칭해 거래비용 반영 수익률을 만들고 IC, rank IC, hit ratio, top-bottom spread 등을 계산한다. 백테스트 결과는 연구 검증용이며 자동 매매 신호가 아니다.

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

---

## 관련 설정·샘플 파일

| 파일 | 용도 |
|---|---|
| `configs/news_impact.example.json` | 기본 로컬 LLM 설정 예시 |
| `configs/news_impact.gemma.example.json` | Gemma/Llama 설정 예시 |
| `configs/news_impact.openai.example.json` | OpenAI 설정 예시 |
| `data/news_impact/watchlist.example.csv` | 모니터링 종목 샘플 |
| `data/news_impact/company_master.example.csv` | 기업 마스터 샘플 |
| `data/news_impact/company_aliases.example.csv` | 기업 별칭 샘플 |
| `data/news_impact/sector_keywords.example.csv` | 섹터 키워드 샘플 |
| `data/news_impact/company_relationships.example.csv` | 기업 관계도 샘플 |
| `data/news_impact/global_market_proxy.example.csv` | 글로벌 프록시 샘플 |

---

## 개선 및 수정 진행 현황 (2026-06-17)

### 완료: P0 문서 인코딩·계약 정리

- 깨진 본문을 UTF-8 한국어 문서로 재작성했다.
- 실제 `LLM_REQUIRED_KEYS`와 `ImpactEvent` 필드 기준으로 설명을 맞췄다.
- 뉴스 임팩트가 표시 전용이며 `predicted_return`, 추천, 순위를 바꾸면 안 된다는 정책을 명시했다.

### 완료: P1 프롬프트 인젝션 방어 보강

- 기사 본문을 `<untrusted_article_text>` 블록으로 격리하는 구조를 문서화했다.
- 한국어 인젝션 문구 감지 테스트를 추가했다.
- `detect_prompt_injection()`에 한국어 위험 문구를 추가했다.

### 완료: P2 표시 전용 feature guard 강화

- `select_feature_columns()`가 `news_impact_` 접두어 컬럼을 전부 제외하도록 강화했다.
- `news_impact_*_missing` 같은 신규 파생 컬럼도 모델 입력에서 제외되는 회귀 테스트를 추가했다.

### 남은 제안

- `FileLLMResponseCache` 캐시 메타데이터에 프롬프트 파일 해시와 모델 alias를 명시적으로 저장한다.
- LLM 리포트에 `temperature`, `llm_model`, `prompt_hash`, `article_hash`를 남겨 재현성을 높인다.
- `stock-news-impact` 단독 실행 경로와 메인 파이프라인 통합 경로를 작은 다이어그램으로 README에 추가한다.
