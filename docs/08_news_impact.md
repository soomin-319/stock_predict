# 08. 뉴스 임팩트 모듈

`src/news_impact/`는 `stock-news-impact` 프로젝트를 이 저장소에 통합한 독립 패키지다. 별도 콘솔 스크립트 `stock-news-impact`로 단독 실행 가능하다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `pipeline.py` | 뉴스 임팩트 파이프라인 진입점 |
| `run.py` | CLI 실행 래퍼 |
| `collectors.py` | 뉴스/공시 수집 |
| `article_fetcher.py` | 기사 본문 가져오기 |
| `deduper.py` | 중복 제거 |
| `impact_judge.py` | LLM 임팩트 판정 |
| `llm_client.py` | LLM API 클라이언트 |
| `llm_config.py` | LLM 설정 로딩 |
| `llm_smoke.py` | LLM 연결 테스트 |
| `mapper.py` | 뉴스 → 종목 매핑 |
| `scorer.py` | 임팩트 점수 집계 |
| `ranking.py` | 리포트 행 순위 계산 |
| `report.py` | CSV/JSON 리포트 출력 |
| `schema.py` | 데이터 클래스 정의 |
| `event_taxonomy.py` | 이벤트 분류 체계 |
| `sector_keywords.py` | 섹터별 키워드 |
| `semantic_clusterer.py` | 의미 기반 뉴스 클러스터링 |
| `safety_filter.py` | 뉴스 안전성 필터 |
| `news_filter.py` | 뉴스 관련성 필터 |
| `market_clock.py` | 한국 시장 시간 유틸리티 |
| `data_cache.py` | 로컬 데이터 캐시 |
| `backtester.py` | 임팩트 백테스트 |
| `backtest_snapshots.py` | 백테스트 스냅샷 관리 |
| `performance_validation.py` | 임팩트 성과 검증 |
| `global_market_collector.py` | 글로벌 시장 데이터 수집 |
| `global_proxy_adjuster.py` | 글로벌 프록시 조정 |
| `global_proxy_loader.py` | 글로벌 프록시 데이터 로딩 |
| `weight_tuning.py` | 임팩트 가중치 튜닝 |
| `operations.py` | 운영 유틸리티 |
| `env_config.py` | 환경 설정 로딩 |
| `stock_factors/` | 주식 팩터 분류 서브패키지 |

---

## 전체 흐름

```
watchlist.csv + company_master.csv
        │
    collectors.py (뉴스/공시 수집)
        │
    article_fetcher.py (기사 본문)
        │
    deduper.py (중복 제거)
        │
    mapper.py (뉴스 → 종목 매핑)
        │
    news_filter.py / safety_filter.py (필터링)
        │
    impact_judge.py (LLM 임팩트 판정)
        │
    scorer.py (점수 집계)
        │
    semantic_clusterer.py (클러스터링)
        │
    ranking.py (순위 계산)
        │
    report.py (CSV/JSON 출력)
```

---

## LLM 임팩트 판정 (`impact_judge.py`)

```python
# src/news_impact/impact_judge.py
def build_system_prompt() -> str
def build_news_user_prompt(input: NewsAnalysisInput) -> str
def judgment_to_impact_event(judgment: dict, item: NewsItem) -> ImpactEvent
def detect_prompt_injection(text: str) -> bool
```

각 뉴스 기사에 대해 LLM에게 다음을 판정하도록 요청:

| 판정 항목 | 설명 |
|-----------|------|
| `impact_direction` | 주가 방향성 (`positive` / `negative` / `neutral`) |
| `impact_magnitude` | 임팩트 크기 (0~10) |
| `confidence` | 판정 신뢰도 (0~1) |
| `event_type` | 이벤트 분류 (실적, 공시, 산업이슈 등) |
| `reasoning` | 판정 근거 (한국어) |

`detect_prompt_injection()`: 악의적 뉴스 콘텐츠가 LLM 프롬프트를 조작하는 것을 방지.

---

## LLM 클라이언트 (`llm_client.py`)

```python
# src/news_impact/llm_client.py
class FileLLMResponseCache:     # 로컬 파일 기반 응답 캐시
class LlamaCppClient:           # 로컬 Llama.cpp 모델 클라이언트
```

지원 LLM:

| LLM | 설정 | 비고 |
|-----|------|------|
| 로컬 Gemma/Llama | `llm_provider: "llama_cpp"` | 기본값 · `configs/news_impact.example.json` |
| OpenAI GPT | `llm_provider: "openai"` + `OPENAI_API_KEY` | 선택 · `configs/news_impact.openai.example.json` |

코드 기본값(`LLMConfig.default()`, `--news-impact-llm-config` 미지정 시)도 로컬 gemma입니다.

### LLM 설정 파일 (`configs/news_impact.json`)

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

---

## 스키마 (`schema.py`)

```python
# src/news_impact/schema.py
@dataclass
class NewsItem:
    title: str
    body: str
    published_at: str
    provider: str
    url: str

@dataclass
class ImpactEvent:
    symbol: str
    impact_direction: str
    impact_magnitude: float
    confidence: float
    event_type: str

@dataclass
class RunAudit:
    run_date: str
    total_items: int
    scored_items: int
    failed_items: int
```

---

## 주식 팩터 분류 (`stock_factors/`)

| 모듈 | 역할 |
|------|------|
| `classifier.py` | 뉴스 → 팩터 분류 |
| `factor_taxonomy.py` | 팩터 분류 체계 정의 |
| `freshness.py` | 뉴스 신선도 점수 |
| `impact_rules.py` | 룰 기반 임팩트 규칙 |
| `output_schema.py` | 출력 스키마 |

팩터 예시: `earnings`, `dividends`, `M&A`, `regulatory`, `management_change`, `market_sentiment` 등

---

## 의미 기반 클러스터링 (`semantic_clusterer.py`)

```python
# src/news_impact/semantic_clusterer.py
class SemanticClusterLLM:
    def cluster(self, items: list[NewsItem]) -> list[ClusteredItem]

def assign_semantic_cluster_ids(items, clusterer) -> list
```

동일 이벤트에 대한 여러 기사를 하나의 클러스터로 묶어 중복 집계를 방지한다.

---

## 백테스트 (`backtester.py`, `backtest_snapshots.py`)

뉴스 임팩트 점수의 실제 주가 예측력을 검증:

```python
# src/news_impact/backtester.py
def run_news_impact_backtest(events: list[ImpactEvent], price_data: pd.DataFrame) -> dict
```

이벤트 발생 후 N일 수익률과 임팩트 방향성의 일치율을 측정.

---

## 독립 실행

```bash
# 설정 파일 복사
Copy-Item configs/news_impact.example.json configs/news_impact.json

# 도움말
stock-news-impact --help

# 실행
stock-news-impact \
    --watchlist data/news_impact/watchlist.csv \
    --company-master data/news_impact/company_master.csv \
    --output result/news_impact_report.json
```

---

## 메인 파이프라인과의 통합

뉴스 임팩트 출력은 **표시 전용**이다:

```python
# src/pipeline.py
if news_impact_report:
    pred_df = append_news_impact_context(pred_df, news_impact_report)
```

`--news-impact-report result/news_impact_report.json` 옵션으로 사전 생성된 리포트를 로드하면, `news_impact_*` 컬럼이 `result_detail.csv`에 추가된다.  
이 컬럼들은 `predicted_return`이나 `recommendation`에 영향을 주지 않는다.

---

## 설정 파일

| 파일 | 용도 |
|------|------|
| `configs/news_impact.example.json` | 기본 설정 템플릿 (로컬 Gemma/Llama) |
| `configs/news_impact.gemma.example.json` | 로컬 Gemma/Llama 설정 템플릿 (코드/챗봇 연동 경로) |
| `configs/news_impact.openai.example.json` | OpenAI 설정 템플릿 (선택) |
| `data/news_impact/watchlist.example.csv` | 모니터링 종목 목록 템플릿 |
| `data/news_impact/company_master.example.csv` | 기업 마스터 데이터 템플릿 |
| `data/news_impact/company_aliases.example.csv` | 기업 별칭 템플릿 |
| `data/news_impact/sector_keywords.example.csv` | 섹터 키워드 템플릿 |
| `data/news_impact/company_relationships.example.csv` | 기업 관계도 템플릿 |
| `data/news_impact/global_market_proxy.example.csv` | 글로벌 프록시 템플릿 |

---

## 개선 및 수정 제안

> 우선순위: **P0(문서/계약 불일치) > P1(보안/견고성) > P2(품질)**.

### P0 — LLM 판정 키 문서가 실제 스키마와 불일치

- **문제**: 문서는 판정 항목을 `impact_direction / impact_magnitude / confidence / event_type / reasoning`으로 적었지만, 실제 필수 키는 `event_type, direction, impact_score, impact_strength, confidence, time_horizon, reason, why_may_be_wrong, risk_flags`다(`impact_judge.py:12-22`). `impact_magnitude`·`reasoning`은 존재하지 않으며 `impact_score/impact_strength/reason/why_may_be_wrong`로 분리되어 있다.
- **제안**: 문서 표를 실제 `LLM_REQUIRED_KEYS`에 맞게 정정하고, `judgment_to_impact_event`가 키 누락 시 `ValueError`를 던진다는 계약을 명시.

### P0 — `detect_prompt_injection` 반환 타입·강도 문서 정정

- **문제**: 문서는 `detect_prompt_injection(text) -> bool`로 적었으나 실제 반환은 `tuple[str, ...]`(위험 플래그)다(`impact_judge.py:89-102`). 또한 탐지는 7개 영어 문구 **부분일치 블록리스트**에 불과해 한국어/유니코드/우회 표현에 쉽게 뚫린다. 게다가 위험 플래그를 붙일 뿐 **본문은 그대로 LLM에 전달**된다.
- **제안**: 문서 시그니처 정정. 방어는 (a) 구조적 격리(이미 `<untrusted_article_text>` 태그·시스템 가드 사용 — 좋음)에 더해, (b) 출력 스키마 강제(JSON-only, buy/sell 금지 후처리 검증), (c) 고위험 플래그 시 LLM 스킵 또는 보수 처리.

### P1 — 백테스트/스코어 함수 시그니처 문서 검증

- **문제**: 문서의 `run_news_impact_backtest(events, price_data)` 등 일부 시그니처가 예시 수준이라 실제 `backtester.py`/`scorer.py` 구현과 다를 수 있다.
- **제안**: 공개 함수 시그니처를 코드와 대조해 정정하고, "이벤트 후 N일 수익률 vs 방향 일치율"의 N·집계 방식을 명시.

### P1 — LLM 캐시·비용·재현성

- **문제**: `FileLLMResponseCache`/`LlamaCppClient`가 있으나(문서) 캐시 키(프롬프트 해시) 구성·무효화 정책이 문서화되어 있지 않다. `build_system_prompt()`는 매 호출 `NEWS_IMPACT_LLM_PROMPT.md`를 디스크에서 읽는다(`impact_judge.py:38`).
- **제안**: 캐시 키에 모델명·프롬프트 버전·기사 해시를 포함하고, 프롬프트 로드를 1회 캐싱. `temperature` 등 결정성 설정을 리포트에 기록.

### P2 — 표시 전용 경계 재확인

- **문제**: 메인 파이프라인 통합 시 `news_impact_*` 컬럼은 표시 전용이어야 한다(문서). `feature_selection.DISPLAY_ONLY_CONTEXT_COLUMNS`가 뉴스 계열을 모델 입력에서 제외하는지(특히 `news_impact_*` 접두사) 회귀 테스트로 고정 권장.
- **제안**: "news_impact_ 접두 컬럼은 `select_feature_columns` 결과에 절대 포함되지 않는다"는 단위 테스트 추가.

### P2 — 모듈 수(28개+)의 응집도/진입점 정리

- 독립 실행(`stock-news-impact`)과 메인 통합 두 경로의 의존성 그래프를 문서에 다이어그램으로 추가하면 유지보수가 쉬워진다.
