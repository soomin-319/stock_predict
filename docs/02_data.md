# 02. 데이터 — 로드·정제·유니버스·수집

`src/data/`는 OHLCV 입력 로드, 정제, 유니버스 필터, 실데이터 수집(yfinance), KRX 종목명 매핑,
투자자/공시/뉴스 컨텍스트 수집을 담당한다. 모든 입력은 리서치·운영 보조용이며, 뉴스/공시는
표시·검토용 컨텍스트로 기대수익률·추천을 바꾸지 않는다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `loaders.py` | OHLCV CSV 로드 및 필수 컬럼 검증 |
| `cleaners.py` | OHLCV 정제, 중복 제거, 품질 플래그 |
| `universe.py` | 유니버스 심볼 로드/필터 (기본 KOSPI200) |
| `krx_universe.py` | KRX 티커↔종목명 매핑, 이름 기반 후보 검색 |
| `fetch_real_data.py` | yfinance 실데이터 다운로드/저장/증분 추가 |
| `cli_refresh.py` | 수집 심볼/시작일 해석(증분 갱신 보조) |
| `investor_context.py` | 투자자 수급·DART 공시·Naver 뉴스 컨텍스트 |

---

## OHLCV 로드 (`loaders.py`)

```python
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

def load_ohlcv_csv(path, symbol=None) -> pd.DataFrame
```

- `utf-8-sig`로 읽고 필수 컬럼 누락 시 `ValueError`.
- `Symbol` 컬럼이 없으면 인자 `symbol`을, 그것도 없으면 `"UNKNOWN"`을 채운다.
- 중복 행은 제거하지 않고 보존한다 — 정제 단계(`clean_ohlcv`)가 결정적으로 최다 거래량 행을 선택한다.

## 정제 (`cleaners.py`)

```python
def clean_ohlcv(df) -> pd.DataFrame
```

정제 규칙:

- `Date`/OHLCV/`Symbol` 결측 행 제거, OHLCV를 숫자로 강제 변환.
- 가격 양수, 거래량 0 이상, `High >= max(O,C,L)`, `Low <= min(O,C,H)` 무결성 검사.
- `(Date, Symbol)` 중복은 거래량이 가장 큰 행을 유지(`keep="last"`, 입력 순서 타이브레이크)하여 결정적으로 선택.
- 품질 플래그 추가: `is_zero_volume`(거래량 0), `is_extreme_return`(전일 대비 일간 변동률 절대값 40% 초과).

## 유니버스 (`universe.py`)

```python
DEFAULT_UNIVERSE_CSV = data/kospi200_symbol_name_map.csv

def load_universe_symbols_list(path) -> list[str]   # 순서 보존 + 중복 제거
def load_universe_symbols(path) -> set[str]          # 집합(순서 비보존)
def load_default_universe_symbols() -> list[str]
def filter_by_universe(df, universe) -> pd.DataFrame
```

- 유니버스 CSV는 `Symbol` 컬럼이 필수이며 비어 있으면 `ValueError`.
- `load_universe_symbols_list`는 `dict.fromkeys`로 입력 순서를 보존한다.

## KRX 종목명 매핑 (`krx_universe.py`)

`data/krx_symbol_name_map.csv`(+ `kospi200_symbol_name_map.csv` 폴백)에서 `Ticker`/`Symbol`/`Name`/`Market`을 읽는다.

| 함수 | 역할 |
|------|------|
| `get_symbol_name_map(symbols)` | yfinance 스타일 Symbol → 한글명. 미해결 심볼은 자기 자신으로 채움 |
| `get_provider_symbol_for_ticker(ticker)` | 6자리 KRX 티커 → 저장소 매핑 provider 심볼(`.KS`/`.KQ`) |
| `find_symbol_candidates_by_name(query, limit)` | 종목명 유사도 검색. 정규화 후 `SequenceMatcher` 점수 0.45 이상 후보 반환 |

- CSV 로드는 `lru_cache`로 캐시한다. `Ticker`는 6자리 zero-pad.
- 이름 매칭 점수: 완전일치 1.0, 부분포함 0.9, 그 외 `SequenceMatcher.ratio()`. 챗봇 종목명 검색에 사용된다.

## 실데이터 수집 (`fetch_real_data.py`)

```python
DEFAULT_REAL_START_DATE = "2020-01-01"
MAX_DOWNLOAD_ATTEMPTS = 3

def fetch_real_ohlcv(symbols, start=..., end=None) -> pd.DataFrame
def save_real_ohlcv_csv(path, symbols, start, end) -> Path
def append_real_ohlcv_csv(path, symbols, start, end) -> Path
def normalize_user_symbols(symbol_inputs) -> list[str]
def get_last_fetch_coverage() -> dict
```

- yfinance는 선택 의존성이며, 없으면 명확한 `RuntimeError`로 안내한다.
- **심볼 정규화**: 6자리 숫자는 `get_provider_symbol_for_ticker`로 provider 심볼을 찾고, 실패 시 `.KS` 접미사를 붙인다.
- **폴백**: `.KS` 실패 시 `.KQ`(또는 반대)로 재시도(`_provider_symbol_candidates`).
- **재시도**: 심볼당 최대 3회, 지수 백오프(`2 ** (attempt-1)`초).
- **MultiIndex/중복 컬럼 방어**: yfinance 버전 차이로 발생하는 MultiIndex 가격 레벨 탐색과 중복 컬럼 제거를 처리.
- **동시성**: 심볼 수에 따라 최대 8개 스레드(`ThreadPoolExecutor`).
- **중복 제거**: `(Date, Symbol)` 기준 `keep="last"`로 dedupe하여 downstream pandas 연산의 `InvalidIndexError`를 방지.
- **커버리지**: `_LAST_FETCH_COVERAGE`에 요청/성공/실패/폴백/재시도 상세를 저장(`get_last_fetch_coverage()`로 조회).
- **선택 컬럼 보존**: 기존 CSV의 투자자/컨텍스트 등 추가 컬럼은 `(Date, Symbol)` 머지로 보존(`_preserve_existing_optional_columns`).
- 모든 CSV는 `utf-8-sig`로 저장.

## CLI 수집 보조 (`cli_refresh.py`)

```python
def resolve_fetch_symbols(real_symbols, universe_csv, input_csv, *, universe_loader=..., fallback_loader=...) -> list[str]
def resolve_incremental_fetch_start(input_csv, requested_start) -> str
def fallback_symbols_from_input_or_default(input_csv, limit=0) -> list[str]
```

- 수집 심볼 우선순위: `--real-symbols` > `--universe-csv` > 저장소 기본 유니버스(KOSPI200 200종목).
- `resolve_incremental_fetch_start`: 기존 CSV의 최신 `Date` + 1일과 요청 시작일 중 더 늦은 날짜를 반환(증분 다운로드).

## 투자자 컨텍스트 (`investor_context.py`)

```python
@dataclass
class InvestorContextConfig:
    enabled: bool = False
    enable_disclosure: bool = True
    dart_api_key: str | None = None
    dart_corp_map_csv: str | None = None
    raw_event_n_jobs: int = 4

def add_investor_context_with_coverage(df, cfg) -> tuple[pd.DataFrame, dict]
def collect_context_raw_events(symbols, start, end, ...) -> pd.DataFrame | tuple[pd.DataFrame, dict]
```

두 가지 산출물을 만든다.

1. **수치 컨텍스트 컬럼** (`add_investor_context_with_coverage`): `foreign_net_buy`, `institution_net_buy`,
   `disclosure_score`, `news_sentiment`, `news_relevance_score`, `news_impact_score`, `news_article_count`.
   누락 컬럼은 0.0으로 채운다.
   - **DART 공시 점수**: `dart_api_key`가 있으면 OpenDART `list.json`을 호출해 보고서명에 긍정 키워드
     (수주/계약/실적/합병 등)가 있으면 가점하는 단순 점수(0~1)를 만든다. `dart_corp_map_csv`로 종목→corp_code 매핑.
2. **원본 이벤트 프레임** (`collect_context_raw_events`): Naver 뉴스 검색 API + DART 공시를 표시용 raw 이벤트로
   수집한다. 종목명 기반 한국어 쿼리(`주가/실적/공시/계약/전망`)를 사용하고, KST로 발행시각을 정규화하며,
   제목/URL/raw_id 기준 중복을 제거한다. `return_status=True`면 수집 상태(success/no_events/partial_failure/
   collection_failed)와 실패 심볼/오류 유형을 함께 반환한다.

> 뉴스/공시 컨텍스트는 모두 표시·검토용이다. `disclosure_score`, `news_*` 컬럼과 합성 점수
> `investor_event_score`는 `feature_selection.DISPLAY_ONLY_CONTEXT_COLUMNS`로 모델 입력에서 제외된다([03](03_features.md) 참조).

---

## 개선 및 수정 제안

> 우선순위: **P0(정확성/버그) > P1(견고성/기능 공백) > P2(품질/결정성)**.

### P1 — 투자자 수급(`_fetch_flow`) 외부 소스 미연결 상태 명시

- **상태**: 반영됨. `_fetch_flow`는 외부 수급 소스가 아직 연결되지 않았음을
  `status="not_configured"`, `source="input_csv_only"`로 `investor_context_coverage.flow`에 노출한다.
- **남은 공백**: 실제 수급 소스(KRX/데이터벤더)는 아직 연결되어 있지 않다. 따라서 외국인/기관 순매수
  (`foreign_net_buy`, `institution_net_buy`)는 **입력 CSV가 직접 제공하지 않는 한 항상 0**이다.
  이 값에 의존하는 피처(`smart_money_*`, `foreign/institution_buy_*`, 고확신 순매수 플래그)와 이벤트 부스트가
  중립으로 고정된다.
- **다음 제안**: 실제 수급 소스(예: KRX/데이터벤더)를 연결한다.

### P2 — 수집 심볼 순서 비결정성

- **상태**: 반영됨. `resolve_fetch_symbols`의 기본 `universe_loader`와 파이프라인 위임 로더를
  `load_universe_symbols_list`로 교체해 다운로드 순서·로그·부분 실패 재현을 결정적으로 만든다.
