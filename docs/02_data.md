# 02. 데이터 레이어

`src/data/` 패키지는 OHLCV 데이터 로딩, 정제, 실시간 갱신, 종목 유니버스 관리, 투자자 컨텍스트 수집을 담당한다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `loaders.py` | CSV → DataFrame 로딩 |
| `cleaners.py` | OHLCV 정제 (중복 제거, 타입 변환) |
| `fetch_real_data.py` | yfinance를 통한 실시간 OHLCV 다운로드 |
| `cli_refresh.py` | CLI 갱신 로직 (전체/증분/심볼 추가) |
| `universe.py` | 유니버스 필터링 및 로딩 |
| `krx_universe.py` | KRX/KOSPI200 심볼-이름 매핑 |
| `investor_context.py` | 외국인/기관 매매, 공시, 뉴스 컨텍스트 수집 |

---

## 데이터 로딩 (`loaders.py`)

```python
# src/data/loaders.py
def load_ohlcv_csv(path: str, symbol: str | None = None) -> pd.DataFrame
```

- 필수 컬럼: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- 선택 컬럼: `Symbol`, `foreign_net_buy`, `institution_net_buy`, `market_type` 등
- 인코딩: `utf-8-sig`로 읽어 UTF-8 / UTF-8-BOM 모두 지원
- `Symbol` 컬럼이 없는 파일은 `symbol=` 값 또는 `"UNKNOWN"`을 사용
- 중복 행은 로더에서 제거하지 않고 `clean_ohlcv()`의 결정적 정책으로 전달

---

## 데이터 정제 (`cleaners.py`)

```python
# src/data/cleaners.py
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame
```

- `Date` → datetime 변환
- `Symbol` 없으면 `"UNKNOWN"` 채움
- 0/음수 Close 제거
- 중복 `(Date, Symbol)`은 `Volume` 최대 행 선택, 동률이면 마지막 입력 행 선택
- 숫자 컬럼 강제 변환
- `Volume=0` 행은 유지하고 `is_zero_volume=True`로 표시
- 종목별 일간 수익률 절댓값이 40%를 초과하면 `is_extreme_return=True`로 표시
- 품질 플래그 행은 정제 이력에는 유지하지만 모델 입력에서는 제외

---

## 실시간 데이터 갱신 (`fetch_real_data.py`)

```python
# src/data/fetch_real_data.py
def save_real_ohlcv_csv(path, symbols, start=DEFAULT_REAL_START_DATE, end=None) -> Path
def append_real_ohlcv_csv(path, symbols, start=DEFAULT_REAL_START_DATE, end=None) -> Path
def normalize_user_symbols(symbol_inputs: Iterable[str]) -> list[str]
```

- yfinance API를 사용하여 KOSPI/KOSDAQ 종목 데이터 수집
- `normalize_user_symbols()`: KRX 매핑을 우선 사용하여 `005930` → `.KS`, `247540` → `.KQ` 변환
- 미등록 6자리 코드는 `.KS` 우선이며, 다운로드 실패 시 `.KQ`를 재시도
- 수정주가(`auto_adjust=True`) 사용
- 각 시장 후보를 최대 3회, 1초 기준 지수 백오프로 재시도
- `save_real_ohlcv_csv()`: 전체 재다운로드
- `append_real_ohlcv_csv()`: 증분 추가 (기존 최신 날짜 이후만)
- 저장 인코딩: `utf-8-sig`
- 기본 시작일: `DEFAULT_REAL_START_DATE = "2020-01-01"` (`--real-start`로 변경 가능)

마지막 수집 결과는 `get_last_fetch_coverage()`로 조회한다. 요청/성공/실패
수, 성공률, 실패 심볼, 시장 폴백, 재시도 횟수, 실제 사용 심볼을 포함하며
CLI 갱신 실행 시 `pipeline_report.json`의 `data_fetch_coverage`에 기록된다.

---

## CLI 갱신 로직 (`cli_refresh.py`)

```python
# src/data/cli_refresh.py
def resolve_fetch_symbols(...) -> list[str]
def resolve_incremental_fetch_start(input_csv: str, requested_start: str) -> str
def fallback_symbols_from_input_or_default(input_csv: str, limit: int = 0) -> list[str]
```

- `--fetch-real`: 전체 재다운로드 → `save_real_ohlcv_csv()`
- `--auto-refresh-real`: 증분 갱신 → `append_real_ohlcv_csv()`
- `--add-symbols`: 특정 종목 추가
- 심볼 미지정 시 → KOSPI200 200개 기본 사용

---

## 유니버스 관리 (`universe.py`, `krx_universe.py`)

```python
# src/data/universe.py
def load_universe_symbols(universe_csv: str) -> set[str]
def filter_by_universe(df: pd.DataFrame, universe: set[str]) -> pd.DataFrame

# src/data/krx_universe.py
def get_symbol_name_map(symbols: list[str]) -> dict[str, str]
def find_symbol_candidates_by_name(name: str) -> list[tuple[str, str]]
```

- 기본 유니버스: `data/kospi200_symbol_name_map.csv` (200개 종목)
- 전체 KRX: `data/krx_symbol_name_map.csv`
- `--universe-csv` 옵션으로 커스텀 유니버스 지정 가능
- `find_symbol_candidates_by_name()`: 챗봇에서 한글 종목명 → 심볼 검색에 사용
- 종목명 검색은 공백 제거·소문자 정규화 후 정확 일치, 부분 일치, 유사도
  순으로 점수를 계산하며 동명 종목은 티커별 후보로 유지

---

## 투자자 컨텍스트 (`investor_context.py`)

```python
# src/data/investor_context.py
@dataclass
class InvestorContextConfig:
    enabled: bool
    enable_disclosure: bool
    dart_api_key: str | None
    dart_corp_map_csv: str | None
    raw_event_n_jobs: int

def add_investor_context_with_coverage(df, cfg) -> tuple[pd.DataFrame, dict]
def collect_context_raw_events(symbols, start, end, ...) -> pd.DataFrame | tuple
```

`--fetch-investor-context` 옵션 활성화 시 실행되며, 3가지 데이터를 수집한다:

| 데이터 | 소스 | 컬럼 |
|--------|------|-------|
| 투자자 매매 흐름 | Pykrx / 직접 수집 | `foreign_net_buy`, `institution_net_buy`, `individual_net_buy` |
| 공시 정보 | DART API | `disclosure_score`, `공시 요약` |
| 뉴스 | Naver 뉴스 API | `뉴스 요약` |

- **표시 전용**: 수집된 뉴스·공시는 예측 결과에 영향을 주지 않음 (디스플레이 컨텍스트)
- 커버리지 비율(`investor_coverage_ratio`)이 기준 미달 시 `coverage_gate_status = "halt"`

---

## 데이터 흐름

```
data/real_ohlcv.csv
        │
   load_ohlcv_csv()
        │
   clean_ohlcv()
        │
   filter_by_universe()  ← universe_csv 있을 때
        │
   add_investor_context_with_coverage()  ← --fetch-investor-context
        │
   → src/features/ 에서 피처 빌드
```

---

## 데이터 파일 위치

| 파일 | 내용 |
|------|------|
| `data/real_ohlcv.csv` | 실운영 OHLCV (기본 입력) |
| `data/sample_ohlcv.csv` | 테스트용 샘플 (네트워크 불필요) |
| `data/kospi200_symbol_name_map.csv` | KOSPI200 심볼-이름 매핑 |
| `data/krx_symbol_name_map.csv` | 전체 KRX 심볼-이름 매핑 |
| `data/news_impact/` | 뉴스 임팩트 모듈용 예시 파일들 |
