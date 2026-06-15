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
def load_ohlcv_csv(path: str) -> pd.DataFrame
```

- 필수 컬럼: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- 선택 컬럼: `Symbol`, `foreign_net_buy`, `institution_net_buy`, `market_type` 등
- 인코딩: UTF-8 / UTF-8-BOM 자동 감지

---

## 데이터 정제 (`cleaners.py`)

```python
# src/data/cleaners.py
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame
```

- `Date` → datetime 변환
- `Symbol` 없으면 `"UNKNOWN"` 채움
- 0/음수 Close 제거
- 중복 `(Date, Symbol)` 행 제거
- 숫자 컬럼 강제 변환

---

## 실시간 데이터 갱신 (`fetch_real_data.py`)

```python
# src/data/fetch_real_data.py
def save_real_ohlcv_csv(path: str, symbols: list[str], start: str) -> None
def append_real_ohlcv_csv(path: str, symbols: list[str], start: str) -> None
def normalize_user_symbols(symbols: list[str]) -> list[str]
```

- yfinance API를 사용하여 KOSPI/KOSDAQ 종목 데이터 수집
- `normalize_user_symbols()`: `005930` → `005930.KS` 자동 변환
- `save_real_ohlcv_csv()`: 전체 재다운로드
- `append_real_ohlcv_csv()`: 증분 추가 (기존 최신 날짜 이후만)

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

---

## 개선 및 수정 제안

> 우선순위: **P0(정확성/버그) > P1(견고성/누수) > P2(성능/품질/문서)**.

### P0 — KOSDAQ 종목이 `.KS`로 강제 변환됨

- **문제**: `_to_yfinance_symbol()`은 6자리 숫자 심볼을 무조건 `.KS`(KOSPI)로 변환한다(`src/data/fetch_real_data.py:43-44`). KOSDAQ 종목(예: `247540` 에코프로비엠)은 `.KQ`여야 하므로 yfinance에서 **빈 데이터**가 반환되거나 잘못된 종목이 매칭된다. 문서(`02_data.md`)는 "KOSPI/KOSDAQ 수집"이라고 설명하지만 코드는 KOSDAQ를 처리하지 못한다.
- **제안**: KOSPI200/KRX 매핑 CSV에 `market`(KOSPI/KOSDAQ) 컬럼을 추가하거나, `.KS` 실패 시 `.KQ`로 폴백 재시도. 최소한 두 접미사를 모두 시도하고 성공한 쪽을 사용.

### P0 — `load_ohlcv_csv`의 BOM 미처리 (문서와 코드 불일치)

- **문제**: 문서는 "UTF-8 / UTF-8-BOM 자동 감지"라고 하지만, `load_ohlcv_csv`는 `pd.read_csv(path)`를 기본 인코딩으로 호출한다(`src/data/loaders.py:12`). BOM이 있는 CSV는 첫 헤더가 `﻿Date`가 되어 `REQUIRED_COLUMNS` 검사에서 `Missing required columns: ['Date']`로 실패한다.
- **제안**: `pd.read_csv(path, encoding="utf-8-sig")`로 변경(BOM 유무 모두 안전). 출력 CSV는 이미 `utf-8-sig`로 저장(`reports/output.py`)되므로 일관성도 개선된다.

### P1 — 미조정(unadjusted) 가격 사용으로 인한 가짜 급등락

- **문제**: 실시간/외부 다운로드 모두 `auto_adjust=False`(`fetch_real_data.py:94`, `external_features.py:38`). 액면분할·유상증자·배당락 시 종가가 급변해 `daily_return`/`target_log_return`에 ±수십% 가짜 신호가 생기고, 모델이 이를 학습한다.
- **제안**: `auto_adjust=True`(수정주가) 사용 또는 분할 이벤트 보정 로직 추가. 최소한 `clean_ohlcv`에서 **단일일 ±40% 초과 같은 극단 수익률을 이상치로 플래그/제거**.

### P1 — `clean_ohlcv` 이상치/중복 정책 보강

- **문제**: 현재 정제는 음수/0 제거와 OHLC 정합성만 검사한다(`src/data/cleaners.py`). (a) 극단 수익률 이상치 미처리, (b) 중복 `(Date, Symbol)`은 `keep="last"`로 **임의 선택**(거래량 많은 행이 아닌 마지막 행).
- **제안**: 일중 수익률 클리핑/윈저화 옵션, 중복 시 `Volume` 최대 행 우선 선택, 거래정지(Volume=0) 구간 처리 정책을 명시.

### P1 — yfinance 호출의 재시도/레이트리밋 부재

- **문제**: `fetch_real_ohlcv`는 실패 시 경고만 남기고 재시도하지 않는다. yfinance는 일시적 429/타임아웃이 잦아, 대량(200종목) 수집 시 일부가 조용히 누락된다(`empty_symbols` 경고만).
- **제안**: 지수 백오프 재시도(예: tenacity), 실패 심볼 목록을 `pipeline_report.json`에 노출, 수집 성공률을 커버리지 지표로 승격.

### P2 — 하드코딩된 기본값/계약 명시

- `fetch_real_ohlcv(start="2020-01-01")` 기본 시작일이 코드에 하드코딩(`fetch_real_data.py:122`). 설정/CLI로 노출 권장.
- `load_ohlcv_csv(symbol=...)` 인자는 문서에 없음 → 문서에 추가하거나 제거.
- `find_symbol_candidates_by_name()`의 동명/유사명 처리(부분일치, 공백·특수문자 정규화) 동작을 문서화.
