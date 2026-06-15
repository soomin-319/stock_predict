# 07. 리포트 및 산출물

`src/reports/`와 `src/utils/`는 파이프라인 결과를 CSV/JSON으로 저장하고 관리한다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `output.py` | CSV 출력 빌드 및 저장 |
| `result_formatter.py` | 숫자/텍스트 포맷팅 |
| `pm_report.py` | 포트폴리오 매니저 JSON 리포트 |
| `run_artifacts.py` | 실행별 아티팩트 디렉터리 관리 |
| `report_metadata.py` | 실행 메타데이터 생성 |
| `issue_summary.py` | 뉴스/공시 이슈 요약 (OpenAI) |
| `news_impact_context.py` | 뉴스 임팩트 컨텍스트 병합 |
| `context_policy.py` | 컨텍스트 날짜 유효성 정책 |
| `utils/atomic_files.py` | 원자적 파일 쓰기 |
| `utils/result_cleanup.py` | 결과 파일 정리 |
| `utils/secrets.py` | API 키 마스킹 |

---

## 출력 파일 구조

```
result/
├── latest/                     # 최신 운영 결과 (symbolic link 또는 복사)
│   ├── manifest.json           # 최신 실행 메타데이터
│   ├── csv/
│   │   ├── result_detail.csv   # 전체 예측 상세
│   │   ├── result_simple.csv   # 사용자용 요약 (챗봇)
│   │   ├── result_news.csv     # 뉴스 이벤트
│   │   └── result_disclosure.csv  # 공시 이벤트
│   ├── pm_report.json          # PM 리포트
│   └── pipeline_report.json    # 파이프라인 전체 리포트
└── runs/
    └── <run_id>/               # 실행별 원본 보관
        └── (same structure)
```

샘플(`--input data/sample_ohlcv.csv`) 실행은 `latest/`를 업데이트하지 않는다.

---

## CSV 출력 파일 상세

### `result_detail.csv` — 전체 예측 상세

최신 예측 행과 피처 컨텍스트를 모두 포함. 주요 컬럼:

| 컬럼 | 설명 |
|------|------|
| `Symbol` | 종목 코드 |
| `종목명` | 종목 한글명 |
| `Date` | 기준 날짜 |
| `Close` | 당일 종가 |
| `predicted_return` | 다음날 예상 수익률 (%) |
| `up_probability` | 상승 확률 |
| `signal_score` | 시그널 점수 |
| `recommendation` | 권고 (매수/매도/관망) |
| `confidence_score` | 예측 신뢰도 |
| `uncertainty_score` | 예측 불확실성 |
| `quantile_low/mid/high` | 분위수 예측 수익률 |
| `history_direction_accuracy` | 과거 방향성 정확도 |
| `내일 예상 종가` | 포맷된 예상 종가 (원) |
| `상승확률(%)` | 포맷된 상승 확률 |
| `예측 신뢰도` | 포맷된 신뢰도 텍스트 |
| 피처 컬럼들 | 모든 모델 입력 피처 |

### `result_simple.csv` — 사용자용 요약

챗봇이 사용하는 간소화된 버전. 핵심 컬럼만 포함:

```python
# src/reports/output.py
def build_pipeline_result_simple(pred_df: pd.DataFrame) -> pd.DataFrame
```

### `result_news.csv` — 뉴스 이벤트

| 컬럼 | 설명 |
|------|------|
| `Date` | 뉴스 날짜 |
| `Symbol` | 관련 종목 |
| `source_type` | `"news"` |
| `title` | 뉴스 제목 |
| `published_at` | 발행 시각 |
| `provider` | 미디어 제공자 |
| `record_type` | `event` / `summary` / `no_data` |
| `collection_status` | `completed` / `failed` / `empty` |

### `result_disclosure.csv` — 공시 이벤트

`result_news.csv`와 동일 구조, `source_type="disclosure"`.

---

## 아티팩트 관리 (`run_artifacts.py`)

```python
# src/reports/run_artifacts.py
class RunArtifactManager:
    def __init__(self, base_dir: Path, metadata: dict)
    def path(self, relative: str) -> Path
    def write_csv(self, relative: str, df: pd.DataFrame) -> Path
    def write_json(self, relative: str, data: dict) -> Path
    def finalize(self) -> dict     # latest/ 링크 업데이트, manifest 저장
```

각 실행마다 `result/runs/<run_id>/` 디렉터리를 생성하고, 완료 시 `result/latest/`로 링크.

---

## 실행 메타데이터 (`report_metadata.py`)

```python
# src/reports/report_metadata.py
def generate_run_id() -> str        # UUID 기반 실행 ID
def build_report_metadata(run_id, environment, data_mode, ...) -> dict
```

```json
{
    "run_id": "20250615_143022_abc123",
    "environment": "production",
    "data_mode": "real",
    "input_as_of_date": "2025-06-13",
    "prediction_for_date": "2025-06-16",
    "context_as_of_date": "2025-06-13",
    "status": "ok"
}
```

- `environment`: 샘플 입력이면 `"smoke"`, 실데이터면 `"production"`
- `prediction_for_date`: `input_as_of_date` 다음 영업일

---

## PM 리포트 (`pm_report.py`)

```python
# src/reports/pm_report.py
def build_pm_report(pred_df: pd.DataFrame, report: dict) -> dict
def save_pm_report(pm_report: dict, path: Path) -> None
```

포트폴리오 매니저용 요약 JSON:

```json
{
    "run_id": "...",
    "prediction_for_date": "2025-06-16",
    "summary": {
        "total_symbols": 200,
        "buy_count": 12,
        "sell_count": 3,
        "hold_count": 185
    },
    "top_picks": [...],
    "backtest_summary": {...}
}
```

---

## 이슈 요약 생성 (`issue_summary.py`)

```python
# src/reports/issue_summary.py
def append_issue_summary_columns(
    pred_df,
    context_raw_df,
    openai_api_key=None,
    openai_model=None,
    summarize_symbols=None,
    summary_n_jobs=1,
) -> pd.DataFrame
```

OpenAI API를 사용하여 뉴스/공시 이벤트를 한국어로 요약:
- `--openai-api-key` 또는 `OPENAI_API_KEY` 환경변수 필요
- `--issue-summary-symbols`로 요약 종목 제한 가능
- 결과는 `뉴스 요약`, `공시 요약` 컬럼으로 추가 (표시 전용)

---

## 뉴스 임팩트 컨텍스트 병합 (`news_impact_context.py`)

```python
# src/reports/news_impact_context.py
def append_news_impact_context(pred_df, news_impact_report: str) -> pd.DataFrame
def append_generated_news_impact_context(pred_df, context_raw_df) -> pd.DataFrame
```

외부 `stock-news-impact` JSON 리포트를 `news_impact_*` 컬럼으로 병합. 표시 전용이며 예측 결과에 영향 없음.

---

## 컨텍스트 날짜 정책 (`context_policy.py`)

```python
# src/reports/context_policy.py
def evaluate_context_policy(input_as_of_date, context_date) -> PolicyResult
```

컨텍스트(뉴스/공시) 날짜가 `input_as_of_date`와 너무 멀리 벗어난 경우 필터링. 지나치게 오래된 뉴스가 결과에 포함되지 않도록 보호.

---

## 파일 안전 처리 (`utils/`)

```python
# src/utils/atomic_files.py
def atomic_write_text(path: Path, content: str) -> None

# src/utils/secrets.py
def redact_text(text: str) -> str      # API 키 등 마스킹
def redact_argv(argv: list) -> list    # CLI 인수 마스킹

# src/utils/result_cleanup.py
def cleanup_old_runs(base_dir: Path, keep_n: int = 10) -> None
```

- `atomic_write_text()`: 임시 파일 → 이름 변경 방식으로 부분 쓰기 방지
- `redact_text()`: 로그/에러 메시지에서 API 키 자동 마스킹

---

## CSV 인코딩

모든 CSV 파일은 `utf-8-sig` (UTF-8 with BOM) 인코딩으로 저장.  
Windows Excel에서 한글이 깨지지 않도록 하기 위함.

---

## 개선 및 수정 제안

> 우선순위: **P0(정확성) > P1(견고성) > P2(품질/문서)**.

### P1 — 입력 CSV 인코딩 일관성

- **문제**: 출력은 `utf-8-sig`로 저장하지만(`output.py:59`), `build_combined_symbol_results`는 외부 요약 CSV를 `pd.read_csv(summary_csv)` 기본 인코딩으로 읽는다(`output.py:194`). 이전 단계가 저장한 `utf-8-sig` 파일을 되읽을 때 BOM 헤더로 `Symbol` 매칭이 어긋날 수 있다.
- **제안**: 프로젝트의 모든 `read_csv`를 `encoding="utf-8-sig"`로 통일(입력 데이터 로더 포함, `02_data.md` 참고).

### P1 — `drop_empty_detail_columns`의 컬럼 변동성

- **문제**: 실행마다 "전부 비어있는" 선택 컬럼을 드롭하므로(`output.py:118-165`), `result_detail.csv`의 **스키마가 실행마다 달라진다**. 다운스트림(엑셀 매크로, 챗봇 파서, BI)이 컬럼 존재를 가정하면 깨진다.
- **제안**: 컬럼은 항상 유지하고 값만 비우거나, 안정 스키마 버전을 manifest에 기록. 드롭 동작은 옵션화.

### P1 — `latest/` 링크 원자성·동시성

- **문제**: `RunArtifactManager.finalize()`가 `result/latest/`를 갱신할 때, 챗봇(`realtime_close_betting`)이 동시에 `result/latest/csv/result_simple.csv`를 읽으면 부분 갱신 상태를 볼 수 있다. Windows에서는 symlink 권한·교체 시맨틱이 제한적이다.
- **제안**: `latest`를 디렉터리 단위로 원자 교체(임시 디렉터리 생성 후 rename)하거나, `manifest.json`의 `run_id`를 단일 진실원으로 두고 reader가 run 디렉터리를 직접 참조.

### P2 — 메타데이터 영업일 계산 정확성

- **문제**: `prediction_for_date = input_as_of_date 다음 영업일`(문서)인데, 단순 +1영업일 로직은 **한국 공휴일/임시휴장**을 반영하지 못할 수 있다(`report_metadata.py`). 토·일만 건너뛰면 설/추석 연휴에 잘못된 날짜가 나온다.
- **제안**: KRX 거래일 캘린더(또는 `pandas_market_calendars`/보유 중인 거래일 인덱스)로 다음 거래일 산출.

### P2 — `pm_report`/`result_simple` 스키마 계약 문서화

- **문제**: 챗봇·PM 리포트가 의존하는 `result_simple.csv` 컬럼 계약이 코드(`build_pipeline_result_simple` → `format_result_simple`)에 흩어져 있다.
- **제안**: 필수 컬럼·타입을 문서/스키마(예: pandera)로 고정해 회귀를 테스트.

### P2 — 결과 정리(`cleanup_old_runs`) 안전장치

- **문제**: `keep_n` 기준으로 오래된 run을 삭제하는데(`utils/result_cleanup.py`), 진행 중이거나 `latest`가 가리키는 run을 보호하는지 문서에 불명확.
- **제안**: `latest`가 참조하는 `run_id`와 최근 N개는 항상 보존하도록 명시·테스트.

### P2 — 이슈 요약 LLM 호출 비용/실패 처리

- **문제**: `issue_summary.py`(661줄)는 OpenAI 호출로 종목별 한국어 요약을 생성한다. 대량 종목·실패·레이트리밋 시 비용/지연이 크다.
- **제안**: 요약 대상 기본 상한(상위 N종목), 캐시(뉴스 임팩트 모듈의 `FileLLMResponseCache` 재사용), 실패 시 부분 결과 보존 정책을 문서화.
