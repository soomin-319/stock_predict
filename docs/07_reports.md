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
+-- latest_manifest.json        # latest production run pointer
+-- latest/                     # compatibility copy of latest production output
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

Sample runs (`--input data/sample_ohlcv.csv`) do not update `latest/` or `latest_manifest.json`.

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
    max_llm_symbols=20,
    llm_cache_dir=None,
) -> pd.DataFrame
```

OpenAI API를 사용하여 뉴스/공시 이벤트를 한국어로 요약:
- `--openai-api-key` 또는 `OPENAI_API_KEY` 환경변수 필요
- `--issue-summary-symbols`로 요약 종목 제한 가능
- Default LLM call budget is 20 symbols; rows outside the budget use rule-based summaries.
- Set `llm_cache_dir` to reuse file-cached LLM summaries for identical model/event inputs.
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

## Current guardrails and remaining improvements

> Priority: **P0(correctness) > P1(robustness) > P2(operations/docs)**.

### Applied guardrails

- `drop_empty_detail_columns()` keeps optional columns by default, so `result_detail.csv` schema is stable. Legacy callers can still opt into pruning with `prune_empty_optional=True`.
- Only production/real runs promote to `latest/`. Sample and smoke runs stay under `result/runs/<run_id>/` and do not change `latest_manifest.json`.
- Chatbot readers prefer `latest_manifest.json` and read `runs/<run_id>/...` directly.
- CSV artifact entries in manifest include `row_count`, `columns`, `schema_kind`, and `schema_version`.
- `pm_report` exposes `PM_REPORT_REQUIRED_FIELDS` and `validate_pm_report_schema()` for contract checks.
- `cleanup_runs()` preserves the run referenced by `latest_manifest.json` or `latest/manifest.json`.
- Issue-summary LLM calls are capped at 20 symbols by default. When `llm_cache_dir` is set, file cache is reused. LLM failure or budget overflow never changes prediction values; display-only summaries fall back to rules.

### Remaining improvements

- KRX business-day logic includes built-in 2025-2026 holidays. For longer operation, move to a KRX calendar package or managed trading-day table.
- When `result_simple.csv` or `pm_report.json` contracts change, bump schema version and update chatbot/BI consumer tests together.
- Treat `latest/` as compatibility copy. New readers should prefer `latest_manifest.json` -> `runs/<run_id>/...`.
