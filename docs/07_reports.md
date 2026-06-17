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

## 게시 산출물 (`published/`)

`stock-predict-publish`(`src/ops/publish_predictions.py`, [01_pipeline.md](01_pipeline.md) 참고)가 운영 run을 GitHub 추적 폴더 `published/`로 복사한다. `result/`는 gitignore 대상이지만 `published/`는 GitHub에 커밋되어 Colab/챗봇이 읽는 **기준데이터(baseline)** 가 된다.

```
published/
├── latest/                     # 최신 게시본 (Colab 기본 읽기 대상)
│   ├── csv/
│   │   ├── result_simple.csv
│   │   ├── result_detail.csv
│   │   ├── result_news.csv
│   │   └── result_disclosure.csv
│   ├── manifest.json
│   ├── pipeline_report.json
│   └── publish_meta.json       # 게시 메타데이터
├── history/
│   └── <거래일>/               # 거래일별 스냅샷 (latest와 동일 구조)
└── index.json                  # 가용 날짜·메타 인덱스 (latest 포인터 포함)
```

`publish_meta.json` (게시 산출 메타, `src/ops/published_store.py:PublishMeta`):

```json
{
    "generated_at_kst": "2026-06-17T18:05:00+09:00",
    "trading_date": "2026-06-17",
    "news_mode": "gemma",
    "source_run_id": "...",
    "symbol_count": 200,
    "git": {"commit": "abc1234", "branch": "main"}
}
```

- `news_mode`: 게시 시 요청된 모드(`gemma`/`rule_based`)를 그대로 기록(표시용).
- `git`: 게시 시점의 실제 HEAD 커밋·브랜치(베스트에포트, 미확인 시 `null`).
- 동일 거래일 재게시는 `index.json` 항목과 `history/<거래일>/`을 덮어쓴다.

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
def generate_run_id() -> str        # 타임스탬프(UTC) + UUID 접미사 실행 ID
def build_report_metadata(run_id, environment, data_mode, ...) -> dict
```

```json
{
    "run_id": "20250615T143022Z_abc12345",
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
def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None

# src/utils/secrets.py
def redact_text(text: object, secret_values: Iterable[object] = ()) -> str    # API 키 등 마스킹
def redact_argv(argv: Iterable[object], secret_values: Iterable[object] = ()) -> list[str]

# src/utils/result_cleanup.py
@dataclass
class RetentionPolicy:                  # successful_run_count=10, *_days 보존 기준
    ...
def cleanup_result_artifacts(result_root: Path, policy: RetentionPolicy, now: datetime | None = None) -> dict
```

- `atomic_write_text()`: 임시 파일 → 이름 변경 방식으로 부분 쓰기 방지
- `redact_text()`: 로그/에러 메시지에서 API 키 자동 마스킹

---

## CSV 인코딩

모든 CSV 파일은 `utf-8-sig` (UTF-8 with BOM) 인코딩으로 저장.  
Windows Excel에서 한글이 깨지지 않도록 하기 위함.

---

## 개선 및 수정 진행 현황

> 우선순위: **P0(정확성) > P1(견고성) > P2(품질/문서)**. 기준일: 2026-06-17.

### 해결됨 — P1 입력 CSV 인코딩 일관성

- 출력 저장(`output.py:59`, `utf-8-sig`)에 이어 되읽기 경로도 통일했다. `build_combined_symbol_results`는 요약 CSV를 `pd.read_csv(summary_csv, encoding="utf-8-sig")`로 읽고(`output.py:200`), 입력 로더도 `load_ohlcv_csv(..., encoding="utf-8-sig")`로 BOM을 처리한다(`loaders.py:12`, `02_data.md` 참고).

### 해결됨 — P1 `drop_empty_detail_columns` 스키마 안정화

- `drop_empty_detail_columns`는 기본값 `prune_empty_optional=False`로 **detail 스키마를 그대로 유지**하고, 비어있는 선택 컬럼 드롭은 레거시 호출자용 옵션으로만 동작한다(`output.py:118-124`). 실행마다 컬럼이 달라지던 문제가 해소됐다.

### 해결됨 — P1 `latest/` 갱신 원자성

- `RunArtifactManager.finalize()`는 임시 디렉터리에 복사 후 rename으로 `result/latest/`를 **디렉터리 단위 원자 교체**하고(`run_artifacts.py:69-86`), `latest_manifest.json` 포인터를 단일 진실원으로 둔다(`resolve_latest_run_dir`). reader는 포인터의 `run_id`로 run 디렉터리를 참조할 수 있다.

### 해결됨 — P2 메타데이터 영업일 계산 정확성

- `next_krx_business_day()`가 토·일에 더해 `KOREA_MARKET_HOLIDAYS`(2025–2026 KRX 휴장일) 집합을 건너뛴다(`report_metadata.py:13-70`). 설/추석 연휴에 잘못된 `prediction_for_date`가 나오던 문제가 해소됐다. (휴장일 집합은 연 단위 갱신 필요.)

### 해결됨 — P2 결과 정리(`cleanup_runs`) 보호장치

- `cleanup_runs`는 `latest_manifest.json`/`latest/manifest.json`이 가리키는 `run_id`를 보호 집합으로 두어 삭제 대상에서 제외하고(`result_cleanup.py:70-108`), `_remove`는 허용 루트 밖 경로 삭제를 거부한다. 보존 정책은 개수(`successful_run_count`)+기간(`*_days`) 기준의 `RetentionPolicy`다.

### 미해결 — P2 `pm_report`/`result_simple` 스키마 계약 문서화

- **문제**: 챗봇·PM 리포트가 의존하는 `result_simple.csv` 컬럼 계약이 코드(`build_pipeline_result_simple` → `format_result_simple`)에 흩어져 있고, 스키마 강제(pandera 등)가 없다.
- **제안**: 필수 컬럼·타입을 문서/스키마로 고정해 회귀를 테스트.

### 부분 해결 — P2 이슈 요약 LLM 호출 비용/실패 처리

- **진행**: `issue_summary.py`는 LLM 호출 실패 시 `try/except`로 규칙 기반 fallback 텍스트를 보존한다(`_llm_symbol_issue_summary`). 프롬프트 입력은 공시 10건·뉴스 15건 등으로 제한된다.
- **남은 문제**: 요약 대상 종목 수 기본 상한과 LLM 응답 캐시(뉴스 임팩트 모듈의 `FileLLMResponseCache` 재사용)는 아직 명시/연결되지 않았다.
- **제안**: 요약 대상 상위 N종목 상한과 캐시 키 정책을 문서화.
