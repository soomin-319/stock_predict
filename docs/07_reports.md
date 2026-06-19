# 07. 리포트 및 산출물

`src/reports/`와 `src/utils/`는 파이프라인 결과를 CSV/JSON으로 저장하고 관리한다.
뉴스·공시·뉴스 임팩트 컨텍스트는 표시/검토용이며, `predicted_return`, 순위, 추천, 신호를 바꾸지 않는다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `output.py` | CSV 출력 빌드 및 저장 |
| `result_formatter.py` | 숫자/텍스트 포맷팅 |
| `pm_report.py` | 포트폴리오 매니저 JSON 리포트 |
| `run_artifacts.py` | 실행별 아티팩트 디렉터리 관리 |
| `report_metadata.py` | 실행 메타데이터 생성, KRX 영업일/캘린더 커버리지 계산, 산출물 스키마 버전 관리 |
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
├── latest_manifest.json        # 최신 운영 실행 포인터
├── latest/                     # 최신 운영 출력의 호환 복사본
│   ├── manifest.json
│   ├── csv/
│   │   ├── result_detail.csv   # 전체 예측 상세
│   │   ├── result_simple.csv   # 사용자용 요약 (챗봇)
│   │   ├── result_news.csv     # 뉴스 이벤트/요약 스냅샷
│   │   └── result_disclosure.csv
│   ├── pm_report.json
│   └── pipeline_report.json
└── runs/
    └── <run_id>/               # 실행별 원본 보관 (동일 구조)
```

샘플 실행(`--input data/sample_ohlcv.csv`)은 `latest/`와 `latest_manifest.json`을 갱신하지 않는다.

---

## CSV 출력

### `result_detail.csv`
최신 예측 행과 피처 컨텍스트를 모두 포함. 주요 컬럼: `Symbol`, `종목명`, `Date`, `Close`,
`predicted_return`, `up_probability`, `signal_score`, `recommendation`, `confidence_score`,
`uncertainty_score`, `quantile_low/mid/high`, `history_direction_accuracy`, 포맷된 표시 컬럼, 모든 모델 피처.

### `result_simple.csv`
챗봇이 사용하는 간소화 버전(`build_pipeline_result_simple`). 핵심 컬럼만 포함.

### `result_news.csv` / `result_disclosure.csv`
| 컬럼 | 설명 |
|------|------|
| `Date`, `Symbol` | 날짜/종목 |
| `source_type` | `"news"` 또는 `"disclosure"` |
| `title`, `published_at`, `provider` | 이벤트 본문/시각/제공자 |
| `record_type` | `event` / `summary` / `no_data` |
| `collection_status` | `completed` / `failed` / `empty` |

모든 CSV는 `utf-8-sig`(BOM) 인코딩으로 저장해 Windows Excel 한글 호환성을 보장한다.

---

## 아티팩트 관리 (`run_artifacts.py`)

```python
class RunArtifactManager:
    def path(self, relative) -> Path
    def write_csv(self, relative, df) -> Path
    def write_json(self, relative, data) -> Path
    def finalize(self) -> dict   # latest/ 갱신, manifest 저장
```

각 실행마다 `result/runs/<run_id>/`를 생성하고, 운영/실데이터 실행만 `latest/`로 승격한다.
매니페스트의 CSV 항목에는 `row_count`, `columns`, `schema_kind`, `schema_version`이 포함된다.
`schema_version`은 `report_metadata.ARTIFACT_SCHEMA_VERSIONS`의 산출물별 계약 값을 사용한다.

## 실행 메타데이터 (`report_metadata.py`)

```python
def generate_run_id() -> str
def build_report_metadata(run_id, environment, data_mode, ...) -> dict
def next_krx_business_day(date_str) -> str
def evaluate_krx_calendar_coverage(reference_date) -> dict
def artifact_schema_version(schema_kind) -> str
```

- `environment`: 샘플 입력이면 `"smoke"`, 실데이터면 `"production"`.
- `prediction_for_date`: `input_as_of_date`의 다음 KRX 영업일.
- KRX 영업일 계산은 주말 + 내장 공휴일 표를 사용한다.
- 메타데이터는 `calendar_status`, `calendar_coverage_end`, `calendar_warnings`를 포함한다.
- 예측일이 내장 공휴일 표 종료일을 넘으면 `calendar_status="expired"`가 되고, 기존 상태가 `pass`이면 `warning`으로 승격한다.
- 예측일이 공휴일 표 종료 60일 이내면 `calendar_status="near_expiry"` 경고를 남긴다.

## PM 리포트 (`pm_report.py`)

```python
def build_pm_report(pred_df, report) -> dict
def validate_pm_report_schema(pm_report) -> ...   # PM_REPORT_REQUIRED_FIELDS 계약 검사
```

`summary`(buy/sell/hold 카운트), `top_picks`, `backtest_summary` 등을 담은 PM용 요약 JSON.

## 이슈 요약 (`issue_summary.py`)

```python
def append_issue_summary_columns(pred_df, context_raw_df, openai_api_key=None, openai_model=None,
                                 summarize_symbols=None, max_llm_symbols=20, llm_cache_dir=None, ...) -> pd.DataFrame
```

OpenAI로 뉴스/공시를 한국어로 요약해 `뉴스 요약`/`공시 요약` 컬럼(표시 전용)을 추가한다.

- 기본 LLM 호출 예산은 20종목. 예산 밖 행은 규칙 기반 요약으로 폴백한다.
- `llm_cache_dir` 설정 시 동일 모델/이벤트 입력에 대해 파일 캐시를 재사용한다.
- LLM 실패나 예산 초과는 **예측값을 절대 바꾸지 않는다**(표시용 요약만 폴백).

## 뉴스 임팩트 컨텍스트 병합 (`news_impact_context.py`)

```python
def append_news_impact_context(pred_df, news_impact_report) -> pd.DataFrame
def append_generated_news_impact_context(pred_df, context_raw_df) -> pd.DataFrame
```

외부 `stock-news-impact` JSON 리포트를 `news_impact_*` 컬럼으로 병합. 표시 전용이며 예측 결과에 영향 없음([08](08_news_impact.md)).

## 컨텍스트 날짜 정책 (`context_policy.py`)

```python
def evaluate_context_policy(input_as_of_date, context_date) -> PolicyResult
```

컨텍스트(뉴스/공시) 날짜가 `input_as_of_date`에서 너무 벗어나면 필터링해, 지나치게 오래된 뉴스가 결과에 섞이지 않게 한다.

## 파일 안전 처리 (`utils/`)

- `atomic_files.atomic_write_text`: 임시 파일 → 이름 변경 방식으로 부분 쓰기 방지.
- `secrets.redact_text` / `redact_argv`: 로그/에러에서 API 키 자동 마스킹.
- `result_cleanup.cleanup_runs`: 오래된 실행 정리. `latest_manifest.json` 또는 `latest/manifest.json`이
  가리키는 실행은 보존한다.

---

## 반영된 운영 보강

> 우선순위: **P2(운영/유지보수)**. 기존 개선 제안은 코드와 테스트에 반영됨.

### P2 — KRX 영업일/공휴일 표 커버리지 경고

- **반영**: `report_metadata.py`가 내장 KRX 공휴일 표 종료일(`calendar_coverage_end`)을 메타데이터에 기록한다.
- **반영**: 예측일이 종료일을 넘으면 `calendar_status="expired"`, 종료 60일 이내면 `calendar_status="near_expiry"` 경고를 남긴다.
- **운영 규칙**: `calendar_warnings`가 있으면 공휴일 표를 갱신하거나 관리형 거래일 테이블/외부 KRX 캘린더로 이전한다.

### P2 — 스키마 버전 변경 시 소비자 동기화

- **반영**: CSV 매니페스트는 `schema_kind`별 `schema_version`을 `ARTIFACT_SCHEMA_VERSIONS`에서 기록한다.
- **반영**: 테스트가 `result_simple.csv`, `result_detail.csv`, `result_news.csv`, `result_disclosure.csv`의 스키마 계약 기록을 검증한다.
- **운영 규칙**: `result_simple.csv`/`pm_report.json` 계약을 바꿀 때 관련 `schema_version`을 올리고 챗봇·BI 소비자 테스트를 함께 갱신한다. 신규 소비자는 `latest/`(호환 복사본) 대신 `latest_manifest.json → runs/<run_id>/...`를 우선 사용한다.
