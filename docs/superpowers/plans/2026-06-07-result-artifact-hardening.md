# Result Artifact Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모든 `result/` 개선안을 구현하여 비밀정보를 보호하고, 실행별 산출물을 검증·격리하며, 검증된 운영 결과만 단계적으로 `latest/`와 기존 호환 경로에 승격한다.

**Architecture:** 작은 정책 모듈(`secrets`, `report_metadata`, `context_policy`, `result_validity`, `run_artifacts`, `result_cleanup`)을 추가하고 기존 pipeline/chatbot이 이를 조합한다. 파이프라인은 먼저 불변 run 디렉터리에 기록하고 manifest 검증 후 운영 결과만 승격하며, 챗봇은 검증된 latest를 우선 읽고 기존 경로를 fallback으로 사용한다.

**Tech Stack:** Python 3.10+, pandas, pytest, pathlib, hashlib, json, shutil

---

## File Structure

**Create**

- `src/utils/secrets.py`: argv, 문자열, 중첩 JSON의 비밀정보 마스킹
- `src/reports/report_metadata.py`: run ID와 공통 리포트 메타데이터 생성
- `src/reports/context_policy.py`: 가격·컨텍스트 기준일 결합 정책
- `src/validation/result_validity.py`: 백테스트·calibration 유효성 판정
- `src/reports/run_artifacts.py`: run staging, manifest, latest 승격, 호환 복사
- `src/utils/result_cleanup.py`: TTL 및 허용 경로 기반 보존 정리
- `tests/test_secret_redaction.py`
- `tests/test_report_metadata.py`
- `tests/test_context_policy.py`
- `tests/test_result_validity.py`
- `tests/test_run_artifacts.py`
- `tests/test_result_cleanup.py`

**Modify**

- `src/chatbot/kakao_colab_bot.py`: runtime 경로, 마스킹, TTL, latest 우선 읽기, sample 차단
- `src/pipeline.py`: run context 사용, 날짜 정책, record type, 공통 메타데이터, 승격
- `src/reports/pm_report.py`: 메타데이터 전달과 원자 UTF-8 저장
- `src/validation/metrics.py`: calibration 표본·bin 진단
- `tests/test_kakao_colab_bot.py`
- `tests/test_pipeline_smoke.py`
- `tests/test_backtest_and_calibration.py`
- `tests/test_news_impact_context.py`
- `tests/conftest.py`
- `docs/RESULT_FILES_GUIDE.md`
- `docs/RESULT_CLEANUP.md`
- `README.md`

### Task 1: Secret Redaction and Secure Runtime Persistence

**Files:**
- Create: `src/utils/secrets.py`
- Create: `tests/test_secret_redaction.py`
- Modify: `src/chatbot/kakao_colab_bot.py:874-885,1015-1088,1631-1688`
- Modify: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Write failing redaction tests**

```python
from src.utils.secrets import redact_argv, redact_text, redact_value


def test_redact_argv_masks_flag_values_and_registered_secrets():
    argv = ["python", "x.py", "--openai-api-key", "sk-live", "--name=ok", "--token=abc"]
    assert redact_argv(argv, secret_values=["sk-live"]) == [
        "python", "x.py", "--openai-api-key", "[REDACTED]", "--name=ok", "--token=[REDACTED]"
    ]


def test_redact_value_recursively_masks_runtime_state():
    payload = {"command": ["--naver-client-secret", "secret"], "error": "failed secret"}
    redacted = redact_value(payload, secret_values=["secret"])
    assert "secret" not in str(redacted)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_secret_redaction.py -q`

Expected: FAIL because `src.utils.secrets` does not exist.

- [ ] **Step 3: Implement minimal redactor**

```python
# src/utils/secrets.py
REDACTED = "[REDACTED]"
SECRET_FLAGS = {
    "--openai-api-key", "--dart-api-key", "--naver-client-id", "--naver-client-secret"
}

def redact_argv(argv, secret_values=()):
    out, mask_next = [], False
    for raw in map(str, argv):
        lower = raw.lower()
        if mask_next:
            out.append(REDACTED)
            mask_next = False
        elif lower in SECRET_FLAGS or any(word in lower for word in ("token", "password")):
            out.append(raw)
            mask_next = "=" not in raw
            if "=" in raw:
                out[-1] = raw.split("=", 1)[0] + "=" + REDACTED
        else:
            out.append(redact_text(raw, secret_values))
    return out

def redact_text(text, secret_values=()):
    result = str(text)
    for value in sorted({str(v) for v in secret_values if str(v)}, key=len, reverse=True):
        result = result.replace(value, REDACTED)
    return result

def redact_value(value, secret_values=()):
    if isinstance(value, dict):
        return {k: redact_value(v, secret_values) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, secret_values) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v, secret_values) for v in value)
    return redact_text(value, secret_values) if isinstance(value, str) else value
```

- [ ] **Step 4: Add chatbot regression tests**

Add tests proving:

```python
def test_job_registry_redacts_secrets(tmp_path, capsys):
    # Start a job with secret-looking extra_args.
    # Assert state JSON and captured console contain no raw secret.

def test_streamed_process_output_redacts_secrets(tmp_path):
    # Feed a stdout line containing a registered key.
    # Assert written log contains [REDACTED], not the key.
```

- [ ] **Step 5: Wire redaction into chatbot**

In `KakaoColabPredictionBot`:

- Keep real `command` only in memory for `process_runner`.
- Store `redact_argv(command, self.runtime_config.secret_values())`.
- Call `_console_log(redact_text(message, secret_values))`.
- Redact each line before `_stream_process_output` writes it.
- Call `redact_value()` inside `_save_registry()` before atomic UTF-8 save.
- On completed jobs, remove `command` and retain only symbol, status, timestamps, exit code, log path, and failure summary.

- [ ] **Step 6: Verify GREEN**

Run: `pytest tests/test_secret_redaction.py tests/test_kakao_colab_bot.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/utils/secrets.py src/chatbot/kakao_colab_bot.py tests/test_secret_redaction.py tests/test_kakao_colab_bot.py
git commit -m "Protect chatbot runtime secrets"
```

### Task 2: Common Run Metadata and UTF-8 Atomic JSON

**Files:**
- Create: `src/reports/report_metadata.py`
- Create: `tests/test_report_metadata.py`
- Modify: `src/reports/pm_report.py`
- Modify: `src/pipeline.py:610-765`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing metadata tests**

```python
from src.reports.report_metadata import build_report_metadata, generate_run_id


def test_report_metadata_contains_required_identity_fields():
    run_id = generate_run_id()
    metadata = build_report_metadata(
        run_id=run_id,
        environment="smoke",
        data_mode="sample",
        input_as_of_date="2023-08-10",
        prediction_for_date="2023-08-11",
        context_as_of_date=None,
        config_payload={"x": 1},
    )
    assert metadata["schema_version"] == "1.0"
    assert metadata["run_id"] == run_id
    assert metadata["environment"] == "smoke"
    assert len(metadata["config_hash"]) == 64
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_report_metadata.py -q`

Expected: FAIL because metadata module is missing.

- [ ] **Step 3: Implement metadata builder**

Implement:

```python
SCHEMA_VERSION = "1.0"

def generate_run_id(now=None) -> str:
    current = now or datetime.now(timezone.utc)
    return f"{current.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

def detect_git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        return None

def hash_config(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def build_report_metadata(*, run_id, environment, data_mode, input_as_of_date,
                          prediction_for_date, context_as_of_date, config_payload,
                          status="pass", blocking_reasons=()) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "environment": environment,
        "data_mode": data_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_as_of_date": input_as_of_date,
        "prediction_for_date": prediction_for_date,
        "context_as_of_date": context_as_of_date,
        "git_commit": detect_git_commit(Path(__file__).resolve().parents[2]),
        "config_hash": hash_config(config_payload),
        "status": status,
        "blocking_reasons": list(blocking_reasons),
    }
```

Use timezone-aware UTC for run ID/generated time. Git lookup failure returns `None`.

- [ ] **Step 4: Write failing UTF-8 report test**

```python
def test_pipeline_report_is_utf8(tmp_path, monkeypatch):
    # Run bundled sample pipeline against redirected result dir.
    # Decode report with UTF-8 and assert Korean text round-trips.
```

- [ ] **Step 5: Wire atomic UTF-8 JSON**

- Replace pipeline direct `report_path.write_text` call with `atomic_write_text(report_path, serialized_report, encoding="utf-8")`.
- Replace `save_pm_report()` implementation with `atomic_write_text`.
- Make `build_pm_report()` copy common metadata fields from pipeline report.

- [ ] **Step 6: Verify GREEN**

Run: `pytest tests/test_report_metadata.py tests/test_pipeline_smoke.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/reports/report_metadata.py src/reports/pm_report.py src/pipeline.py tests/test_report_metadata.py tests/test_pipeline_smoke.py
git commit -m "Add common report metadata"
```

### Task 3: Run Artifacts, Manifest, Latest Promotion, and Compatibility Copies

**Files:**
- Create: `src/reports/run_artifacts.py`
- Create: `tests/test_run_artifacts.py`
- Modify: `src/reports/output.py`
- Modify: `src/pipeline.py:610-765,768-918`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing artifact lifecycle tests**

```python
def test_artifacts_share_run_id(tmp_path):
    manager = RunArtifactManager(tmp_path, metadata={
        "run_id": "run-1", "environment": "production", "data_mode": "real",
        "status": "pass", "blocking_reasons": [],
    })
    # Write required CSV/JSON files and finalize.
    manifest = manager.build_manifest()
    assert manifest["run_id"] == "run-1"
    assert all(item["sha256"] for item in manifest["artifacts"])


def test_latest_promoted_only_after_success(tmp_path):
    # Seed latest marker, fail finalization, assert marker unchanged.


def test_smoke_output_cannot_replace_production_latest(tmp_path):
    # Finalize smoke run; assert existing production latest unchanged.


def test_compatibility_copies_update_only_after_latest_promotion(tmp_path):
    # Assert top-level result_simple.csv is copied only after successful promotion.
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_run_artifacts.py -q`

Expected: FAIL because `RunArtifactManager` is missing.

- [ ] **Step 3: Implement run artifact manager**

Required API:

```python
REQUIRED_ARTIFACTS = (
    "csv/result_simple.csv",
    "csv/result_detail.csv",
    "csv/result_news.csv",
    "csv/result_disclosure.csv",
    "pm_report.json",
    "pipeline_report.json",
)

class RunArtifactManager:
    def __init__(self, result_root: Path, metadata: dict):
        self.result_root = result_root
        self.metadata = dict(metadata)
        self.run_dir = result_root / "runs" / metadata["run_id"]
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def path(self, relative_path: str) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relative_path: str, payload: dict) -> Path:
        target = self.path(relative_path)
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return target

    def write_csv(self, relative_path: str, frame: pd.DataFrame) -> Path:
        return safe_to_csv(frame, self.path(relative_path), allow_fallback=False)

    def build_manifest(self) -> dict:
        return build_manifest_payload(self.run_dir, self.metadata)

    def validate(self) -> tuple[str, list[str]]:
        return validate_manifest(self.build_manifest(), REQUIRED_ARTIFACTS)

    def finalize(self) -> dict:
        return finalize_run(self.result_root, self.run_dir, self.build_manifest())
```

`build_manifest()` records relative path, SHA-256, size, generated time, and CSV row count. `finalize()` writes manifest, rejects non-production/sample/fail runs, atomically replaces `latest/`, then copies latest core files to legacy top-level paths.

- [ ] **Step 4: Integrate pipeline staging**

Change `_write_pipeline_artifacts()` to receive a `RunArtifactManager`, write all core CSV/JSON and figure paths inside its run directory, then call `finalize()` only after every write succeeds.

Keep custom `--report-json` as an additional run-local report alias; canonical report remains `pipeline_report.json`.

- [ ] **Step 5: Add smoke integration assertions**

Extend `test_run_pipeline_generates_report_and_figures`:

- all core artifacts exist under one run
- manifest hashes match
- smoke/sample does not alter seeded production latest
- report/PM/manifest run IDs match

- [ ] **Step 6: Verify GREEN**

Run: `pytest tests/test_run_artifacts.py tests/test_pipeline_smoke.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/reports/run_artifacts.py src/reports/output.py src/pipeline.py tests/test_run_artifacts.py tests/test_pipeline_smoke.py
git commit -m "Isolate and promote result runs"
```

### Task 4: Date/Context Policy and Sample-Safe Chatbot Reads

**Files:**
- Create: `src/reports/context_policy.py`
- Create: `tests/test_context_policy.py`
- Modify: `src/pipeline.py:580-730`
- Modify: `src/chatbot/kakao_colab_bot.py:168-200,647-685,1212-1251,1848-1905`
- Modify: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Write failing date policy tests**

```python
from src.reports.context_policy import evaluate_context_policy


def test_context_date_matches_prediction_policy():
    result = evaluate_context_policy("2026-06-05", "2026-06-07", max_gap_days=3)
    assert result.allowed is True


def test_stale_context_is_excluded():
    result = evaluate_context_policy("2023-08-10", "2026-06-07", max_gap_days=3)
    assert result.allowed is False
    assert result.reason == "context_date_gap_exceeded"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_context_policy.py -q`

Expected: FAIL because policy module is missing.

- [ ] **Step 3: Implement policy**

```python
@dataclass(frozen=True)
class ContextPolicyResult:
    allowed: bool
    input_as_of_date: str | None
    context_as_of_date: str | None
    gap_days: int | None
    reason: str | None

def evaluate_context_policy(input_as_of_date, context_as_of_date, max_gap_days=3):
    input_date = pd.to_datetime(input_as_of_date, errors="coerce")
    context_date = pd.to_datetime(context_as_of_date, errors="coerce")
    if pd.isna(input_date) or pd.isna(context_date):
        return ContextPolicyResult(False, _iso(input_date), _iso(context_date), None, "missing_context_date")
    gap_days = abs((context_date.normalize() - input_date.normalize()).days)
    reason = None if gap_days <= max_gap_days else "context_date_gap_exceeded"
    return ContextPolicyResult(reason is None, _iso(input_date), _iso(context_date), gap_days, reason)
```

- [ ] **Step 4: Apply policy and metadata columns**

- Derive `input_as_of_date` from final input price date.
- Derive `prediction_for_date` with the existing next-business-day behavior used by pipeline outputs.
- Derive `context_as_of_date` from accepted context rows.
- If policy rejects context, export explicit `no_data` rows/reason but do not merge stale context into prediction display columns.
- Add `environment`, `data_mode`, and date columns to simple/detail/context CSV outputs.

- [ ] **Step 5: Add latest-first chatbot resolver and sample block tests**

Add three concrete fixtures: one valid production/latest manifest whose CSV differs from the legacy CSV, one directory containing only a legacy CSV, and one sample/latest manifest. Assert respectively that the bot returns the latest row, returns the legacy row, and refuses a production recommendation from the sample row.

Implement a resolver that reads `result/latest/manifest.json`, verifies `environment=production`, `data_mode=real`, and chooses manifest artifact paths; otherwise fallback to legacy paths. Recommendation requests must reject sample/smoke cached results with a clear refresh message.

- [ ] **Step 6: Verify GREEN**

Run: `pytest tests/test_context_policy.py tests/test_kakao_colab_bot.py tests/test_pipeline_smoke.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/reports/context_policy.py src/pipeline.py src/chatbot/kakao_colab_bot.py tests/test_context_policy.py tests/test_kakao_colab_bot.py tests/test_pipeline_smoke.py
git commit -m "Enforce result date and sample policies"
```

### Task 5: Backtest and Calibration Validity

**Files:**
- Create: `src/validation/result_validity.py`
- Create: `tests/test_result_validity.py`
- Modify: `src/validation/metrics.py`
- Modify: `src/pipeline.py:700-754`
- Modify: `tests/test_backtest_and_calibration.py`

- [ ] **Step 1: Write failing validity tests**

```python
from src.validation.result_validity import evaluate_backtest_validity


def test_invalid_backtest_reports_blocking_reason():
    validity = evaluate_backtest_validity(
        backtest={"days": 20, "halted_days": 20, "avg_selected_count": 0.0},
        tradable_prediction_count=0,
    )
    assert validity["backtest_valid"] is False
    assert "tradable_prediction_count_zero" in validity["blocking_reasons"]
    assert "all_days_halted" in validity["blocking_reasons"]
```

Add calibration tests:

```python
def test_calibration_insufficient_sample_returns_null_ece():
    result = probability_calibration_metrics([1], [0.8], min_samples=20)
    assert result["ece"] is None
    assert result["valid"] is False
    assert result["reason"] == "insufficient_samples"

def test_calibration_reports_non_empty_bins():
    result = probability_calibration_metrics([0, 1] * 20, [0.1, 0.9] * 20)
    assert result["valid"] is True
    assert result["bins"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_result_validity.py tests/test_backtest_and_calibration.py -q`

Expected: FAIL for missing validity API/new calibration fields.

- [ ] **Step 3: Implement validity and diagnostics**

`evaluate_backtest_validity()` must add reasons for:

- zero tradable predictions
- all days halted
- zero average selected count
- no evaluation days

Update `probability_calibration_metrics()` to return:

```python
{
    "valid": bool,
    "reason": str | None,
    "sample_count": int,
    "brier": float | None,
    "ece": float | None,
    "bins": [{"lower": 0.0, "upper": 0.1, "count": 4, "confidence": 0.08, "accuracy": 0.0}],
}
```

- [ ] **Step 4: Wire report status**

Merge validity blocking reasons into common report metadata. Invalid backtest produces `status="warning"` unless another hard failure exists. Preserve raw performance numbers but label them invalid.

- [ ] **Step 5: Verify GREEN**

Run: `pytest tests/test_result_validity.py tests/test_backtest_and_calibration.py tests/test_pipeline_smoke.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/validation/result_validity.py src/validation/metrics.py src/pipeline.py tests/test_result_validity.py tests/test_backtest_and_calibration.py tests/test_pipeline_smoke.py
git commit -m "Report backtest and calibration validity"
```

### Task 6: Explicit News/Disclosure Record Types Without Signal Changes

**Files:**
- Modify: `src/pipeline.py:655-716`
- Modify: `tests/test_news_impact_context.py`
- Modify: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write failing record type tests**

Create one pipeline export fixture for each case: a collected raw event, an issue-summary fallback, and rejected stale context. Assert their record types are `event`, `summary`, and `no_data`. Snapshot `predicted_return`, recommendation, and predicted-return ordering before export classification, then assert exact equality afterward.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_news_impact_context.py tests/test_pipeline_smoke.py -q`

Expected: FAIL because context exports lack `record_type` and collection fields.

- [ ] **Step 3: Implement export classification**

Ensure both news/disclosure CSVs always contain:

```text
record_type, collection_status, no_data_reason, collection_error
```

Rules:

- collected raw row: `event`, `completed`, empty reasons
- issue snapshot fallback: `summary`, `completed`, empty reasons
- no accepted data: `no_data`, `empty|excluded|failed`, explicit reason/error

Do not pass these fields into any model, policy, ranking, or backtest function.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_news_impact_context.py tests/test_display_only_feature_guard.py tests/test_pipeline_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/pipeline.py tests/test_news_impact_context.py tests/test_pipeline_smoke.py
git commit -m "Classify news and disclosure outputs"
```

### Task 7: Runtime TTL, Retention, and Safe Cleanup

**Files:**
- Create: `src/utils/result_cleanup.py`
- Create: `tests/test_result_cleanup.py`
- Modify: `src/chatbot/kakao_colab_bot.py`
- Modify: `tests/test_kakao_colab_bot.py`
- Modify: `tests/conftest.py`
- Modify: `docs/RESULT_CLEANUP.md`

- [ ] **Step 1: Write failing cleanup tests**

Create concrete filesystem fixtures containing `latest/`, an outside sentinel, expired/current runs, expired/current logs, and expired/current registry entries. Run cleanup and assert only expired allowed-root entries disappear; `latest/`, the outside sentinel, current entries, and runtime state JSON remain.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_result_cleanup.py tests/test_kakao_colab_bot.py -q`

Expected: FAIL because cleanup/TTL APIs are missing.

- [ ] **Step 3: Implement safe cleanup**

Required API:

```python
@dataclass(frozen=True)
class RetentionPolicy:
    successful_run_count: int = 10
    successful_run_days: int = 30
    failed_run_days: int = 30
    runtime_log_days: int = 14

def cleanup_result_artifacts(result_root: Path, policy: RetentionPolicy, now=None) -> dict:
    current = now or datetime.now(timezone.utc)
    removed = []
    removed.extend(cleanup_runs(result_root / "runs", policy, current))
    removed.extend(cleanup_logs(result_root / "runtime" / "logs", policy, current))
    return {"removed": removed, "removed_count": len(removed)}

def prune_registry(data: dict, *, timestamp_field: str, ttl: timedelta, now=None) -> dict:
    current = now or datetime.now(timezone.utc)
    return {
        key: value for key, value in data.items()
        if current - parse_utc(value.get(timestamp_field)) <= ttl
    }
```

Before recursive deletion, resolve every target and assert it is inside exactly one allowed root: `runs/`, `test/`, or `runtime/logs/`. Never delete `latest/` or runtime state JSON.

- [ ] **Step 4: Move runtime defaults with fallback**

Default new paths:

```text
result/runtime/chatbot_jobs.json
result/runtime/chatbot_sessions.json
result/runtime/prewarm_cache_meta.json
result/runtime/logs/
```

If a new path is absent and a legacy path exists, load legacy data, redact/prune it, then save to the new path. Keep constructor-supplied paths unchanged for tests/users.

- [ ] **Step 5: Make pytest artifacts disposable**

Update `tests/conftest.py` with a session fixture that removes the session-specific test temp root after successful completion unless `KEEP_TEST_ARTIFACTS=1`. Do not recursively remove the shared `result/.pytest_tmp` root while another test process may use it.

- [ ] **Step 6: Verify GREEN**

Run: `pytest tests/test_result_cleanup.py tests/test_kakao_colab_bot.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/utils/result_cleanup.py src/chatbot/kakao_colab_bot.py tests/test_result_cleanup.py tests/test_kakao_colab_bot.py tests/conftest.py docs/RESULT_CLEANUP.md
git commit -m "Add safe result retention cleanup"
```

### Task 8: Documentation and Compatibility Guide

**Files:**
- Modify: `docs/RESULT_FILES_GUIDE.md`
- Modify: `docs/RESULT_ANALYSIS_AND_IMPROVEMENTS.md`
- Modify: `README.md`

- [ ] **Step 1: Update result guide**

Document:

- official `latest/` and immutable `runs/<run_id>/`
- runtime/test locations
- manifest fields and validity meanings
- top-level compatibility copies and deprecation direction
- news/disclosure display-only and `record_type`
- sample/smoke isolation

- [ ] **Step 2: Mark improvement checklist status**

Add an implementation-status section to `RESULT_ANALYSIS_AND_IMPROVEMENTS.md`, mapping each P0/P1/P2 recommendation to code/tests and noting that exposed API keys still require manual revocation.

- [ ] **Step 3: Update README commands**

Explain that sample smoke runs do not promote production latest, and show where to inspect manifest:

```powershell
Get-Content -Encoding utf8 result/latest/manifest.json
```

- [ ] **Step 4: Check docs**

Run:

```powershell
git diff --check
Select-String -Path docs\RESULT_*.md,README.md -Pattern 'result/chatbot_jobs.json|result/chatbot_logs'
```

Expected: no stale path claims except explicitly labeled legacy compatibility paths.

- [ ] **Step 5: Commit**

```powershell
git add docs/RESULT_FILES_GUIDE.md docs/RESULT_ANALYSIS_AND_IMPROVEMENTS.md README.md
git commit -m "Document hardened result lifecycle"
```

### Task 9: Full Verification, Secret Audit, and PR

**Files:**
- Modify only if verification exposes defects.

- [ ] **Step 1: Run focused tests**

```powershell
pytest tests/test_secret_redaction.py tests/test_report_metadata.py tests/test_run_artifacts.py tests/test_context_policy.py tests/test_result_validity.py tests/test_result_cleanup.py -q
```

Expected: PASS.

- [ ] **Step 2: Run integration and guard tests**

```powershell
pytest tests/test_pipeline_smoke.py tests/test_kakao_colab_bot.py tests/test_backtest_and_calibration.py tests/test_news_impact_context.py tests/test_display_only_feature_guard.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run: `pytest`

Expected: PASS with no warnings indicating failed artifact promotion or leaked secrets.

- [ ] **Step 4: Run bundled smoke pipeline**

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json --figure-dir figures_smoke
```

Expected:

- smoke run generated under run/test area
- JSON decodes as UTF-8
- manifest identifies `environment=smoke`, `data_mode=sample`
- existing production `latest/` is unchanged

- [ ] **Step 5: Audit generated JSON/logs for configured secret values**

Run a Python audit that reads secret environment variables only to search for exact values, never prints them, and exits nonzero if found:

```powershell
python -c "import os,pathlib,sys; vals=[os.getenv(k,'') for k in ('OPENAI_API_KEY','DART_API_KEY','NAVER_CLIENT_ID','NAVER_CLIENT_SECRET')]; vals=[v for v in vals if v]; bad=[]; [bad.append(str(p)) for p in pathlib.Path('result').rglob('*') if p.is_file() and p.suffix.lower() in {'.json','.log'} and any(v in p.read_text(encoding='utf-8',errors='ignore') for v in vals)]; sys.exit(1 if bad else 0)"
```

Expected: exit code 0.

- [ ] **Step 6: Review diff and status**

```powershell
git diff --check
git status --short
git log --oneline --decorate -12
```

Expected: clean formatting; only intentional changes.

- [ ] **Step 7: Request code review**

Use `superpowers:requesting-code-review`; fix all actionable findings and rerun affected tests.

- [ ] **Step 8: Publish PR**

Use `github:yeet` to push the current branch and create a draft PR. PR body must include:

- security hardening summary and manual credential revocation warning
- staged result lifecycle and compatibility behavior
- date/sample/backtest/calibration policy summary
- exact focused/full/smoke test results
- artifact paths demonstrating the new structure
