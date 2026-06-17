# Daily Publish → Colab Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 로컬에서 수동 1회 실행하는 `stock-predict-publish` 명령으로 기본 200종목 예측을 GitHub `published/`에 커밋하고, Colab 봇은 그 기준데이터를 기본 서빙하며 사용자가 요청한 종목만 세션에서 예측해 오버레이한다.

**Architecture:** 새 공유 모듈 `src/ops/published_store.py`가 게시 세트 복사·메타·인덱스 I/O를 담당한다. `src/ops/publish_predictions.py`는 파이프라인 실행 → `published_store`로 게시 → git push를 오케스트레이션한다. 봇(`kakao_colab_bot.py`)은 `published/latest/`를 베이스라인으로 읽고 세션 `result/` 행을 종목코드 기준 오버레이하며, 자동 부트스트랩을 기본 OFF로 전환한다. Colab 러너는 파이프라인을 돌리지 않는 `load_published_predictions()`를 추가한다.

**Tech Stack:** Python 3.12, pandas, pytest, 기존 `src/reports/run_artifacts.py`(run dir/manifest), `src/pipeline.py:run_pipeline`, `src/data/fetch_real_data.py`.

**참고 spec:** `docs/superpowers/specs/2026-06-17-daily-publish-colab-baseline-design.md`

---

## File Structure

- Create: `src/ops/__init__.py` — 빈 패키지 마커.
- Create: `src/ops/published_store.py` — 게시 세트 상수, `PublishMeta`, copy/meta/index I/O, published 읽기 헬퍼. (순수 파일 I/O, 네트워크/파이프라인 의존 없음)
- Create: `src/ops/publish_predictions.py` — `run_publish()` 코어 + `main()` CLI. 파이프라인 실행·git 호출은 주입 가능.
- Create: `tests/test_published_store.py`
- Create: `tests/test_publish_predictions.py`
- Modify: `src/chatbot/kakao_colab_bot.py` — `PipelineRuntimeConfig`(published_dir, 세션 입력, 부트스트랩 기본 OFF), 베이스라인 로더 + 오버레이, `_is_bootstrap_required` False.
- Modify: `tests/test_kakao_colab_bot.py` — 오버레이/부트스트랩 OFF 테스트 추가.
- Modify: `colab/stock_predict_colab.py` — `load_published_predictions()`.
- Modify: `tests/test_colab_runner.py` — `load_published_predictions` 테스트 추가.
- Modify: `pyproject.toml` — 콘솔 스크립트 등록.
- Modify: `README.md`, `docs/OPERATIONS.md` — Daily Publish 절차.

게시 세트(각 published 폴더에 동일): `csv/result_simple.csv`, `csv/result_detail.csv`, `csv/result_news.csv`, `csv/result_disclosure.csv`, `manifest.json`, `pipeline_report.json`, `publish_meta.json`.

---

## Task 1: `src/ops` 패키지 + published_store 상수·경로 헬퍼

**Files:**
- Create: `src/ops/__init__.py`
- Create: `src/ops/published_store.py`
- Test: `tests/test_published_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_published_store.py
from __future__ import annotations

from pathlib import Path

from src.ops.published_store import (
    PUBLISHED_ARTIFACTS,
    resolve_published_dir,
)


def test_published_artifacts_set():
    assert PUBLISHED_ARTIFACTS == (
        "csv/result_simple.csv",
        "csv/result_detail.csv",
        "csv/result_news.csv",
        "csv/result_disclosure.csv",
        "manifest.json",
        "pipeline_report.json",
    )


def test_resolve_published_dir_latest_and_history(tmp_path: Path):
    root = tmp_path / "published"
    assert resolve_published_dir(root, None) == root / "latest"
    assert resolve_published_dir(root, "2026-06-17") == root / "history" / "2026-06-17"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_published_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ops'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ops/__init__.py
```

(빈 파일)

```python
# src/ops/published_store.py
from __future__ import annotations

from pathlib import Path

PUBLISHED_ARTIFACTS: tuple[str, ...] = (
    "csv/result_simple.csv",
    "csv/result_detail.csv",
    "csv/result_news.csv",
    "csv/result_disclosure.csv",
    "manifest.json",
    "pipeline_report.json",
)

PUBLISH_META_NAME = "publish_meta.json"
INDEX_NAME = "index.json"


def resolve_published_dir(published_root: str | Path, date: str | None = None) -> Path:
    root = Path(published_root)
    if date is None:
        return root / "latest"
    return root / "history" / str(date)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_published_store.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ops/__init__.py src/ops/published_store.py tests/test_published_store.py
git commit -m "feat(publish): add src.ops package and published_store path helpers"
```

---

## Task 2: PublishMeta + 게시 세트 복사

**Files:**
- Modify: `src/ops/published_store.py`
- Test: `tests/test_published_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_published_store.py  (append)
import json

from src.ops.published_store import PublishMeta, copy_published_set


def _make_run_dir(run_dir: Path) -> None:
    (run_dir / "csv").mkdir(parents=True)
    (run_dir / "csv" / "result_simple.csv").write_text("종목코드\n005930\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_detail.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_news.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_disclosure.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "manifest.json").write_text('{"run_id": "rid-1", "promoted": true}', encoding="utf-8")
    (run_dir / "pipeline_report.json").write_text('{"ok": true}', encoding="utf-8")


def test_copy_published_set_copies_all_artifacts(tmp_path: Path):
    run_dir = tmp_path / "runs" / "rid-1"
    _make_run_dir(run_dir)
    dest = tmp_path / "published" / "latest"

    copy_published_set(run_dir, dest)

    assert (dest / "csv" / "result_simple.csv").read_text(encoding="utf-8-sig") == "종목코드\n005930\n"
    assert json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["run_id"] == "rid-1"
    for rel in (
        "csv/result_detail.csv",
        "csv/result_news.csv",
        "csv/result_disclosure.csv",
        "pipeline_report.json",
    ):
        assert (dest / rel).exists()


def test_copy_published_set_overwrites_existing(tmp_path: Path):
    run_dir = tmp_path / "runs" / "rid-1"
    _make_run_dir(run_dir)
    dest = tmp_path / "published" / "history" / "2026-06-17"
    dest.mkdir(parents=True)
    (dest / "stale.txt").write_text("old", encoding="utf-8")
    (dest / "csv").mkdir()
    (dest / "csv" / "result_simple.csv").write_text("종목코드\n000660\n", encoding="utf-8-sig")

    copy_published_set(run_dir, dest)

    assert (dest / "csv" / "result_simple.csv").read_text(encoding="utf-8-sig") == "종목코드\n005930\n"
    # 게시 세트는 항상 전량 교체되지만, 게시 세트에 없는 부수 파일은 건드리지 않는다.
    assert (dest / "stale.txt").read_text(encoding="utf-8") == "old"


def test_publish_meta_to_dict_shape():
    meta = PublishMeta(
        generated_at_kst="2026-06-17T18:05:00+09:00",
        trading_date="2026-06-17",
        news_mode="gemma",
        source_run_id="rid-1",
        symbol_count=200,
        git_commit="abc123",
        git_branch="main",
    )
    assert meta.to_dict() == {
        "generated_at_kst": "2026-06-17T18:05:00+09:00",
        "trading_date": "2026-06-17",
        "news_mode": "gemma",
        "source_run_id": "rid-1",
        "symbol_count": 200,
        "git": {"commit": "abc123", "branch": "main"},
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_published_store.py -v`
Expected: FAIL — `ImportError: cannot import name 'PublishMeta'`.

- [ ] **Step 3: Write minimal implementation**

`src/ops/published_store.py`에 추가(상단 import에 `shutil`, `dataclasses`, `typing` 보강):

```python
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PublishMeta:
    generated_at_kst: str
    trading_date: str
    news_mode: str
    source_run_id: str
    symbol_count: int
    git_commit: str | None = None
    git_branch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_kst": self.generated_at_kst,
            "trading_date": self.trading_date,
            "news_mode": self.news_mode,
            "source_run_id": self.source_run_id,
            "symbol_count": self.symbol_count,
            "git": {"commit": self.git_commit, "branch": self.git_branch},
        }


def copy_published_set(source_dir: str | Path, dest_dir: str | Path) -> None:
    source = Path(source_dir)
    dest = Path(dest_dir)
    missing = [rel for rel in PUBLISHED_ARTIFACTS if not (source / rel).exists()]
    if missing:
        raise FileNotFoundError(f"게시 세트 누락 아티팩트: {missing} (source={source})")
    for rel in PUBLISHED_ARTIFACTS:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        shutil.copy2(source / rel, tmp)
        tmp.replace(target)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_published_store.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ops/published_store.py tests/test_published_store.py
git commit -m "feat(publish): add PublishMeta and copy_published_set"
```

---

## Task 3: publish_meta + index.json 기록

**Files:**
- Modify: `src/ops/published_store.py`
- Test: `tests/test_published_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_published_store.py  (append)
from src.ops.published_store import (
    read_index,
    write_publish_meta,
    update_index,
)


def _meta(date: str, mode: str = "gemma") -> PublishMeta:
    return PublishMeta(
        generated_at_kst=f"{date}T18:05:00+09:00",
        trading_date=date,
        news_mode=mode,
        source_run_id=f"rid-{date}",
        symbol_count=200,
    )


def test_write_publish_meta(tmp_path: Path):
    dest = tmp_path / "published" / "latest"
    dest.mkdir(parents=True)
    write_publish_meta(dest, _meta("2026-06-17"))
    payload = json.loads((dest / "publish_meta.json").read_text(encoding="utf-8"))
    assert payload["trading_date"] == "2026-06-17"
    assert payload["git"] == {"commit": None, "branch": None}


def test_update_index_dedups_and_sorts(tmp_path: Path):
    root = tmp_path / "published"
    root.mkdir()
    update_index(root, _meta("2026-06-16"))
    update_index(root, _meta("2026-06-17"))
    update_index(root, _meta("2026-06-17", mode="rule_based"))  # 같은 날짜 재publish

    index = read_index(root)
    assert index["latest"] == "2026-06-17"
    dates = [e["trading_date"] for e in index["entries"]]
    assert dates == ["2026-06-17", "2026-06-16"]  # 최신 우선 정렬
    latest_entry = index["entries"][0]
    assert latest_entry["news_mode"] == "rule_based"  # 재publish가 덮어씀
    assert latest_entry["symbol_count"] == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_published_store.py -k "publish_meta or index" -v`
Expected: FAIL — `ImportError: cannot import name 'read_index'`.

- [ ] **Step 3: Write minimal implementation**

`src/ops/published_store.py`에 추가(`import json` 보강):

```python
import json


def write_publish_meta(dest_dir: str | Path, meta: PublishMeta) -> Path:
    target = Path(dest_dir) / PUBLISH_META_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def read_index(published_root: str | Path) -> dict[str, Any]:
    path = Path(published_root) / INDEX_NAME
    if not path.exists():
        return {"latest": None, "entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"latest": None, "entries": []}
    if not isinstance(payload, dict):
        return {"latest": None, "entries": []}
    payload.setdefault("latest", None)
    payload.setdefault("entries", [])
    return payload


def update_index(published_root: str | Path, meta: PublishMeta) -> dict[str, Any]:
    root = Path(published_root)
    index = read_index(root)
    entry = {
        "trading_date": meta.trading_date,
        "generated_at_kst": meta.generated_at_kst,
        "news_mode": meta.news_mode,
        "symbol_count": meta.symbol_count,
        "source_run_id": meta.source_run_id,
    }
    entries = [e for e in index["entries"] if e.get("trading_date") != meta.trading_date]
    entries.append(entry)
    entries.sort(key=lambda e: str(e.get("trading_date", "")), reverse=True)
    payload = {"latest": entries[0]["trading_date"] if entries else None, "entries": entries}
    (root / INDEX_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_published_store.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ops/published_store.py tests/test_published_store.py
git commit -m "feat(publish): add publish_meta writer and history index"
```

---

## Task 4: published 읽기 헬퍼 (봇/Colab 공용)

**Files:**
- Modify: `src/ops/published_store.py`
- Test: `tests/test_published_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_published_store.py  (append)
import pandas as pd

from src.ops.published_store import load_published_simple, read_publish_meta


def test_load_published_simple_reads_codes_as_str(tmp_path: Path):
    dest = tmp_path / "published" / "latest"
    (dest / "csv").mkdir(parents=True)
    (dest / "csv" / "result_simple.csv").write_text(
        "종목코드,종목명\n005930,삼성전자\n", encoding="utf-8-sig"
    )
    df = load_published_simple(dest)
    assert list(df["종목코드"]) == ["005930"]
    assert df["종목코드"].dtype == object


def test_load_published_simple_missing_returns_empty(tmp_path: Path):
    df = load_published_simple(tmp_path / "nope")
    assert df.empty


def test_read_publish_meta_missing_returns_empty(tmp_path: Path):
    assert read_publish_meta(tmp_path / "nope") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_published_store.py -k "load_published or publish_meta_missing" -v`
Expected: FAIL — `ImportError: cannot import name 'load_published_simple'`.

- [ ] **Step 3: Write minimal implementation**

`src/ops/published_store.py`에 추가(`import pandas as pd` 보강):

```python
import pandas as pd


def load_published_simple(published_dir: str | Path) -> pd.DataFrame:
    path = Path(published_dir) / "csv" / "result_simple.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"종목코드": str}, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def read_publish_meta(published_dir: str | Path) -> dict[str, Any]:
    path = Path(published_dir) / PUBLISH_META_NAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_published_store.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ops/published_store.py tests/test_published_store.py
git commit -m "feat(publish): add published read helpers for serving"
```

---

## Task 5: publish 아티팩트 게시 오케스트레이션 (`publish_artifacts`)

파이프라인 실행과 분리해, "run dir → published/latest + history + index" 부분만 먼저 테스트한다.

**Files:**
- Create: `src/ops/publish_predictions.py`
- Test: `tests/test_publish_predictions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_publish_predictions.py
from __future__ import annotations

import json
from pathlib import Path

from src.ops.publish_predictions import publish_artifacts
from src.ops.published_store import read_index


def _make_run_dir(run_dir: Path) -> None:
    (run_dir / "csv").mkdir(parents=True)
    detail = "Symbol,Date\n005930.KS,2026-06-17\n000660.KS,2026-06-17\n"
    (run_dir / "csv" / "result_simple.csv").write_text(
        "종목코드,종목명\n005930,삼성전자\n000660,SK하이닉스\n", encoding="utf-8-sig"
    )
    (run_dir / "csv" / "result_detail.csv").write_text(detail, encoding="utf-8-sig")
    (run_dir / "csv" / "result_news.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "csv" / "result_disclosure.csv").write_text("Symbol\n005930.KS\n", encoding="utf-8-sig")
    (run_dir / "manifest.json").write_text('{"run_id": "rid-1", "promoted": true}', encoding="utf-8")
    (run_dir / "pipeline_report.json").write_text('{"ok": true}', encoding="utf-8")


def test_publish_artifacts_writes_latest_history_index(tmp_path: Path):
    run_dir = tmp_path / "result" / "runs" / "rid-1"
    _make_run_dir(run_dir)
    published_root = tmp_path / "published"

    meta = publish_artifacts(
        run_dir=run_dir,
        published_root=published_root,
        trading_date="2026-06-17",
        news_mode="gemma",
        source_run_id="rid-1",
        symbol_count=2,
    )

    assert meta.trading_date == "2026-06-17"
    latest_simple = published_root / "latest" / "csv" / "result_simple.csv"
    hist_simple = published_root / "history" / "2026-06-17" / "csv" / "result_simple.csv"
    assert latest_simple.exists() and hist_simple.exists()
    assert json.loads((published_root / "latest" / "publish_meta.json").read_text(encoding="utf-8"))["news_mode"] == "gemma"
    assert read_index(published_root)["latest"] == "2026-06-17"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_publish_predictions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ops.publish_predictions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ops/publish_predictions.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.ops.published_store import (
    PublishMeta,
    copy_published_set,
    resolve_published_dir,
    update_index,
    write_publish_meta,
)


def publish_artifacts(
    *,
    run_dir: str | Path,
    published_root: str | Path,
    trading_date: str,
    news_mode: str,
    source_run_id: str,
    symbol_count: int,
    git_commit: str | None = None,
    git_branch: str | None = None,
    generated_at_kst: str | None = None,
) -> PublishMeta:
    run_dir = Path(run_dir)
    published_root = Path(published_root)
    meta = PublishMeta(
        generated_at_kst=generated_at_kst
        or datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        trading_date=str(trading_date),
        news_mode=str(news_mode),
        source_run_id=str(source_run_id),
        symbol_count=int(symbol_count),
        git_commit=git_commit,
        git_branch=git_branch,
    )
    for dest in (
        resolve_published_dir(published_root, None),
        resolve_published_dir(published_root, meta.trading_date),
    ):
        copy_published_set(run_dir, dest)
        write_publish_meta(dest, meta)
    update_index(published_root, meta)
    return meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_publish_predictions.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ops/publish_predictions.py tests/test_publish_predictions.py
git commit -m "feat(publish): add publish_artifacts orchestration (run dir -> published)"
```

---

## Task 6: trading_date 추론 + manifest 운영 검증

**Files:**
- Modify: `src/ops/publish_predictions.py`
- Test: `tests/test_publish_predictions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_publish_predictions.py  (append)
import pytest

from src.ops.publish_predictions import infer_trading_date, ensure_operational_manifest


def test_infer_trading_date_from_detail(tmp_path: Path):
    run_dir = tmp_path / "runs" / "rid-1"
    (run_dir / "csv").mkdir(parents=True)
    (run_dir / "csv" / "result_detail.csv").write_text(
        "Symbol,Date\n005930.KS,2026-06-16\n005930.KS,2026-06-17\n", encoding="utf-8-sig"
    )
    assert infer_trading_date(run_dir) == "2026-06-17"


def test_ensure_operational_manifest_accepts_promoted_pass():
    ensure_operational_manifest(
        {"promoted": True, "status": "pass", "environment": "production", "data_mode": "real"}
    )  # 예외 없음


def test_ensure_operational_manifest_rejects_non_operational():
    with pytest.raises(ValueError):
        ensure_operational_manifest({"promoted": False, "status": "fail"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_publish_predictions.py -k "trading_date or operational" -v`
Expected: FAIL — `ImportError: cannot import name 'infer_trading_date'`.

- [ ] **Step 3: Write minimal implementation**

`src/ops/publish_predictions.py`에 추가(`import pandas as pd` 보강):

```python
import pandas as pd


def infer_trading_date(run_dir: str | Path) -> str:
    detail_path = Path(run_dir) / "csv" / "result_detail.csv"
    if not detail_path.exists():
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    df = pd.read_csv(detail_path, encoding="utf-8-sig")
    if "Date" not in df.columns or df.empty:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if dates.empty:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    return dates.max().strftime("%Y-%m-%d")


def ensure_operational_manifest(manifest: dict | None) -> None:
    ok = bool(
        manifest
        and manifest.get("promoted") is True
        and manifest.get("status") in {"pass", "warning"}
    )
    if not ok:
        raise ValueError(
            f"운영 산출물이 아니어서 publish를 중단합니다 (manifest={manifest}). "
            "파이프라인 상태를 확인하세요."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_publish_predictions.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ops/publish_predictions.py tests/test_publish_predictions.py
git commit -m "feat(publish): add trading-date inference and operational manifest guard"
```

---

## Task 7: publish 코어 `run_publish` (파이프라인·git 주입) + CLI

**Files:**
- Modify: `src/ops/publish_predictions.py`
- Test: `tests/test_publish_predictions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_publish_predictions.py  (append)
from dataclasses import dataclass


@dataclass
class _Args:
    news_mode: str = "gemma"
    full_refresh: bool = False
    no_push: bool = True
    dry_run: bool = False
    config_json: str | None = None


def test_run_publish_invokes_pipeline_and_publishes(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    result_root = project_root / "result"
    run_dir = result_root / "runs" / "rid-9"
    _make_run_dir(run_dir)
    (result_root / "latest_manifest.json").write_text(
        '{"run_id": "rid-9"}', encoding="utf-8"
    )

    pipeline_calls = []

    def fake_pipeline(news_impact_llm_config, full_refresh):
        pipeline_calls.append({"cfg": news_impact_llm_config, "full": full_refresh})
        return {"manifest": {"promoted": True, "status": "pass", "run_id": "rid-9"}}

    git_calls = []

    result = run_publish(
        _Args(),
        project_root=project_root,
        pipeline_fn=fake_pipeline,
        git_fn=lambda *a, **k: git_calls.append((a, k)),
    )

    assert pipeline_calls[0]["cfg"].endswith("news_impact.gemma.example.json")
    assert (project_root / "published" / "latest" / "csv" / "result_simple.csv").exists()
    assert result["trading_date"] == "2026-06-17"
    assert git_calls == []  # no_push=True


def test_run_publish_rule_mode_uses_no_llm_config(tmp_path: Path):
    project_root = tmp_path
    run_dir = project_root / "result" / "runs" / "rid-r"
    _make_run_dir(run_dir)
    (project_root / "result" / "latest_manifest.json").write_text('{"run_id": "rid-r"}', encoding="utf-8")

    captured = {}

    def fake_pipeline(news_impact_llm_config, full_refresh):
        captured["cfg"] = news_impact_llm_config
        return {"manifest": {"promoted": True, "status": "pass", "run_id": "rid-r"}}

    run_publish(
        _Args(news_mode="rule"),
        project_root=project_root,
        pipeline_fn=fake_pipeline,
        git_fn=lambda *a, **k: None,
    )
    assert captured["cfg"] is None
```

(상단 import에 `from src.ops.publish_predictions import run_publish` 추가)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_publish_predictions.py -k "run_publish" -v`
Expected: FAIL — `ImportError: cannot import name 'run_publish'`.

- [ ] **Step 3: Write minimal implementation**

`src/ops/publish_predictions.py`에 추가. `resolve_latest_run_dir`로 run dir를 찾고, 주입된 `pipeline_fn`/`git_fn`을 사용한다.

```python
import argparse
import subprocess
import sys
from typing import Any, Callable

from src.reports.run_artifacts import resolve_latest_run_dir

GEMMA_CONFIG = "configs/news_impact.gemma.example.json"


def _news_config_for_mode(news_mode: str) -> str | None:
    return GEMMA_CONFIG if news_mode == "gemma" else None


def _default_pipeline_fn(project_root: Path) -> Callable[..., dict[str, Any]]:
    from src.data.fetch_real_data import append_real_ohlcv_csv, save_real_ohlcv_csv
    from src.pipeline import (
        _fallback_symbols_from_input_or_default,
        _resolve_incremental_fetch_start,
        run_pipeline,
    )

    input_csv = str(project_root / "data" / "real_ohlcv.csv")

    def _run(news_impact_llm_config: str | None, full_refresh: bool) -> dict[str, Any]:
        symbols = _fallback_symbols_from_input_or_default(input_csv)
        if full_refresh:
            save_real_ohlcv_csv(input_csv, symbols=symbols, start="2020-01-01")
        else:
            start = _resolve_incremental_fetch_start(input_csv, "2020-01-01")
            append_real_ohlcv_csv(input_csv, symbols=symbols, start=start)
        return run_pipeline(
            input_csv=input_csv,
            output_csv="result_detail.csv",
            report_json="pipeline_report.json",
            use_external=False,
            use_investor_context=True,
            news_impact_llm_config=news_impact_llm_config,
        )

    return _run


def _default_git_fn(project_root: Path) -> Callable[[list[str]], None]:
    def _git(args: list[str]) -> None:
        subprocess.run(["git", *args], cwd=project_root, check=True)

    return _git


def run_publish(
    args: argparse.Namespace,
    *,
    project_root: str | Path,
    pipeline_fn: Callable[..., dict[str, Any]] | None = None,
    git_fn: Callable[[list[str]], None] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    result_root = project_root / "result"
    published_root = project_root / "published"
    pipeline_fn = pipeline_fn or _default_pipeline_fn(project_root)
    git_fn = git_fn or _default_git_fn(project_root)

    news_config = _news_config_for_mode(args.news_mode)
    report = pipeline_fn(news_impact_llm_config=news_config, full_refresh=bool(args.full_refresh))
    manifest = report.get("manifest") if isinstance(report, dict) else None
    ensure_operational_manifest(manifest)

    run_dir = resolve_latest_run_dir(result_root)
    if run_dir is None:
        raise FileNotFoundError(f"run dir를 찾을 수 없습니다: {result_root}")
    trading_date = infer_trading_date(run_dir)
    symbol_count = _symbol_count(run_dir)
    news_mode = "rule_based" if news_config is None else _effective_news_mode(report)

    meta = publish_artifacts(
        run_dir=run_dir,
        published_root=published_root,
        trading_date=trading_date,
        news_mode=news_mode,
        source_run_id=str(manifest.get("run_id", "")),
        symbol_count=symbol_count,
        git_branch="main",
    )

    if not args.no_push and not args.dry_run:
        git_fn(["add", "published"])
        git_fn(["commit", "-m", f"chore(publish): {meta.trading_date} predictions ({meta.news_mode})"])
        git_fn(["push"])

    return meta.to_dict()


def _symbol_count(run_dir: Path) -> int:
    path = Path(run_dir) / "csv" / "result_simple.csv"
    try:
        return int(len(pd.read_csv(path, encoding="utf-8-sig")))
    except Exception:
        return 0


def _effective_news_mode(report: dict[str, Any]) -> str:
    # gemma 설정으로 호출했어도 서버 폴백 시 규칙기반이 쓰일 수 있다.
    # 파이프라인 리포트에 news-impact 모드 신호가 있으면 반영, 없으면 gemma로 표기.
    flags = report.get("news_impact") if isinstance(report.get("news_impact"), dict) else {}
    if str(flags.get("mode", "")).lower() in {"rule", "rule_based", "heuristic"}:
        return "rule_based"
    return "gemma"


def main() -> None:
    parser = argparse.ArgumentParser(description="기본 200종목 예측을 published/에 게시")
    parser.add_argument("--news-mode", choices=["gemma", "rule"], default="gemma")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-json", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    meta = run_publish(args, project_root=project_root)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

> 주의: `_effective_news_mode`는 파이프라인 리포트의 news-impact 신호를 best-effort로 읽는다. 리포트 구조가 다르면 `gemma`로 표기되며, 이는 표시용 메타이므로 예측에 영향 없음. 실제 폴백 신호 키가 확인되면 그 키로 맞춘다(구현 시 `pipeline_report.json` 확인).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_publish_predictions.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ops/publish_predictions.py tests/test_publish_predictions.py
git commit -m "feat(publish): add run_publish core with injectable pipeline/git and CLI"
```

---

## Task 8: 콘솔 스크립트 등록

**Files:**
- Modify: `pyproject.toml:25-28`

- [ ] **Step 1: 변경 적용**

`[project.scripts]` 블록에 한 줄 추가:

```toml
[project.scripts]
stock-predict = "src.pipeline:main"
stock-predict-kakao = "src.chatbot.kakao_colab_bot:main"
stock-predict-publish = "src.ops.publish_predictions:main"
stock-news-impact = "src.news_impact.run:main"
```

- [ ] **Step 2: 재설치 및 확인**

Run: `python -m pip install -e . && stock-predict-publish --help`
Expected: argparse 도움말 출력(에러 없음). `--news-mode`, `--full-refresh`, `--no-push`, `--dry-run` 노출.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(publish): register stock-predict-publish console script"
```

---

## Task 9: 봇 — `published_dir` 설정 + 세션 입력 분리 + 부트스트랩 기본 OFF

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py:72-93` (`PipelineRuntimeConfig`), `:1100-1109` (`_is_bootstrap_required`), `:180-232` (`__init__`)
- Test: `tests/test_kakao_colab_bot.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kakao_colab_bot.py  (append)
def test_runtime_config_defaults_disable_bootstrap():
    cfg = PipelineRuntimeConfig()
    assert cfg.bootstrap_on_launch is False
    assert cfg.prewarm_default_predictions is False
    assert cfg.bootstrap_default_symbols is False
    assert cfg.published_dir == "published/latest"
    assert cfg.input_csv == "result/session/session_ohlcv.csv"


def test_is_bootstrap_required_always_false(tmp_path):
    cfg = PipelineRuntimeConfig(project_root=tmp_path)
    bot = KakaoColabPredictionBot(
        runtime_config=cfg,
        result_simple_path="result/result_simple.csv",
        state_path="result/runtime/jobs.json",
        session_path="result/runtime/sessions.json",
    )
    assert bot._is_bootstrap_required() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k "bootstrap_required or defaults_disable" -v`
Expected: FAIL — `AssertionError` (현재 기본값은 True) / `AttributeError: published_dir`.

- [ ] **Step 3: Write minimal implementation**

`PipelineRuntimeConfig`(`src/chatbot/kakao_colab_bot.py:72-93`) 기본값 변경:

```python
    input_csv: str = "result/session/session_ohlcv.csv"
    ...
    use_external: bool = False
    bootstrap_default_symbols: bool = False
    bootstrap_on_launch: bool = False
    async_issue_summary_on_demand: bool = True
    real_start: str = "2018-01-01"
    prewarm_default_predictions: bool = False
    news_impact_llm_config: str | None = None
    published_dir: str = "published/latest"
    extra_args: tuple[str, ...] = ()
```

`__init__`(`:180-232`)에 published 경로 보관 + 세션 입력 디렉터리 생성 추가(예: `self.result_root` 인근):

```python
        self.published_dir = self.project_root / self.runtime_config.published_dir
        # 온디맨드 세션 입력 CSV의 부모 디렉터리를 보장(--add-symbols append 대상)
        (self.project_root / self.runtime_config.input_csv).parent.mkdir(parents=True, exist_ok=True)
```

`main()`의 `--input` argparse 기본값(`:2127`)도 세션 경로로 동기화한다(CLI 실행 시에도 publish 입력과 분리):

```python
    parser.add_argument("--input", default="result/session/session_ohlcv.csv")
```

`_is_bootstrap_required`(`:1100-1109`) 본문을 단순화:

```python
    def _is_bootstrap_required(self) -> bool:
        # published 베이스라인 서빙 모델에서는 첫 요청 시 전 종목 부트스트랩을 하지 않는다.
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k "bootstrap_required or defaults_disable" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: 회귀 확인 + Commit**

Run: `python -m pytest tests/test_kakao_colab_bot.py -v`
Expected: 기존 테스트 중 prewarm/bootstrap 기본값에 의존하던 케이스가 있으면 수정 필요. 실패 시 해당 테스트가 명시적으로 `bootstrap_on_launch=True` 등을 주도록 갱신한 뒤 통과 확인.

```bash
git add src/chatbot/kakao_colab_bot.py tests/test_kakao_colab_bot.py
git commit -m "feat(chatbot): default to published baseline, disable auto-bootstrap, separate session input"
```

---

## Task 10: 봇 — published 베이스라인 + 세션 오버레이

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py:747-793` (`_load_cached_result_simple`)
- Test: `tests/test_kakao_colab_bot.py`

핵심: `_load_cached_result_simple`이 published 베이스라인을 읽고, 기존 세션 결과(`_resolve_result_path`)를 종목코드 기준으로 위에 덮는다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kakao_colab_bot.py  (append)
def _write_simple_csv(path: Path, rows: list[tuple[str, str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["종목코드,종목명,권고"]
    lines += [f"{code},{name},{rec}" for code, name, rec in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _make_overlay_bot(tmp_path: Path) -> KakaoColabPredictionBot:
    cfg = PipelineRuntimeConfig(project_root=tmp_path, published_dir="published/latest")
    # published 베이스라인: 2종목
    _write_simple_csv(
        tmp_path / "published" / "latest" / "csv" / "result_simple.csv",
        [("005930", "삼성전자", "관망"), ("000660", "SK하이닉스", "매수")],
    )
    bot = KakaoColabPredictionBot(
        runtime_config=cfg,
        result_simple_path="result/result_simple.csv",
        state_path="result/runtime/jobs.json",
        session_path="result/runtime/sessions.json",
    )
    return bot


def test_baseline_served_without_session(tmp_path):
    bot = _make_overlay_bot(tmp_path)
    df = bot._load_cached_result_simple()
    assert set(df["종목코드"]) == {"005930", "000660"}
    row = df[df["종목코드"] == "005930"].iloc[0]
    assert row["권고"] == "관망"


def test_session_row_overrides_baseline(tmp_path):
    bot = _make_overlay_bot(tmp_path)
    # 세션 온디맨드 결과: 005930 최신화(권고 변경)
    _write_simple_csv(
        tmp_path / "result" / "result_simple.csv",
        [("005930", "삼성전자", "매수")],
    )
    df = bot._load_cached_result_simple()
    assert set(df["종목코드"]) == {"005930", "000660"}  # 베이스라인 + 세션 합집합
    samsung = df[df["종목코드"] == "005930"].iloc[0]
    assert samsung["권고"] == "매수"  # 세션이 베이스라인을 덮음
```

> 참고: 세션 경로를 `result/result_simple.csv`로 명시(`result_simple_path=...`)했으므로 `_allow_unvalidated_result_paths=True`가 되어 manifest 검증을 우회한다(기존 동작). published 베이스라인은 `published_store.load_published_simple`로 직접 읽는다.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k "baseline_served or session_row_overrides" -v`
Expected: FAIL — 베이스라인만 읽거나(세션 우선 미적용) published를 읽지 못해 빈 DF.

- [ ] **Step 3: Write minimal implementation**

`src/chatbot/kakao_colab_bot.py` 상단 import에 추가:

```python
from src.ops.published_store import load_published_simple
```

`_load_cached_result_simple`를 베이스라인+오버레이로 재작성. 기존 세션 로딩 본문은 내부 헬퍼 `_load_session_result_simple()`로 추출하고, 공개 메서드는 오버레이를 수행한다:

```python
    def _load_session_result_simple(self) -> pd.DataFrame:
        # (기존 _load_cached_result_simple 본문을 그대로 이동: _resolve_result_path +
        #  mtime 캐시 + 스키마 검증 → 세션 result/ 결과 DataFrame 반환, 없으면 빈 DF)
        ...

    def _load_cached_result_simple(self) -> pd.DataFrame:
        baseline = load_published_simple(self.published_dir)
        session = self._load_session_result_simple()
        if baseline.empty:
            return session
        if session.empty:
            return baseline.copy()
        baseline = baseline.copy()
        session = session.copy()
        baseline["종목코드"] = baseline["종목코드"].astype(str)
        session["종목코드"] = session["종목코드"].astype(str)
        session_codes = set(session["종목코드"])
        kept_baseline = baseline[~baseline["종목코드"].isin(session_codes)]
        merged = pd.concat([session, kept_baseline], ignore_index=True)
        return merged
```

구현 메모: 기존 `_load_cached_result_simple` 본문(라인 747-793)을 `_load_session_result_simple`로 이름만 바꿔 이동하고, 위 오버레이 메서드를 새로 추가한다. mtime 캐시 필드(`_result_simple_cache*`)는 세션 로더에 그대로 둔다.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k "baseline_served or session_row_overrides" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: 회귀 확인 + Commit**

Run: `python -m pytest tests/test_kakao_colab_bot.py -v`
Expected: 전체 통과(기존 캐시 동작 테스트 포함). 실패 시 세션 로더 추출 누락 점검.

```bash
git add src/chatbot/kakao_colab_bot.py tests/test_kakao_colab_bot.py
git commit -m "feat(chatbot): overlay session predictions on published baseline"
```

---

## Task 11: 봇 — detail/news/disclosure 경로도 세션 우선, published 폴백

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py:720-745` (`_latest_prediction_date_from_detail`) 및 `_resolve_result_path` 호출부
- Test: `tests/test_kakao_colab_bot.py`

`_latest_prediction_date_from_detail`이 세션 detail에 종목이 없으면 published detail에서도 찾도록 한다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kakao_colab_bot.py  (append)
def test_detail_date_falls_back_to_published(tmp_path):
    bot = _make_overlay_bot(tmp_path)
    # published detail에만 000660 존재
    detail = tmp_path / "published" / "latest" / "csv" / "result_detail.csv"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(
        "Symbol,Date\n000660.KS,2026-06-17\n", encoding="utf-8-sig"
    )
    assert bot._latest_prediction_date_from_detail("000660.KS") == "2026-06-17"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k "detail_date_falls_back" -v`
Expected: FAIL — 세션 detail이 없어 None 반환.

- [ ] **Step 3: Write minimal implementation**

`_latest_prediction_date_from_detail`에서 후보 경로를 세션→published 순으로 시도하도록 변경:

```python
    def _detail_path_candidates(self) -> list[Path]:
        candidates = [self._resolve_result_path("csv/result_detail.csv", self.result_detail_path)]
        published_detail = self.published_dir / "csv" / "result_detail.csv"
        candidates.append(published_detail)
        return [p for p in candidates if p and Path(p).exists()]

    def _latest_prediction_date_from_detail(self, symbol: str) -> str | None:
        normalized = normalize_user_symbols([symbol])
        symbol_aliases = {str(symbol)}
        if normalized:
            symbol_aliases.add(str(normalized[0]))
        display_code = self._display_code(symbol)
        symbol_aliases.update({f"{display_code}.KS", f"{display_code}.KQ"})
        for detail_path in self._detail_path_candidates():
            try:
                detail_df = pd.read_csv(detail_path, dtype={"Symbol": str}, encoding="utf-8-sig")
            except Exception:
                continue
            if detail_df.empty or "Symbol" not in detail_df.columns or "Date" not in detail_df.columns:
                continue
            matched = detail_df[detail_df["Symbol"].astype(str).isin(symbol_aliases)].copy()
            if matched.empty:
                continue
            matched["Date"] = pd.to_datetime(matched["Date"], errors="coerce")
            matched = matched.dropna(subset=["Date"])
            if matched.empty:
                continue
            return matched["Date"].max().strftime("%Y-%m-%d")
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k "detail_date_falls_back" -v`
Expected: PASS.

- [ ] **Step 5: 회귀 확인 + Commit**

Run: `python -m pytest tests/test_kakao_colab_bot.py -v`
Expected: 전체 통과.

```bash
git add src/chatbot/kakao_colab_bot.py tests/test_kakao_colab_bot.py
git commit -m "feat(chatbot): resolve detail date from session then published baseline"
```

---

## Task 12: Colab 러너 — `load_published_predictions`

**Files:**
- Modify: `colab/stock_predict_colab.py`
- Test: `tests/test_colab_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_colab_runner.py  (append; 기존 import 스타일에 맞춤)
from pathlib import Path

import pandas as pd

import colab.stock_predict_colab as colab_mod
from colab.stock_predict_colab import load_published_predictions


def _seed_published(root: Path, date: str):
    for sub in ("latest", f"history/{date}"):
        d = root / "published" / sub / "csv"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result_simple.csv").write_text(
            "종목코드,종목명\n005930,삼성전자\n", encoding="utf-8-sig"
        )
        (root / "published" / sub / "publish_meta.json").write_text(
            f'{{"trading_date": "{date}", "news_mode": "gemma", "symbol_count": 1}}',
            encoding="utf-8",
        )
    (root / "published" / "index.json").write_text(
        f'{{"latest": "{date}", "entries": [{{"trading_date": "{date}"}}]}}',
        encoding="utf-8",
    )


def test_load_published_predictions_latest(tmp_path, monkeypatch):
    _seed_published(tmp_path, "2026-06-17")
    monkeypatch.setattr(colab_mod, "PROJECT_ROOT", tmp_path)
    paths = load_published_predictions(date=None)
    assert Path(paths["result_simple_csv"]).exists()
    assert paths["trading_date"] == "2026-06-17"


def test_load_published_predictions_specific_date(tmp_path, monkeypatch):
    _seed_published(tmp_path, "2026-06-17")
    monkeypatch.setattr(colab_mod, "PROJECT_ROOT", tmp_path)
    paths = load_published_predictions(date="2026-06-17")
    assert "history/2026-06-17" in Path(paths["result_simple_csv"]).as_posix()


def test_load_published_predictions_missing_returns_empty_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_mod, "PROJECT_ROOT", tmp_path)
    paths = load_published_predictions(date="2099-01-01")
    assert paths["result_simple_csv"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_colab_runner.py -k "load_published" -v`
Expected: FAIL — `ImportError: cannot import name 'load_published_predictions'`.

- [ ] **Step 3: Write minimal implementation**

`colab/stock_predict_colab.py`에 추가(상단 import에 `from src.ops.published_store import resolve_published_dir, read_publish_meta` 추가):

```python
def load_published_predictions(date: str | None = None, rows: int = 5) -> dict[str, str]:
    """published/ 베이스라인을 읽어 경로를 반환하고 프리뷰를 출력한다. 파이프라인 미실행."""
    published_root = PROJECT_ROOT / "published"
    target = resolve_published_dir(published_root, date)
    simple = target / "csv" / "result_simple.csv"
    if not simple.exists():
        index_path = published_root / "index.json"
        print(f"[Colab] published 데이터가 없습니다: {target}")
        if index_path.exists():
            print(f"[Colab] 가용 인덱스: {index_path.read_text(encoding='utf-8')}")
        return {
            "result_simple_csv": "",
            "result_detail_csv": "",
            "report_json": "",
            "trading_date": "",
        }
    meta = read_publish_meta(target)
    _print_colab_preview(simple, rows=rows)
    print(f"[Colab] published baseline trading_date={meta.get('trading_date', '?')} "
          f"news_mode={meta.get('news_mode', '?')} symbols={meta.get('symbol_count', '?')}")
    return {
        "result_simple_csv": simple.as_posix(),
        "result_detail_csv": (target / "csv" / "result_detail.csv").as_posix(),
        "report_json": (target / "pipeline_report.json").as_posix(),
        "trading_date": str(meta.get("trading_date", "")),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_colab_runner.py -k "load_published" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add colab/stock_predict_colab.py tests/test_colab_runner.py
git commit -m "feat(colab): add load_published_predictions for github baseline serving"
```

---

## Task 13: 문서 갱신 (README + OPERATIONS)

**Files:**
- Modify: `README.md` (Kakao Bot 섹션 인근), `docs/OPERATIONS.md`

- [ ] **Step 1: README에 Daily Publish 섹션 추가**

README "Kakao Bot" 섹션 위 또는 아래에 추가:

````markdown
## Daily Publish (기본 200종목 → GitHub)

로컬에서 수동 1회 실행해 기본 200종목 예측을 `published/`에 게시하고 GitHub에 push한다.

```powershell
# gemma 서버(localhost:8001)가 떠 있으면 gemma 뉴스 임팩트, 아니면 규칙기반 폴백
stock-predict-publish                 # 증분 갱신 + 게시 + commit/push
stock-predict-publish --no-push       # 커밋까지만(푸시 안 함)
stock-predict-publish --dry-run       # 게시 파일만 만들고 commit/push 안 함
stock-predict-publish --news-mode rule --full-refresh
```

산출물:

- `published/latest/` — 최신 게시본(Colab 기본 읽기 대상)
- `published/history/<거래일>/` — 거래일별 스냅샷
- `published/index.json` — 가용 날짜·메타 인덱스

각 폴더는 `csv/result_*.csv`, `manifest.json`, `pipeline_report.json`, `publish_meta.json`을 포함한다.
뉴스/공시 점수는 표시용이며 `predicted_return`·추천·신호 정책에 영향을 주지 않는다.
````

- [ ] **Step 2: README Colab/Kakao 사용법을 published 기반으로 갱신**

"Kakao Bot" 섹션의 동작 설명을 다음 흐름으로 수정:

````markdown
Colab 기본 흐름(기준데이터 서빙):

```python
# 1) 최신 코드/기준데이터 받기
!git pull
# 2) GitHub 기준데이터 표시 (파이프라인 미실행)
from colab.stock_predict_colab import load_published_predictions
load_published_predictions()           # 최신; 특정일은 load_published_predictions("2026-06-17")
# 3) 봇 실행 (자동 부트스트랩 OFF, 기준데이터 베이스라인 서빙)
from src.chatbot.kakao_colab_bot import launch_colab_kakao_bot, PyngrokTunnelConfig
launch_colab_kakao_bot(tunnel_config=PyngrokTunnelConfig(auth_token="..."), prewarm_cache=False)
```

봇은 `published/latest/`를 기준으로 응답하며, 사용자가 종목코드/이름을 입력하거나 '최신화'를 요청할 때만
해당 종목을 세션에서 예측해 기준데이터 위에 덮어 보여준다(세션 한정, GitHub push 없음).
기본 200종목 재예측은 사용자가 명시적으로 `run_colab_pipeline(...)`을 호출할 때만 수행한다.
````

- [ ] **Step 3: docs/OPERATIONS.md에 publish 절차 한 단락 추가**

`docs/OPERATIONS.md`에 "일일 게시(Publish)" 절을 추가하고 위 명령/산출물/폴백 동작을 1문단으로 요약한다(README와 중복 최소화, 링크로 연결).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/OPERATIONS.md
git commit -m "docs(publish): document daily publish and published-baseline colab flow"
```

---

## Task 14: 전체 회귀 + 스모크 검증

**Files:** 없음(검증만)

- [ ] **Step 1: 전체 테스트**

Run: `python -m pytest -q`
Expected: 전체 통과. 실패가 있으면 해당 Task로 돌아가 수정.

- [ ] **Step 2: publish dry-run 스모크 (네트워크 가능 환경)**

Run: `stock-predict-publish --dry-run --news-mode rule`
Expected: `published/latest/`, `published/history/<오늘 거래일>/`, `published/index.json` 생성. git 변경 없음(dry-run). 콘솔에 meta JSON 출력.

> 주의: 이 스모크는 실데이터 fetch(yfinance)를 수행하므로 네트워크가 필요하다. 네트워크 불가 시 이 단계는 생략하고 Step 1 단위테스트로 대체한다.

- [ ] **Step 3: 봇 베이스라인 서빙 수동 확인**

`published/latest/`가 있는 상태에서 봇을 임포트해 `_load_cached_result_simple()`이 베이스라인을 반환하는지 빠르게 확인(또는 Task 10 테스트로 대체).

- [ ] **Step 4: 최종 커밋(있으면)**

```bash
git add -A
git commit -m "test(publish): verify full suite and publish smoke" || echo "nothing to commit"
```

---

## 완료 기준 (spec 대응)

- 요구사항 #1: `stock-predict-publish`가 200종목 예측을 `published/`에 게시·push → Task 5–8, 13.
- 요구사항 #2: Colab이 `published/`를 읽어 결과 표시 → Task 12.
- 요구사항 #3: Colab 기본 = published 베이스라인, 자동 부트스트랩 OFF, 200 재실행은 명시 요청 시만 → Task 9, 12, 13.
- 요구사항 #4: 사용자 요청 종목만 세션 예측 + 오버레이(push 없음) → Task 10, 11.
- 히스토리: `published/history/<거래일>/` + `index.json` → Task 3, 5, 12.
- gemma 배치 + 규칙기반 폴백, 표시용 점수 → Task 7(`_news_config_for_mode`, `_effective_news_mode`).
