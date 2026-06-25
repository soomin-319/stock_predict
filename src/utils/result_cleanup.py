"""`result/` 산출물 보존·정리 정책 (단일 출처).

이 모듈은 `result/` 트리의 보존 기간·정리 대상·보호 규칙을 코드로 강제하는 동시에
운영 정책의 명문화된 단일 출처다. CODEBASE_ANALYSIS §11(산출물 라이프사이클)은 이
docstring을 정책의 권위 있는 참조로 가리킨다.

디렉터리 레이아웃과 정리 대상
-----------------------------
`cleanup_result_artifacts(result_root, policy)`가 정리하는 하위 트리는 셋뿐이다.

- `result/runs/<run_id>/` — 실행별 원본 산출물.  `cleanup_runs`가 처리한다.
- `result/test/<session>/` — 테스트 세션 아티팩트.  `cleanup_test_artifacts`가 처리한다.
- `result/runtime/logs/` — 런타임 로그 파일.  `cleanup_logs`가 처리한다.

승격된 공식 최신본 `result/latest/`와 `result/runtime/`의 레지스트리 파일
(`chatbot_jobs.json` 등)은 정리 대상이 **아니다**.  레지스트리 항목의 TTL 만료는
`prune_registry`로 별도 관리한다.

보존 기간 (`RetentionPolicy` 기본값)
-----------------------------------
- `successful_run_count=10` — 성공 run은 수정시각 기준 최신 10개만 보존하고, 이를
  넘는 더 오래된 성공 run은 나이와 무관하게 제거한다.
- `successful_run_days=30` — 보존 개수 안에 들어도 30일을 초과한 성공 run은 제거한다.
- `failed_run_days=30` — 실패 run(`manifest.status == "fail"` 또는 manifest 판독 실패)은
  30일 초과 시 제거한다.  `result/test/`도 같은 30일 TTL을 쓴다.
- `runtime_log_days=14` — `result/runtime/logs/`의 로그 파일은 14일 초과 시 제거한다.

run의 나이는 `manifest.json`(있으면) 또는 디렉터리 자체의 mtime으로 판정한다
(`_modified_at`).

보호 규칙 (안전망)
-----------------
- `result/latest_manifest.json` 또는 `result/latest/manifest.json`이 가리키는
  `run_id`의 run은 보존 정책과 무관하게 **절대 제거하지 않는다**(`_protected_latest_run_ids`).
- `_remove`는 허용 루트 내부가 아닌 경로, 그리고 허용 루트 자신은 삭제를 거부한다
  (`refusing cleanup outside allowed root`).  심볼릭링크/상위경로 탈출을 차단한다.

정리 주기 (수동 호출)
--------------------
이 정리 루틴은 메인 파이프라인이나 콘솔 진입점에서 **자동 호출되지 않는다**.  운영자가
명시적으로 `cleanup_result_artifacts(...)`를 호출하거나 외부 스케줄러로 돌려야 한다.
자동화하려면 이 함수를 운영 cron/CI 잡에 배선할 것.

용량 상한
---------
바이트 단위의 하드 용량 상한은 없다.  디스크 사용량은 (1) 성공 run 개수 상한
(`successful_run_count`)과 (2) 위 나이 기반 TTL로 **간접적으로만** 제한된다.  엄격한
용량 캡이 필요하면 `RetentionPolicy`에 크기 예산을 추가하고 `cleanup_runs`에서
최신순으로 누적 크기를 집계해 초과분을 제거하는 방식으로 확장할 수 있다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetentionPolicy:
    successful_run_count: int = 10
    successful_run_days: int = 30
    failed_run_days: int = 30
    runtime_log_days: int = 14


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def prune_registry(
    data: dict[str, Any],
    *,
    timestamp_field: str,
    ttl: timedelta,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    result = {}
    for key, value in data.items():
        timestamp = parse_utc(value.get(timestamp_field)) if isinstance(value, dict) else None
        if timestamp is not None and current - timestamp <= ttl:
            result[key] = value
    return result


def _inside(target: Path, root: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return target.resolve() != root.resolve()
    except ValueError:
        return False


def _remove(target: Path, allowed_root: Path) -> str:
    if not _inside(target, allowed_root):
        raise ValueError(f"refusing cleanup outside allowed root: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return str(target)


def _modified_at(path: Path) -> datetime:
    marker = path / "manifest.json" if path.is_dir() and (path / "manifest.json").exists() else path
    return datetime.fromtimestamp(marker.stat().st_mtime, tz=timezone.utc)


def _protected_latest_run_ids(runs_root: Path) -> set[str]:
    protected: set[str] = set()
    for marker in (runs_root.parent / "latest_manifest.json", runs_root.parent / "latest" / "manifest.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if run_id:
            protected.add(str(run_id))
    return protected


def cleanup_runs(runs_root: Path, policy: RetentionPolicy, now: datetime) -> list[str]:
    if not runs_root.exists():
        return []
    protected_run_ids = _protected_latest_run_ids(runs_root)
    successful: list[Path] = []
    failed: list[Path] = []
    for run in (path for path in runs_root.iterdir() if path.is_dir()):
        try:
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            manifest = {"status": "fail"}
        (failed if manifest.get("status") == "fail" else successful).append(run)
    successful.sort(key=_modified_at, reverse=True)
    removed = []
    for index, run in enumerate(successful):
        if run.name in protected_run_ids:
            continue
        age = now - _modified_at(run)
        if index >= policy.successful_run_count or age > timedelta(days=policy.successful_run_days):
            removed.append(_remove(run, runs_root))
    for run in failed:
        if run.name in protected_run_ids:
            continue
        if now - _modified_at(run) > timedelta(days=policy.failed_run_days):
            removed.append(_remove(run, runs_root))
    return removed


def cleanup_logs(log_root: Path, policy: RetentionPolicy, now: datetime) -> list[str]:
    if not log_root.exists():
        return []
    removed = []
    for path in (item for item in log_root.iterdir() if item.is_file()):
        if now - _modified_at(path) > timedelta(days=policy.runtime_log_days):
            removed.append(_remove(path, log_root))
    return removed


def cleanup_test_artifacts(test_root: Path, policy: RetentionPolicy, now: datetime) -> list[str]:
    if not test_root.exists():
        return []
    removed = []
    for artifact in test_root.iterdir():
        file_times = [_modified_at(path) for path in artifact.rglob("*") if path.is_file()] if artifact.is_dir() else []
        modified_at = max(file_times) if file_times else _modified_at(artifact)
        if now - modified_at > timedelta(days=policy.failed_run_days):
            removed.append(_remove(artifact, test_root))
    return removed


def cleanup_result_artifacts(
    result_root: Path,
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    result_root = Path(result_root)
    removed = []
    removed.extend(cleanup_runs(result_root / "runs", policy, current))
    removed.extend(cleanup_test_artifacts(result_root / "test", policy, current))
    removed.extend(cleanup_logs(result_root / "runtime" / "logs", policy, current))
    return {"removed": removed, "removed_count": len(removed)}


__all__ = [
    "RetentionPolicy",
    "cleanup_result_artifacts",
    "cleanup_logs",
    "cleanup_runs",
    "cleanup_test_artifacts",
    "parse_utc",
    "prune_registry",
]
