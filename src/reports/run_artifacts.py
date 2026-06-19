from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.reports.output import safe_to_csv
from src.reports.report_metadata import artifact_schema_version
from src.utils.atomic_files import atomic_write_text

REQUIRED_ARTIFACTS = (
    "csv/result_simple.csv",
    "csv/result_detail.csv",
    "csv/result_news.csv",
    "csv/result_disclosure.csv",
    "pm_report.json",
    "pipeline_report.json",
)
LATEST_POINTER = "latest_manifest.json"

COMPATIBILITY_COPIES = {
    "csv/result_simple.csv": "result_simple.csv",
    "csv/result_detail.csv": "result_detail.csv",
    "csv/result_news.csv": "result_news.csv",
    "csv/result_disclosure.csv": "result_disclosure.csv",
    "pm_report.json": "pm_report.json",
}
SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int | None:
    try:
        return int(len(pd.read_csv(path, encoding="utf-8-sig")))
    except Exception:
        return None


def _csv_columns(path: Path) -> list[str] | None:
    try:
        return [str(column) for column in pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns]
    except Exception:
        return None


def _artifact_entries(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item.name != "manifest.json"):
        stat = path.stat()
        entry = {
            "relative_path": path.relative_to(run_dir).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": int(stat.st_size),
            "generated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
        if path.suffix.lower() == ".csv":
            entry["row_count"] = _csv_row_count(path)
            entry["columns"] = _csv_columns(path) or []
            entry["schema_kind"] = path.stem
            entry["schema_version"] = artifact_schema_version(path.stem)
        entries.append(entry)
    return entries


def _replace_directory(source: Path, target: Path) -> None:
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.bak")
    shutil.copytree(source, temp)
    try:
        if target.exists():
            target.replace(backup)
        temp.replace(target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if backup.exists():
            backup.replace(target)
        raise
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def _copy_compatibility_files(result_root: Path) -> None:
    latest = result_root / "latest"
    for source_name, target_name in COMPATIBILITY_COPIES.items():
        source = latest / source_name
        target = result_root / target_name
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(source, temp)
        temp.replace(target)


def _write_latest_pointer(result_root: Path, manifest: dict[str, Any]) -> None:
    run_id = str(manifest["run_id"])
    payload = {
        "run_id": run_id,
        "run_dir": f"runs/{run_id}",
        "manifest_path": f"runs/{run_id}/manifest.json",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(
        result_root / LATEST_POINTER,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_latest_run_dir(result_root: Path) -> Path | None:
    pointer_path = Path(result_root) / LATEST_POINTER
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_id = str(pointer["run_id"])
    except Exception:
        try:
            pointer = json.loads((Path(result_root) / "latest" / "manifest.json").read_text(encoding="utf-8"))
            run_id = str(pointer["run_id"])
        except Exception:
            return None
    if not SAFE_RUN_ID_PATTERN.fullmatch(run_id):
        return None
    run_dir = Path(result_root) / "runs" / run_id
    return run_dir if run_dir.exists() else None


class RunArtifactManager:
    def __init__(self, result_root: Path, metadata: dict[str, Any]):
        self.result_root = Path(result_root)
        self.metadata = dict(metadata)
        run_id = str(self.metadata["run_id"])
        if not SAFE_RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id}")
        self.run_dir = self.result_root / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def path(self, relative_path: str) -> Path:
        target = (self.run_dir / relative_path).resolve()
        try:
            target.relative_to(self.run_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"artifact path outside run directory: {relative_path}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> Path:
        target = self.path(relative_path)
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    def write_csv(self, relative_path: str, frame: pd.DataFrame) -> Path:
        return safe_to_csv(frame, self.path(relative_path), allow_fallback=False)

    def build_manifest(self) -> dict[str, Any]:
        return {**self.metadata, "artifacts": _artifact_entries(self.run_dir)}

    def validate(self, manifest: dict[str, Any] | None = None) -> tuple[str, list[str]]:
        payload = manifest or self.build_manifest()
        present = {item["relative_path"] for item in payload["artifacts"]}
        reasons = [f"missing_artifact:{name}" for name in REQUIRED_ARTIFACTS if name not in present]
        for report_name in ("pipeline_report.json", "pm_report.json"):
            path = self.run_dir / report_name
            if not path.exists():
                continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                reasons.append(f"invalid_json:{report_name}")
                continue
            if report.get("run_id") != self.metadata.get("run_id"):
                reasons.append(f"run_id_mismatch:{report_name}")
        return ("fail" if reasons else str(self.metadata.get("status", "pass"))), reasons

    def finalize(self) -> dict[str, Any]:
        manifest = self.build_manifest()
        status, reasons = self.validate(manifest)
        manifest["status"] = status
        manifest["blocking_reasons"] = list(dict.fromkeys([*manifest.get("blocking_reasons", []), *reasons]))
        promotable = (
            status in {"pass", "warning"}
            and self.metadata.get("environment") == "production"
            and self.metadata.get("data_mode") == "real"
        )
        manifest["promoted"] = bool(promotable)
        self.write_json("manifest.json", manifest)
        if promotable:
            _replace_directory(self.run_dir, self.result_root / "latest")
            _write_latest_pointer(self.result_root, manifest)
            _copy_compatibility_files(self.result_root)
        return manifest


__all__ = [
    "COMPATIBILITY_COPIES",
    "LATEST_POINTER",
    "REQUIRED_ARTIFACTS",
    "RunArtifactManager",
    "resolve_latest_run_dir",
]
