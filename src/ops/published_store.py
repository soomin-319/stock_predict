from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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
