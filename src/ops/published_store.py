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
