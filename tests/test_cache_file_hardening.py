from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.news_impact.data_cache import DataCache
from src.news_impact.llm_client import FileLLMResponseCache
from src.reports.output import safe_to_csv


def test_critical_safe_to_csv_raises_without_fallback_when_target_locked(monkeypatch, tmp_path: Path):
    target = tmp_path / "result_simple.csv"
    calls: list[Path] = []
    original_replace = Path.replace

    def _locked_replace(self: Path, target_path: Path):
        calls.append(Path(target_path))
        if Path(target_path) == target:
            raise PermissionError("locked")
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", _locked_replace)

    with pytest.raises(PermissionError):
        safe_to_csv(pd.DataFrame([{"a": 1}]), target, allow_fallback=False)

    assert calls == [target]
    assert not (tmp_path / "result_simple_fallback.csv").exists()


def test_noncritical_safe_to_csv_still_uses_fallback_when_target_locked(monkeypatch, tmp_path: Path):
    target = tmp_path / "result_news.csv"
    original_replace = Path.replace

    def _locked_replace(self: Path, target_path: Path):
        if Path(target_path) == target:
            raise PermissionError("locked")
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", _locked_replace)

    saved = safe_to_csv(pd.DataFrame([{"a": 1}]), target)

    assert saved == tmp_path / "result_news_fallback.csv"
    assert saved.exists()


def test_file_llm_response_cache_writes_via_replace(monkeypatch, tmp_path: Path):
    replaced: list[Path] = []
    original_replace = Path.replace

    def _record_replace(self: Path, target_path: Path):
        replaced.append(Path(target_path))
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", _record_replace)

    cache = FileLLMResponseCache(tmp_path)
    cache.set("abc", {"판단": "중립"})

    assert replaced == [tmp_path / "abc.json"]
    assert json.loads((tmp_path / "abc.json").read_text(encoding="utf-8")) == {"판단": "중립"}


def test_data_cache_writes_json_via_replace(monkeypatch, tmp_path: Path):
    replaced: list[Path] = []
    original_replace = Path.replace

    def _record_replace(self: Path, target_path: Path):
        replaced.append(Path(target_path))
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", _record_replace)

    cache = DataCache(tmp_path)
    path = cache.write_market_price("test", "005930.KS", pd.Timestamp("2026-06-05").date(), {"close": 100})

    assert replaced == [path]
    assert json.loads(path.read_text(encoding="utf-8"))["close"] == 100
