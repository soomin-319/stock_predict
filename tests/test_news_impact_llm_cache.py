from __future__ import annotations

import json
from pathlib import Path

from src.news_impact.llm_client import (
    FileLLMResponseCache,
    LlamaCppClient,
    _cache_key,
    sha256_text,
)
from src.news_impact.llm_config import LLMConfig


def test_cache_set_with_metadata_round_trips_response(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path)
    response = {"event_type": "earnings", "direction": "positive"}

    cache.set("k1", response, metadata={"model": "gemma", "prompt_hash": "abc"})

    assert cache.get("k1") == response


def test_cache_set_with_metadata_persists_metadata_on_disk(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path)

    cache.set("k1", {"x": 1}, metadata={"model": "gemma", "prompt_hash": "abc"})

    stored = json.loads((tmp_path / "k1.json").read_text(encoding="utf-8"))
    assert stored["schema"] == "stock-news-impact.llm_cache.v1"
    assert stored["metadata"]["model"] == "gemma"
    assert stored["metadata"]["prompt_hash"] == "abc"
    assert stored["response"] == {"x": 1}


def test_cache_without_metadata_stays_bare_and_readable(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path)

    cache.set("k1", {"판단": "중립"})

    stored = json.loads((tmp_path / "k1.json").read_text(encoding="utf-8"))
    assert stored == {"판단": "중립"}
    assert cache.get("k1") == {"판단": "중립"}


def test_cache_get_reads_legacy_bare_dict(tmp_path: Path):
    # A cache file written before the envelope format existed.
    (tmp_path / "legacy.json").write_text(
        json.dumps({"direction": "negative"}), encoding="utf-8"
    )
    cache = FileLLMResponseCache(tmp_path)

    assert cache.get("legacy") == {"direction": "negative"}


def test_cache_get_returns_none_for_expired_enveloped_entry(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path, ttl_seconds=60)
    (tmp_path / "expired.json").write_text(
        json.dumps(
            {
                "schema": "stock-news-impact.llm_cache.v1",
                "metadata": {"cached_at": "2000-01-01T00:00:00+00:00"},
                "response": {"direction": "positive"},
            }
        ),
        encoding="utf-8",
    )

    assert cache.get("expired") is None


def test_cache_get_returns_none_when_expected_metadata_differs(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path)
    cache.set("k1", {"direction": "positive"}, metadata={"model": "gemma"})

    assert cache.get("k1", expected_metadata={"model": "other"}) is None
    assert cache.get("k1", expected_metadata={"model": "gemma"}) == {
        "direction": "positive"
    }


def test_cache_set_prunes_oldest_files_when_max_entries_exceeded(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path, max_entries=2)

    cache.set("k1", {"x": 1})
    cache.set("k2", {"x": 2})
    cache.set("k3", {"x": 3})

    remaining = sorted(path.stem for path in tmp_path.glob("*.json"))
    assert remaining == ["k2", "k3"]


class _FakeTransport:
    def __init__(self) -> None:
        self.posts: list[dict] = []

    def get_json(self, url, timeout_seconds):
        return {"data": [{"id": "gpt-5-mini"}]}

    def post_json(self, url, payload, timeout_seconds):
        self.posts.append(payload)
        return {
            "model": "gpt-5-mini",
            "choices": [{"message": {"content": json.dumps({"direction": "positive"})}}],
        }


def test_chat_json_records_reproducibility_metadata_in_cache(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path)
    client = LlamaCppClient(LLMConfig.default(), transport=_FakeTransport(), cache=cache)

    client.chat_json("SYSTEM PROMPT", "USER PROMPT", required_keys=("direction",))

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text(encoding="utf-8"))
    meta = stored["metadata"]
    assert meta["model"] == "gpt-5-mini"
    assert meta["temperature"] == 0.1
    assert meta["prompt_hash"] == sha256_text("SYSTEM PROMPT")
    assert meta["article_hash"] == sha256_text("USER PROMPT")
    assert meta["required_keys"] == ["direction"]
    assert stored["response"] == {"direction": "positive"}


def test_chat_json_ignores_cache_when_metadata_is_stale(tmp_path: Path):
    cache = FileLLMResponseCache(tmp_path)
    payload = {
        "model": "gpt-5-mini",
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "USER PROMPT"},
        ],
        "response_format": {"type": "json_object"},
    }
    stale_key = _cache_key(payload, required_keys=("direction",))
    cache.set(
        stale_key,
        {"direction": "negative"},
        metadata={
            "model": "stale-model",
            "temperature": 0.1,
            "prompt_hash": sha256_text("SYSTEM PROMPT"),
            "article_hash": sha256_text("USER PROMPT"),
            "required_keys": ["direction"],
        },
    )
    assert cache.get(
        stale_key,
    ) == {"direction": "negative"}
    transport = _FakeTransport()
    client = LlamaCppClient(LLMConfig.default(), transport=transport, cache=cache)

    result = client.chat_json("SYSTEM PROMPT", "USER PROMPT", required_keys=("direction",))

    assert result == {"direction": "positive"}
    assert len(transport.posts) == 1
