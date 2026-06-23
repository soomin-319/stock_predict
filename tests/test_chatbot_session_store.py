from __future__ import annotations

import json
import threading
from pathlib import Path

from src.chatbot.session_store import ChatbotSessionStore, load_registry, save_registry


def test_load_registry_returns_empty_for_missing_invalid_or_non_dict(tmp_path: Path):
    missing = tmp_path / "missing.json"
    assert load_registry(missing) == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad-json", encoding="utf-8")
    assert load_registry(invalid) == {}

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_registry(non_dict) == {}


def test_load_registry_filters_non_dict_entries(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"ok": {"status": "completed"}, "bad": "value"}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert load_registry(path) == {"ok": {"status": "completed"}}


def test_save_registry_redacts_secret_values(tmp_path: Path):
    path = tmp_path / "registry.json"

    save_registry(path, {"job": {"token": "secret-token", "status": "running"}}, ["secret-token"])

    saved = path.read_text(encoding="utf-8")
    assert "secret-token" not in saved
    assert "[REDACTED]" in saved


def test_chatbot_session_store_updates_and_reads_session(tmp_path: Path):
    path = tmp_path / "sessions.json"
    store = ChatbotSessionStore(path=path, registry={}, lock=threading.RLock(), secret_values=[])

    store.update("user-1", symbol="005930.KS", display_code="005930", intent="tracking")

    assert store.symbol_for("user-1") == "005930.KS"
    assert store.intent_for("user-1") == "tracking"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["user-1"]["last_symbol"] == "005930.KS"
    assert saved["user-1"]["last_display_code"] == "005930"
    assert saved["user-1"]["last_intent"] == "tracking"
    assert saved["user-1"]["updated_at"]


def test_chatbot_session_store_missing_user_defaults(tmp_path: Path):
    store = ChatbotSessionStore(path=tmp_path / "sessions.json", registry={}, lock=threading.RLock(), secret_values=[])

    assert store.symbol_for(None) is None
    assert store.symbol_for("missing") is None
    assert store.intent_for(None) == ""
    assert store.intent_for("missing") == ""
