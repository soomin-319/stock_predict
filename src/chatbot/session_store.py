from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from src.utils.atomic_files import atomic_write_text
from src.utils.secrets import redact_value


class LockLike(Protocol):
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


@dataclass(slots=True)
class UserSessionState:
    user_id: str
    last_symbol: str | None = None
    last_display_code: str | None = None
    last_intent: str = "idle"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def save_registry(path: Path, data: dict[str, dict[str, Any]], secret_values: Iterable[str]) -> None:
    safe_data = redact_value(data, secret_values)
    atomic_write_text(path, json.dumps(safe_data, ensure_ascii=False, indent=2), encoding="utf-8")


class ChatbotSessionStore:
    def __init__(
        self,
        path: Path,
        registry: dict[str, dict[str, Any]] | None,
        lock: LockLike,
        secret_values: Iterable[str],
    ) -> None:
        self.path = path
        self._registry = registry if registry is not None else {}
        self._lock = lock
        self._secret_values = tuple(secret_values)

    @property
    def data(self) -> dict[str, dict[str, Any]]:
        return self._registry

    def update(self, user_id: str | None, symbol: str | None, display_code: str | None, intent: str) -> None:
        if not user_id:
            return
        with self._lock:
            self._registry[user_id] = asdict(
                UserSessionState(
                    user_id=user_id,
                    last_symbol=symbol,
                    last_display_code=display_code,
                    last_intent=intent,
                )
            )
            save_registry(self.path, self._registry, self._secret_values)

    def symbol_for(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        with self._lock:
            session = self._registry.get(user_id, {})
        symbol = session.get("last_symbol")
        return str(symbol) if symbol else None

    def intent_for(self, user_id: str | None) -> str:
        if not user_id:
            return ""
        with self._lock:
            session = self._registry.get(user_id, {})
        return str(session.get("last_intent") or "").strip()
