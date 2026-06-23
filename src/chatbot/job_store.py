from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.chatbot.session_store import LockLike, save_registry


@dataclass(slots=True)
class PredictionJobState:
    symbol: str
    display_code: str
    command: list[str]
    log_path: str
    submitted_at: str
    status: str = "running"
    pid: int | None = None
    exit_code: int | None = None
    completed_at: str | None = None


class ChatbotJobStore:
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

    def save(self) -> None:
        save_registry(self.path, self._registry, self._secret_values)

    def get(self, symbol: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._registry.get(symbol, {}))

    def set(self, symbol: str, state: PredictionJobState | dict[str, Any]) -> dict[str, Any]:
        next_state = asdict(state) if isinstance(state, PredictionJobState) else dict(state)
        with self._lock:
            self._registry[symbol] = next_state
            self.save()
            return dict(next_state)

    def running_prediction_count(self, bootstrap_key: str) -> int:
        with self._lock:
            return sum(
                1
                for key, state in self._registry.items()
                if key != bootstrap_key and state.get("status") == "running"
            )

    def mark_failed(self, symbol: str, exit_code: int, note: str = "") -> dict[str, Any]:
        with self._lock:
            job_state = dict(self._registry.get(symbol, {}))
            job_state.update(
                {
                    "status": "failed",
                    "exit_code": int(exit_code),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "failure_note": str(note or ""),
                }
            )
            job_state.pop("command", None)
            job_state.pop("pid", None)
            self._registry[symbol] = job_state
            self.save()
            return dict(job_state)

    def mark_completed(self, symbol: str, exit_code: int) -> dict[str, Any]:
        status = "completed" if exit_code == 0 else "failed"
        with self._lock:
            job_state = dict(self._registry.get(symbol, {}))
            job_state.update(
                {
                    "status": status,
                    "exit_code": int(exit_code),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            job_state.pop("command", None)
            job_state.pop("pid", None)
            self._registry[symbol] = job_state
            self.save()
            return dict(job_state)

    def elapsed_seconds(self, job_state: dict[str, Any]) -> float | None:
        completed_at = str(job_state.get("completed_at") or "").strip()
        if not completed_at:
            return None
        try:
            completed_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except Exception:
            return None
        now_utc = datetime.now(timezone.utc)
        if completed_dt.tzinfo is None:
            completed_dt = completed_dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now_utc - completed_dt.astimezone(timezone.utc)).total_seconds())

    def mark_stale_running_on_startup(self) -> bool:
        changed = False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for state in self._registry.values():
                if state.get("status") != "running":
                    continue
                state["status"] = "failed"
                state["exit_code"] = -2
                state["completed_at"] = now
                state["failure_note"] = "stale_running_on_startup"
                state.pop("command", None)
                state.pop("pid", None)
                changed = True
            if changed:
                self.save()
        return changed
