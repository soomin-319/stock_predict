from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class APIKeys:
    naver_client_id: str | None
    naver_client_secret: str | None
    opendart_api_key: str | None


def find_dotenv(start: str | Path | None = None) -> Path | None:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path) if path is not None else find_dotenv()
    if env_path is None or not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_optional_quotes(value.strip())
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def load_api_keys(path: str | Path | None = None) -> APIKeys:
    load_dotenv(path)
    return APIKeys(
        naver_client_id=_empty_to_none(os.environ.get("NAVER_CLIENT_ID")),
        naver_client_secret=_empty_to_none(os.environ.get("NAVER_CLIENT_SECRET")),
        opendart_api_key=_empty_to_none(os.environ.get("OPENDART_API_KEY")),
    )


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _empty_to_none(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value
