from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

REDACTED = "[REDACTED]"

SECRET_FLAGS = {
    "--openai-api-key",
    "--dart-api-key",
    "--naver-client-id",
    "--naver-client-secret",
}
_SECRET_NAME_PATTERN = re.compile(r"(?:api[-_]?key|token|secret|password)", re.IGNORECASE)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<name>(?:--)?[A-Za-z0-9_-]*(?:api[-_]?key|token|secret|password)[A-Za-z0-9_-]*)"
    r"(?P<separator>\s*=\s*|\s+)(?P<value>[^\s]+)",
    re.IGNORECASE,
)


def _secret_values(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}, key=len, reverse=True))


def redact_text(text: object, secret_values: Iterable[object] = ()) -> str:
    result = str(text)
    for value in _secret_values(secret_values):
        result = result.replace(value, REDACTED)
    return _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group('name')}{match.group('separator')}{REDACTED}",
        result,
    )


def redact_argv(argv: Iterable[object], secret_values: Iterable[object] = ()) -> list[str]:
    values = _secret_values(secret_values)
    out: list[str] = []
    mask_next = False
    for raw_value in argv:
        raw = str(raw_value)
        lower = raw.lower()
        if mask_next:
            out.append(REDACTED)
            mask_next = False
            continue
        if "=" in raw:
            name, _value = raw.split("=", 1)
            if name.lower() in SECRET_FLAGS or _SECRET_NAME_PATTERN.search(name):
                out.append(f"{name}={REDACTED}")
                continue
        if lower in SECRET_FLAGS or _SECRET_NAME_PATTERN.search(lower):
            out.append(raw)
            mask_next = True
            continue
        out.append(redact_text(raw, values))
    return out


def redact_value(value: Any, secret_values: Iterable[object] = ()) -> Any:
    if isinstance(value, dict):
        return {key: redact_value(item, secret_values) for key, item in value.items()}
    if isinstance(value, list):
        return redact_argv(value, secret_values)
    if isinstance(value, tuple):
        return tuple(redact_argv(value, secret_values))
    if isinstance(value, str):
        return redact_text(value, secret_values)
    return value


__all__ = ["REDACTED", "redact_argv", "redact_text", "redact_value"]
