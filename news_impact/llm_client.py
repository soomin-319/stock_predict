from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib import error as urlerror, request

from news_impact.llm_config import LLMConfig


class LLMResponseError(RuntimeError):
    """Raised when an LLM response violates expected response shape."""


class LLMModelAliasError(RuntimeError):
    """Raised when the configured model alias is not served by the runtime."""


class JsonTransport(Protocol):
    def get_json(
        self,
        url: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        ...

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        ...


class LLMResponseCache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None:
        ...

    def set(self, key: str, value: dict[str, Any]) -> None:
        ...


class FileLLMResponseCache:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise LLMResponseError("Cached LLM response must be an object")
        return payload

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.json"


class UrllibJsonTransport:
    def get_json(
        self,
        url: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return self._request_json(url=url, timeout_seconds=timeout_seconds, method="GET")

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return self._request_json(
            url=url,
            timeout_seconds=timeout_seconds,
            method="POST",
            body=body,
        )

    def _request_json(
        self,
        url: str,
        timeout_seconds: float,
        method: str,
        body: bytes | None = None,
    ) -> dict[str, Any]:
        http_request = request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
        parsed = json.loads(response_body)
        if not isinstance(parsed, dict):
            raise LLMResponseError("HTTP JSON response must be an object")
        return parsed


class LlamaCppClient:
    def __init__(
        self,
        config: LLMConfig,
        transport: JsonTransport | None = None,
        cache: LLMResponseCache | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self.last_requested_model: str | None = None
        self.last_response_model: str | None = None

    def verify_model_alias(self) -> None:
        response = self._transport.get_json(
            url=f"{self._config.base_url}/models",
            timeout_seconds=self._config.timeout_seconds,
        )
        served_models = _extract_model_ids(response)
        if self._config.model not in served_models:
            available = ", ".join(sorted(served_models)) or "<none>"
            raise LLMModelAliasError(
                f"Configured model alias '{self._config.model}' not found. "
                f"Available models: {available}"
            )

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._config.json_schema_required:
            payload["response_format"] = {"type": "json_object"}

        cache_key = _cache_key(payload, required_keys)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                _validate_required_keys(cached, required_keys)
                return cached

        attempts = self._config.max_retries + 1
        self.last_requested_model = self._config.model
        self.last_response_model = None
        last_error: Exception | None = None
        for attempt_index in range(attempts):
            try:
                response = self._transport.post_json(
                    url=f"{self._config.base_url}/chat/completions",
                    payload=payload,
                    timeout_seconds=self._config.timeout_seconds,
                )
                self.last_response_model = _extract_response_model(response)
                parsed = _parse_assistant_json(response)
                _validate_required_keys(parsed, required_keys)
                if self._cache is not None:
                    self._cache.set(cache_key, parsed)
                return parsed
            except LLMResponseError as error:
                last_error = error
            except Exception as error:  # network/HTTP transient path; validation errors should not land here
                if not _is_transient_error(error) or attempt_index == attempts - 1:
                    raise
                last_error = error
        if last_error is not None:
            raise last_error
        raise LLMResponseError("LLM request was not attempted")


def _extract_assistant_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMResponseError("LLM response missing choices[0].message.content") from error
    if not isinstance(content, str):
        raise LLMResponseError("Assistant content must be a string")
    return content


def _parse_assistant_json(response: dict[str, Any]) -> dict[str, Any]:
    content = _extract_assistant_content(response)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise LLMResponseError("Assistant content was not valid JSON") from error
    if not isinstance(parsed, dict):
        raise LLMResponseError("Assistant JSON content must be an object")
    return parsed


def _extract_model_ids(response: dict[str, Any]) -> set[str]:
    data = response.get("data")
    if not isinstance(data, list):
        raise LLMResponseError("Models response missing data list")
    model_ids: set[str] = set()
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.add(item["id"])
    return model_ids


def _extract_response_model(response: dict[str, Any]) -> str | None:
    model = response.get("model")
    if isinstance(model, str):
        return model
    return None


def _validate_required_keys(
    parsed: dict[str, Any],
    required_keys: Iterable[str] | None,
) -> None:
    if required_keys is None:
        return
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        raise LLMResponseError(f"Assistant JSON missing required keys: {', '.join(missing)}")


def _cache_key(payload: dict[str, Any], required_keys: Iterable[str] | None) -> str:
    cache_payload = {
        "payload": payload,
        "required_keys": sorted(required_keys or ()),
    }
    encoded = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_transient_error(error: Exception) -> bool:
    if isinstance(error, urlerror.HTTPError):
        return error.code == 429 or 500 <= error.code <= 599
    return isinstance(error, (TimeoutError, urlerror.URLError, OSError))
