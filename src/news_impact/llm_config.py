from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.news_impact.env_config import find_dotenv, load_dotenv


OLLAMA_ONLY_KEYS = {"ollama_base_url", "ollama_model"}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    temperature: float
    max_retries: int
    json_schema_required: bool
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 60.0

    @classmethod
    def default(cls) -> "LLMConfig":
        return cls(
            provider="llama_cpp",
            base_url="http://localhost:8001/v1",
            model="gemma-4-26b-a4b",
            temperature=0.1,
            max_retries=2,
            json_schema_required=True,
        )


def load_llm_config(path: str | Path) -> LLMConfig:
    config_path = Path(path)
    env_path = find_dotenv(config_path)
    if env_path is not None:
        load_dotenv(env_path)

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    _reject_ollama_only_keys(raw_config)

    default = LLMConfig.default()
    provider = str(_config_value(raw_config, "llm_provider", "LLM_PROVIDER", default.provider))
    return LLMConfig(
        provider=provider,
        base_url=_normalize_base_url(
            str(_config_value(raw_config, "llm_base_url", "LLM_BASE_URL", default.base_url))
        ),
        model=str(_config_value(raw_config, "llm_model", "LLM_MODEL", default.model)),
        temperature=float(raw_config.get("temperature", default.temperature)),
        max_retries=int(raw_config.get("max_retries", default.max_retries)),
        json_schema_required=bool(
            raw_config.get("json_schema_required", default.json_schema_required)
        ),
        api_key=_api_key_value(raw_config, provider),
        timeout_seconds=float(raw_config.get("timeout_seconds", default.timeout_seconds)),
    )


def _config_value(
    raw_config: dict[str, Any],
    config_key: str,
    env_key: str,
    default: str,
) -> str:
    if config_key in raw_config:
        return str(raw_config[config_key])
    env_value = os.environ.get(env_key)
    if env_value:
        return env_value
    return default


def _reject_ollama_only_keys(raw_config: dict[str, Any]) -> None:
    found = sorted(OLLAMA_ONLY_KEYS.intersection(raw_config.keys()))
    if found:
        keys = ", ".join(found)
        raise ValueError(f"Ollama-only keys are not supported: {keys}")


def _api_key_value(raw_config: dict[str, Any], provider: str) -> str | None:
    if "llm_api_key" in raw_config:
        return str(raw_config["llm_api_key"])
    if provider.lower() != "openai":
        return None
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")
