from __future__ import annotations

import shutil
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from news_impact.llm_client import JsonTransport, LlamaCppClient
from news_impact.llm_config import LLMConfig


def run_llm_smoke(
    config: LLMConfig,
    transport: JsonTransport | None = None,
) -> dict[str, Any]:
    client = LlamaCppClient(config=config, transport=transport)
    client.verify_model_alias()
    return {
        "status": "ok",
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
    }


def check_llama_cpp_prerequisites(
    config: LLMConfig,
    model_path: str | Path | None = None,
    grammar_path: str | Path | None = None,
    command_lookup: Any | None = None,
    port_probe: Any | None = None,
) -> dict[str, Any]:
    """Check local runtime prerequisites before running the live LLM smoke test."""
    parsed = urlparse(config.base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if config.provider.lower() == "openai":
        missing = [] if config.api_key else ["OPENAI_API_KEY"]
        return {
            "status": "ready" if not missing else "blocked",
            "provider": config.provider,
            "base_url": config.base_url,
            "model": config.model,
            "host": host,
            "port": port,
            "runtime_paths": {},
            "listener_ready": None,
            "model_path": None,
            "grammar_path": None,
            "missing": missing,
        }

    lookup = command_lookup or shutil.which
    probe = port_probe or _can_connect

    runtime_paths = {
        command: lookup(command)
        for command in ("llama-server", "llama-cli")
    }
    listener_ready = probe(host, port, min(config.timeout_seconds, 2.0))
    missing: list[str] = []
    if not any(runtime_paths.values()) and not listener_ready:
        missing.append("llama-server or llama-cli")
    if model_path is not None and not Path(model_path).exists():
        missing.append(f"model file: {model_path}")
    if grammar_path is not None and not Path(grammar_path).exists():
        missing.append(f"grammar file: {grammar_path}")
    if not listener_ready:
        missing.append(f"{host}:{port} listener")

    return {
        "status": "ready" if not missing else "blocked",
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "host": host,
        "port": port,
        "runtime_paths": runtime_paths,
        "listener_ready": listener_ready,
        "model_path": str(model_path) if model_path is not None else None,
        "grammar_path": str(grammar_path) if grammar_path is not None else None,
        "missing": missing,
    }


def _can_connect(host: str, port: int, timeout_seconds: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False
