import json
from dataclasses import replace
from urllib import request

from src.news_impact.llm_client import UrllibJsonTransport
from src.news_impact.llm_config import LLMConfig, load_llm_config
from src.news_impact.llm_smoke import check_llama_cpp_prerequisites


def test_news_impact_llm_default_uses_openai_without_hardcoded_api_key():
    config = LLMConfig.default()

    assert config.provider == "openai"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.model == "gpt-5-mini"
    assert config.api_key is None


def test_news_impact_llm_config_reads_openai_api_key_from_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "news_impact.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-colab")

    config = load_llm_config(config_path)

    assert config.provider == "openai"
    assert config.api_key == "sk-colab"


def test_news_impact_llm_config_keeps_gemma_as_optional_provider_config(tmp_path):
    config_path = tmp_path / "news_impact.gemma.json"
    config_path.write_text(
        json.dumps(
            {
                "llm_provider": "llama_cpp",
                "llm_base_url": "http://localhost:8001/v1",
                "llm_model": "gemma-4-26b-a4b",
            }
        ),
        encoding="utf-8",
    )

    config = load_llm_config(config_path)

    assert config.provider == "llama_cpp"
    assert config.base_url == "http://localhost:8001/v1"
    assert config.model == "gemma-4-26b-a4b"
    assert config.api_key is None


def test_news_impact_openai_api_key_env_has_priority_over_llm_api_key(tmp_path, monkeypatch):
    config_path = tmp_path / "news_impact.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "sk-generic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    config = load_llm_config(config_path)

    assert config.api_key == "sk-openai"


def test_urllib_transport_sends_openai_authorization_header(monkeypatch):
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"ok": true}'

    def _fake_urlopen(http_request, timeout):
        captured["authorization"] = http_request.get_header("Authorization")
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(request, "urlopen", _fake_urlopen)
    transport = UrllibJsonTransport(default_headers={"Authorization": "Bearer sk-colab"})

    response = transport.post_json(
        "https://api.openai.com/v1/chat/completions",
        {"model": "gpt-5-mini"},
        timeout_seconds=12.0,
    )

    assert response == {"ok": True}
    assert captured["authorization"] == "Bearer sk-colab"
    assert captured["timeout"] == 12.0


def test_openai_preflight_uses_environment_key_not_local_llama_runtime():
    config = replace(LLMConfig.default(), api_key="sk-colab")

    result = check_llama_cpp_prerequisites(
        config,
        command_lookup=lambda command: None,
        port_probe=lambda host, port, timeout: False,
    )

    assert result["status"] == "ready"
    assert result["provider"] == "openai"
    assert result["missing"] == []


def test_openai_preflight_reports_missing_api_key_without_llama_requirements():
    result = check_llama_cpp_prerequisites(
        LLMConfig.default(),
        command_lookup=lambda command: None,
        port_probe=lambda host, port, timeout: False,
    )

    assert result["status"] == "blocked"
    assert result["missing"] == ["OPENAI_API_KEY"]
