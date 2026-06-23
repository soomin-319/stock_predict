from __future__ import annotations

import src.chatbot.kakao_colab_bot as bot
from src.chatbot.runtime_config import PipelineRuntimeConfig, PyngrokTunnelConfig


def test_runtime_config_classes_are_reexported_from_bot_module():
    # The bot module and src.chatbot package expose these as public names, so the
    # extracted module must remain the same object reachable via the old paths.
    assert bot.PipelineRuntimeConfig is PipelineRuntimeConfig
    assert bot.PyngrokTunnelConfig is PyngrokTunnelConfig


def test_build_command_disables_external_and_sets_report_json():
    cfg = PipelineRuntimeConfig(
        use_external=False,
        report_json="r.json",
        fetch_investor_context=False,
    )

    cmd = cfg.build_command("005930")

    assert "--disable-external" in cmd
    assert cmd[cmd.index("--report-json") + 1] == "r.json"
    assert "--add-symbols" in cmd and "005930" in cmd
    assert "--fetch-investor-context" not in cmd


def test_build_subprocess_env_moves_secrets_into_env():
    cfg = PipelineRuntimeConfig(openai_api_key="sk-test")

    env = cfg.build_subprocess_env(base_env={})

    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_pyngrok_tunnel_config_defaults():
    tunnel = PyngrokTunnelConfig()

    assert tunnel.port == 8000
    assert tunnel.bind_tls is True
    assert tunnel.auth_token is None
