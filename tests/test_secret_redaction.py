from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from src.chatbot.kakao_colab_bot import KakaoColabPredictionBot, PipelineRuntimeConfig
from src.utils.secrets import redact_argv, redact_text, redact_value


def test_redact_argv_masks_flag_values_and_registered_secrets():
    argv = ["python", "x.py", "--openai-api-key", "sk-live", "--name=ok", "--token=abc"]

    assert redact_argv(argv, secret_values=["sk-live"]) == [
        "python",
        "x.py",
        "--openai-api-key",
        "[REDACTED]",
        "--name=ok",
        "--token=[REDACTED]",
    ]


def test_redact_value_recursively_masks_runtime_state():
    payload = {"command": ["--naver-client-secret", "secret"], "error": "failed secret"}

    redacted = redact_value(payload, secret_values=["secret"])

    assert redacted["command"] == ["--naver-client-secret", "[REDACTED]"]
    assert redacted["error"] == "failed [REDACTED]"


def test_job_registry_and_streamed_log_redact_configured_secrets(tmp_path: Path):
    cfg = PipelineRuntimeConfig(project_root=tmp_path, openai_api_key="sk-live")
    bot = KakaoColabPredictionBot(runtime_config=cfg)
    state_path = tmp_path / "state.json"

    bot._save_registry(
        state_path,
        {"job": {"command": ["--openai-api-key", "sk-live"], "error": "failed sk-live"}},
    )
    log = io.StringIO()
    bot._stream_process_output("005930.KS", SimpleNamespace(stdout=["key=sk-live\n"]), log)

    saved = state_path.read_text(encoding="utf-8")
    assert "sk-live" not in saved
    assert json.loads(saved)["job"]["command"][-1] == "[REDACTED]"
    assert log.getvalue() == "key=[REDACTED]\n"


def test_redact_text_masks_registered_values():
    assert redact_text("failed secret-value", ["secret-value"]) == "failed [REDACTED]"
