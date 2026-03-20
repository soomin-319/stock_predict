from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from src.chatbot.kakao_colab_bot import (
    KakaoColabPredictionBot,
    PipelineRuntimeConfig,
    PyngrokTunnelConfig,
    start_pyngrok_tunnel,
)


class FakeProcess:
    def __init__(self, return_code=None, pid=1234):
        self._return_code = return_code
        self.pid = pid

    def poll(self):
        return self._return_code


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, cwd, stdout, stderr, text):
        self.calls.append({"command": command, "cwd": cwd, "stderr": stderr, "text": text})
        return FakeProcess(return_code=None)


class ImmediateSuccessRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, cwd, stdout, stderr, text):
        self.calls.append({"command": command, "cwd": cwd})
        return FakeProcess(return_code=0)


def make_bot(tmp_path: Path, runner=None) -> KakaoColabPredictionBot:
    runtime_config = PipelineRuntimeConfig(
        project_root=tmp_path,
        python_executable="python",
        input_csv="data/real_ohlcv.csv",
        report_json="pipeline_report_with_context.json",
        figure_dir="figures_with_context",
        dart_api_key="demo-key",
        dart_corp_map_csv="data/dart_corp_map.csv",
    )
    return KakaoColabPredictionBot(
        runtime_config=runtime_config,
        result_simple_path="result/result_simple.csv",
        state_path="result/chatbot_jobs.json",
        session_path="result/chatbot_sessions.json",
        process_runner=runner,
    )


def test_returns_cached_prediction_message_from_kakao_payload(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "삼성전자",
                "권고": "매수",
                "내일 예상 종가": 71200,
                "내일 예상 수익률(%)": 1.234,
                "예측 신뢰도": 0.88,
                "예측 이유": "테스트 사유",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-1"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "삼성전자" in text
    assert "권고: 매수" in text
    assert response["template"]["quickReplies"][0]["label"] == "최신화"


def test_starts_new_prediction_job_and_saves_session(tmp_path: Path):
    runner = RecordingRunner()
    bot = make_bot(tmp_path, runner=runner)

    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "000660",
                "user": {"id": "user-77"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "000660 예측을 시작합니다" in text
    assert len(runner.calls) == 1
    command = runner.calls[0]["command"]
    assert "--add-symbols" in command
    assert "000660.KS" in command
    assert "--fetch-investor-context" in command

    session_path = tmp_path / "result" / "chatbot_sessions.json"
    assert session_path.exists()
    assert "user-77" in session_path.read_text(encoding="utf-8")


def test_status_request_uses_previous_user_symbol(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "삼성전자",
                "권고": "매수",
                "내일 예상 종가": 71200,
                "내일 예상 수익률(%)": 1.234,
                "예측 신뢰도": 0.88,
                "예측 이유": "테스트 사유",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path, runner=ImmediateSuccessRunner())
    bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-55"},
            }
        }
    )

    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "결과",
                "user": {"id": "user-55"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "삼성전자" in text
    assert "내일 예측 수익률" in text


def test_start_pyngrok_tunnel_returns_public_url(monkeypatch):
    calls = {}

    class FakeListener:
        public_url = "https://demo.ngrok-free.app/"

    class FakeNgrok:
        def set_auth_token(self, token):
            calls["auth_token"] = token

        def connect(self, **kwargs):
            calls["kwargs"] = kwargs
            return FakeListener()

    fake_module = ModuleType("pyngrok")
    fake_module.ngrok = FakeNgrok()
    monkeypatch.setitem(sys.modules, "pyngrok", fake_module)

    public_url = start_pyngrok_tunnel(
        PyngrokTunnelConfig(
            port=9000,
            auth_token="token-123",
        )
    )

    assert public_url == "https://demo.ngrok-free.app"
    assert calls["auth_token"] == "token-123"
    assert calls["kwargs"]["addr"] == 9000
    assert calls["kwargs"]["proto"] == "http"
