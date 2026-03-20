from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.chatbot.kakao_colab_bot import KakaoColabPredictionBot, PipelineRuntimeConfig


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


def make_bot(tmp_path: Path, runner=None) -> KakaoColabPredictionBot:
    project_root = tmp_path
    runtime_config = PipelineRuntimeConfig(
        project_root=project_root,
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
        process_runner=runner,
    )


def test_returns_cached_prediction_message(tmp_path: Path):
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
    response = bot.handle_utterance("005930")
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "삼성전자" in text
    assert "권고: 매수" in text
    assert "신뢰도: 0.880 (높음)" in text


def test_starts_new_prediction_job_for_missing_symbol(tmp_path: Path):
    runner = RecordingRunner()
    bot = make_bot(tmp_path, runner=runner)

    response = bot.handle_utterance("000660")
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "000660 예측을 시작합니다" in text
    assert len(runner.calls) == 1
    command = runner.calls[0]["command"]
    assert "--add-symbols" in command
    assert "000660.KS" in command
    assert "--fetch-investor-context" in command
