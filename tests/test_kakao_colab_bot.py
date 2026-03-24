from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from src.chatbot.kakao_colab_bot import (
    KakaoColabPredictionBot,
    PipelineRuntimeConfig,
    PyngrokTunnelConfig,
    launch_colab_kakao_bot,
    _cache_signature_hash,
    _runtime_cache_signature,
    prewarm_prediction_cache,
    start_pyngrok_tunnel,
)


class FakeProcess:
    def __init__(self, return_code=None, pid=1234):
        self._return_code = return_code
        self.pid = pid

    def poll(self):
        return self._return_code


class WaitableFakeProcess(FakeProcess):
    def wait(self):
        return self._return_code if self._return_code is not None else 0


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, cwd, stdout, stderr, text, **kwargs):
        self.calls.append({"command": command, "cwd": cwd, "stderr": stderr, "text": text, **kwargs})
        return FakeProcess(return_code=None)


class ImmediateSuccessRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, cwd, stdout, stderr, text, **kwargs):
        self.calls.append({"command": command, "cwd": cwd, **kwargs})
        return WaitableFakeProcess(return_code=0)


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
                "내일 예상 수익률(%)": "1.234%",
                "상승확률(%)": "78.9%",
                "예측 신뢰도": "88.0%",
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
    assert "[005930 삼성전자]" in text
    assert "상승확률: 78.9%" in text
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
    assert "--disable-news-context" not in command
    assert "--disable-external" in command

    session_path = tmp_path / "result" / "chatbot_sessions.json"
    assert session_path.exists()
    assert "user-77" in session_path.read_text(encoding="utf-8")


def test_start_job_skips_disable_external_flag_when_external_features_enabled(tmp_path: Path):
    runner = RecordingRunner()
    runtime_config = PipelineRuntimeConfig(
        project_root=tmp_path,
        python_executable="python",
        input_csv="data/real_ohlcv.csv",
        report_json="pipeline_report_with_context.json",
        figure_dir="figures_with_context",
        dart_api_key="demo-key",
        dart_corp_map_csv="data/dart_corp_map.csv",
        use_external=True,
    )
    bot = KakaoColabPredictionBot(
        runtime_config=runtime_config,
        result_simple_path="result/result_simple.csv",
        state_path="result/chatbot_jobs.json",
        session_path="result/chatbot_sessions.json",
        process_runner=runner,
    )

    bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "000660",
                "user": {"id": "user-external-enabled"},
            }
        }
    )

    command = runner.calls[0]["command"]
    assert "--disable-external" not in command


def test_start_job_prints_console_progress_hint(tmp_path: Path, capsys):
    runner = RecordingRunner()
    bot = make_bot(tmp_path, runner=runner)

    bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-progress"},
            }
        }
    )

    captured = capsys.readouterr()
    assert "예측 작업 시작" in captured.out
    assert "005930" in captured.out


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
                "내일 예상 수익률(%)": "1.234%",
                "상승확률(%)": "78.9%",
                "예측 신뢰도": "88.0%",
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


def test_cached_prediction_message_formats_price_string_from_result_simple_csv(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "삼성전자",
                "권고": "매수",
                "내일 예상 종가": "71,200원",
                "내일 예상 수익률(%)": "1.234%",
                "상승확률(%)": "78.9%",
                "예측 신뢰도": "88.0%",
                "예측 이유": "테스트 사유",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-price"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "내일 예측 종가: 71,200원" in text


def test_cached_prediction_message_formats_rule_based_reason_labels(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "삼성전자",
                "권고": "매수",
                "내일 예상 종가": 71200,
                "내일 예상 수익률(%)": "1.234%",
                "상승확률(%)": "78.9%",
                "예측 신뢰도": "88.0%",
                "예측 이유": (
                    "종배수급: 거래대금 15위 이내 상위 종목입니다 / "
                    "수급조건: 외국인 1,200억, 기관 1,100억 순매수입니다 / "
                    "해외조건: 나스닥 선물 +1% 이상"
                ),
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-reason-format"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "사유: 거래대금 15위 이내" in text
    assert "외국인/기관 각각 1,000억 이상 순매수" in text
    assert "나스닥선물 +1% / -1%" in text
    assert "해당 기준을 충족합니다" in text


def test_name_query_with_exact_match_starts_prediction(tmp_path: Path, monkeypatch):
    runner = RecordingRunner()
    bot = make_bot(tmp_path, runner=runner)
    monkeypatch.setattr(
        "src.chatbot.kakao_colab_bot.find_symbol_candidates_by_name",
        lambda query, limit=5: [{"ticker": "005930", "name": "삼성전자", "market": "KOSPI", "score": 1.0}],
    )

    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "삼성전자",
                "user": {"id": "user-name-exact"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "005930 예측을 시작합니다" in text
    assert "005930.KS" in runner.calls[0]["command"]


def test_name_query_with_similar_matches_returns_candidates(tmp_path: Path, monkeypatch):
    bot = make_bot(tmp_path)
    monkeypatch.setattr(
        "src.chatbot.kakao_colab_bot.find_symbol_candidates_by_name",
        lambda query, limit=5: [
            {"ticker": "005930", "name": "삼성전자", "market": "KOSPI", "score": 0.91},
            {"ticker": "005935", "name": "삼성전자우", "market": "KOSPI", "score": 0.88},
        ],
    )

    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "삼성전",
                "user": {"id": "user-name-similar"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "비슷한 종목" in text
    assert "삼성전자 (005930, KOSPI)" in text
    assert response["template"]["quickReplies"][0]["messageText"] == "005930"


def test_alpha_query_uses_name_lookup_instead_of_invalid_symbol_prediction(tmp_path: Path, monkeypatch):
    runner = RecordingRunner()
    bot = make_bot(tmp_path, runner=runner)
    monkeypatch.setattr(
        "src.chatbot.kakao_colab_bot.find_symbol_candidates_by_name",
        lambda query, limit=None: [
            {"ticker": "000660", "name": "SK하이닉스", "market": "KOSPI", "score": 0.9},
            {"ticker": "034730", "name": "SK", "market": "KOSPI", "score": 0.88},
        ],
    )

    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "sk",
                "user": {"id": "user-name-alpha"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "비슷한 종목" in text
    assert "SK하이닉스 (000660, KOSPI)" in text
    assert response["template"]["quickReplies"][0]["messageText"] == "000660"
    assert runner.calls == []


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


def test_prewarm_prediction_cache_runs_colab_pipeline(monkeypatch, tmp_path: Path):
    captured = {}

    def _fake_run_colab_pipeline(**kwargs):
        captured.update(kwargs)
        result_dir = tmp_path / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        return {"result_simple_csv": str(result_dir / "result_simple.csv")}

    monkeypatch.setattr("colab.stock_predict_colab.run_colab_pipeline", _fake_run_colab_pipeline)

    runtime_config = PipelineRuntimeConfig(
        project_root=tmp_path,
        input_csv="data/sample_ohlcv.csv",
        report_json="prewarm_report.json",
        figure_dir="prewarm_figures",
        fetch_investor_context=True,
        bootstrap_default_symbols=True,
        real_start="2020-01-01",
    )

    out = prewarm_prediction_cache(runtime_config, force=True)

    assert captured["input_csv"] == "data/sample_ohlcv.csv"
    assert captured["report_json"] == "prewarm_report.json"
    assert captured["figure_dir"] == "prewarm_figures"
    assert captured["use_investor_context"] is True
    assert captured["bootstrap_default_symbols"] is True
    assert captured["real_start"] == "2020-01-01"
    assert out["result_simple_csv"].endswith("result/result_simple.csv")


def test_launch_colab_kakao_bot_prewarms_cache_before_server_start(monkeypatch, tmp_path: Path):
    events = []

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True
            events.append("thread_started")

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.prewarm_prediction_cache", lambda *a, **k: events.append("prewarm"))
    monkeypatch.setattr("src.chatbot.kakao_colab_bot.create_app", lambda runtime_config=None, bot=None: object())
    monkeypatch.setattr("src.chatbot.kakao_colab_bot.start_pyngrok_tunnel", lambda tunnel_config=None: "https://demo.ngrok")
    monkeypatch.setattr("src.chatbot.kakao_colab_bot.threading.Thread", FakeThread)

    launched = launch_colab_kakao_bot(
        runtime_config=PipelineRuntimeConfig(project_root=tmp_path, prewarm_default_predictions=True),
        tunnel_config=PyngrokTunnelConfig(port=8000),
    )

    assert events == ["prewarm", "thread_started"]
    assert launched["webhook_url"] == "https://demo.ngrok/kakao/webhook"


def test_name_query_lists_all_candidates_without_five_item_cap(tmp_path: Path, monkeypatch):
    bot = make_bot(tmp_path)
    monkeypatch.setattr(
        "src.chatbot.kakao_colab_bot.find_symbol_candidates_by_name",
        lambda query, limit=None: [
            {"ticker": f"10000{i}", "name": f"테스트종목{i}", "market": "KOSPI", "score": 0.8 - i * 0.01}
            for i in range(6)
        ],
    )

    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "테스트",
                "user": {"id": "user-name-many"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "6) 테스트종목5 (100005, KOSPI)" in text
    assert response["template"]["quickReplies"][0]["messageText"] == "100000"


def test_prewarm_prediction_cache_reuses_cache_only_when_signature_matches(monkeypatch, tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result_simple.csv").write_text("종목코드,종목명\n005930,삼성전자\n", encoding="utf-8-sig")

    runtime_config = PipelineRuntimeConfig(project_root=tmp_path, input_csv="data/real_ohlcv.csv")
    input_path = tmp_path / "data" / "real_ohlcv.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("Date,Symbol,Open,High,Low,Close,Volume\n", encoding="utf-8")
    universe_path = tmp_path / "data" / "default_universe_kospi50_kosdaq50.csv"
    universe_path.write_text("Symbol\n005930.KS\n", encoding="utf-8")

    signature = _runtime_cache_signature(runtime_config, tmp_path)
    meta_path = result_dir / "prewarm_cache_meta.json"
    meta_path.write_text(
        json.dumps({"signature": signature, "signature_hash": _cache_signature_hash(signature)}, ensure_ascii=False),
        encoding="utf-8",
    )

    called = {"count": 0}

    def _fake_run_colab_pipeline(**kwargs):
        called["count"] += 1
        return {"result_simple_csv": str(result_dir / "result_simple.csv")}

    monkeypatch.setattr("colab.stock_predict_colab.run_colab_pipeline", _fake_run_colab_pipeline)

    out = prewarm_prediction_cache(runtime_config, force=False)

    assert called["count"] == 0
    assert out["result_simple_csv"].endswith("result/result_simple.csv")

    input_path.write_text("Date,Symbol,Open,High,Low,Close,Volume\n2024-01-02,005930.KS,1,1,1,1,1\n", encoding="utf-8")
    out = prewarm_prediction_cache(runtime_config, force=False)

    assert called["count"] == 1
    assert out["result_simple_csv"].endswith("result/result_simple.csv")


def test_load_cached_result_simple_logs_parse_failures(tmp_path: Path, capsys):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result_simple.csv").write_bytes(b"\x80\x81invalid")

    bot = make_bot(tmp_path)
    out = bot._load_cached_result_simple()

    captured = capsys.readouterr()
    assert out.empty
    assert "예측 캐시 CSV 로드 실패" in captured.out


def test_finalize_process_falls_back_when_prediction_message_format_fails(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "삼성전자",
                "권고": "매수",
                "내일 예상 종가": 71200,
                "내일 예상 수익률(%)": "1.234%",
                "상승확률(%)": "78.9%",
                "예측 신뢰도": "88.0%",
                "예측 이유": "종배수급: 거래대금 15위 이내 상위 종목입니다",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    log_path = result_dir / "dummy.log"
    log_handle = log_path.open("w", encoding="utf-8")
    bot._active_processes["005930.KS"] = {"log_handle": log_handle, "log_thread": None}
    bot._job_registry["005930.KS"] = {"status": "running"}

    monkeypatch.setattr(
        bot,
        "_format_prediction_message",
        lambda row: (_ for _ in ()).throw(NameError("rationale_block")),
    )
    logs: list[str] = []
    monkeypatch.setattr(bot, "_console_log", lambda message: logs.append(message))

    bot._finalize_process("005930.KS", 0)

    assert any("메시지 포맷 오류(NameError)" in log for log in logs)
    assert any("사유: 종배수급: 거래대금 15위 이내 상위 종목입니다" in log for log in logs)
