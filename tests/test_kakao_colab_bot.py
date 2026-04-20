from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
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
        naver_client_id="naver-id",
        naver_client_secret="naver-secret",
        bootstrap_default_symbols=False,
        async_issue_summary_on_demand=False,
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
    assert "사유: 테스트 사유" in text
    assert response["template"]["quickReplies"][0]["label"] == "최신화"


def test_returns_cached_prediction_when_result_stock_code_has_market_suffix(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "종목코드": "034020.KS",
                "종목명": "두산에너빌리티",
                "권고": "관망",
                "내일 예상 종가": 30100,
                "내일 예상 수익률(%)": "0.321%",
                "상승확률(%)": "51.2%",
                "예측 신뢰도": "61.0%",
                "예측 이유": "테스트 사유",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "034020",
                "user": {"id": "user-suffix"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "두산에너빌리티" in text
    assert "권고: 관망" in text


def test_cached_prediction_generates_issue_summary_for_each_requested_symbol_with_prediction_date_filter(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"종목코드": "005930", "종목명": "삼성전자", "권고": "매수", "내일 예상 종가": 100, "내일 예상 수익률(%)": "1.0%", "상승확률(%)": "60.0%", "예측 신뢰도": "70.0%", "예측 이유": "r"},
            {"종목코드": "000660", "종목명": "SK하이닉스", "권고": "관망", "내일 예상 종가": 200, "내일 예상 수익률(%)": "0.5%", "상승확률(%)": "55.0%", "예측 신뢰도": "65.0%", "예측 이유": "r2"},
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame(
        [
            {"Symbol": "005930.KS", "Date": "2026-03-26"},
            {"Symbol": "000660.KS", "Date": "2026-03-26"},
        ]
    ).to_csv(result_dir / "result_detail.csv", index=False)
    pd.DataFrame(
        [
            {"Date": "2026-03-26", "Symbol": "005930.KS", "source_type": "disclosure", "title": "당일 공시"},
            {"Date": "2026-03-25", "Symbol": "005930.KS", "source_type": "disclosure", "title": "전일 공시"},
            {"Date": "2026-03-26", "Symbol": "000660.KS", "source_type": "news", "title": "당일 뉴스"},
        ]
    ).to_csv(result_dir / "result_news.csv", index=False)

    captured: list[dict] = []

    def _fake_append(pred_df, context_raw_df=None, **kwargs):
        captured.append(
            {
                "symbol": pred_df.iloc[0]["Symbol"],
                "dates": sorted(context_raw_df["Date"].astype(str).unique().tolist()) if context_raw_df is not None and not context_raw_df.empty else [],
            }
        )
        out = pred_df.copy()
        out["오늘 종목 이슈 한줄 요약"] = "요약"
        out["공시 요약"] = "[공시 요약]\n- 테스트"
        out["뉴스 요약"] = "[뉴스 요약]\n- 테스트"
        out["종합 판단"] = "중립"
        out["주의사항"] = "참고용"
        out["원문 개수"] = 1
        out["핵심 원문 목록"] = "[]"
        return out

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.append_issue_summary_columns", _fake_append)

    bot = make_bot(tmp_path)
    response_a = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-a"}}})
    response_b = bot.handle_kakao_payload({"userRequest": {"utterance": "000660", "user": {"id": "u-b"}}})

    text_a = response_a["template"]["outputs"][0]["simpleText"]["text"]
    text_b = response_b["template"]["outputs"][0]["simpleText"]["text"]
    assert "[공시 요약]" in text_a
    assert "[뉴스 요약]" in text_b
    assert captured[0]["symbol"] == "005930.KS"
    assert captured[0]["dates"] in ([], [datetime.now(timezone.utc).date().isoformat()])
    assert captured[1]["symbol"] == "000660.KS"


def test_cached_prediction_retries_when_detail_date_is_too_old(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"종목코드": "000660", "종목명": "SK하이닉스", "권고": "관망", "내일 예상 종가": 200, "내일 예상 수익률(%)": "0.5%", "상승확률(%)": "55.0%", "예측 신뢰도": "65.0%", "예측 이유": "r2"}]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "000660.KS", "Date": "2026-03-06"}]).to_csv(result_dir / "result_detail.csv", index=False)
    pd.DataFrame([{"Date": "2026-03-06", "Symbol": "000660.KS", "source_type": "news", "title": "과거 뉴스"}]).to_csv(
        result_dir / "result_news.csv", index=False
    )

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "000660", "user": {"id": "u-stale"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "최신 예측을 다시 시작합니다" in text


def test_collect_live_events_uses_short_ttl_cache(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"종목코드": "005930", "종목명": "삼성전자", "권고": "매수", "내일 예상 종가": 71200, "내일 예상 수익률(%)": "1.2%", "상승확률(%)": "78.0%", "예측 신뢰도": "88.0%", "예측 이유": "테스트"}]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "005930.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)

    calls = {"count": 0}

    def _fake_collect_context_raw_events(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame([{"Date": "2026-03-26", "Symbol": "005930.KS", "source_type": "news", "title": "당일 뉴스"}])

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.collect_context_raw_events", _fake_collect_context_raw_events)
    runtime_config = PipelineRuntimeConfig(
        project_root=tmp_path,
        python_executable="python",
        input_csv="data/real_ohlcv.csv",
        report_json="pipeline_report_with_context.json",
        figure_dir="figures_with_context",
        naver_client_id="real_id",
        naver_client_secret="real_secret",
        bootstrap_default_symbols=False,
        async_issue_summary_on_demand=False,
    )
    bot = KakaoColabPredictionBot(
        runtime_config=runtime_config,
        result_simple_path="result/result_simple.csv",
        state_path="result/chatbot_jobs.json",
        session_path="result/chatbot_sessions.json",
    )
    _ = bot._collect_live_symbol_events("005930.KS", "2026-03-26")
    _ = bot._collect_live_symbol_events("005930.KS", "2026-03-26")

    assert calls["count"] == 1


def test_cached_prediction_uses_today_only_when_today_events_are_missing(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"종목코드": "000660", "종목명": "SK하이닉스", "권고": "관망", "내일 예상 종가": 200, "내일 예상 수익률(%)": "0.5%", "상승확률(%)": "55.0%", "예측 신뢰도": "65.0%", "예측 이유": "r2"}]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "000660.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)
    pd.DataFrame([{"Date": "2026-03-25", "Symbol": "000660.KS", "source_type": "disclosure", "title": "전일 공시"}]).to_csv(
        result_dir / "result_news.csv", index=False
    )

    captured = {}

    def _fake_append(pred_df, context_raw_df=None, **kwargs):
        captured["dates"] = (
            sorted(context_raw_df["Date"].astype(str).unique().tolist())
            if context_raw_df is not None and "Date" in context_raw_df.columns
            else []
        )
        out = pred_df.copy()
        out["오늘 종목 이슈 한줄 요약"] = "요약"
        out["공시 요약"] = "[공시 요약]\n- 전일 공시"
        out["뉴스 요약"] = "[뉴스 요약]\n- 없음"
        out["종합 판단"] = "중립"
        out["주의사항"] = "참고용"
        out["원문 개수"] = 1
        out["핵심 원문 목록"] = "[]"
        return out

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.append_issue_summary_columns", _fake_append)
    monkeypatch.setattr(KakaoColabPredictionBot, "_collect_live_symbol_events", lambda self, symbol, reference_date: pd.DataFrame())
    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "000660", "user": {"id": "u-yday"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert captured["dates"] == []
    assert "[공시 요약]" in text


def test_cached_prediction_attempts_live_fetch_only_once_for_today(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"종목코드": "000660", "종목명": "SK하이닉스", "권고": "관망", "내일 예상 종가": 200, "내일 예상 수익률(%)": "0.5%", "상승확률(%)": "55.0%", "예측 신뢰도": "65.0%", "예측 이유": "r2"}]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "000660.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)
    pd.DataFrame([{"Date": "2026-03-24", "Symbol": "005930.KS", "source_type": "news", "title": "other"}]).to_csv(
        result_dir / "result_news.csv", index=False
    )

    calls = {"count": 0}

    def _fake_collect(self, symbol, reference_date):
        calls["count"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(KakaoColabPredictionBot, "_collect_live_symbol_events", _fake_collect)
    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "000660", "user": {"id": "u-once"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert calls["count"] == 1
    assert "권고:" in text


def test_cached_prediction_can_generate_summary_when_result_news_is_missing(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"종목코드": "005930", "종목명": "삼성전자", "권고": "매수", "내일 예상 종가": 71000, "내일 예상 수익률(%)": "1.2%", "상승확률(%)": "70.0%", "예측 신뢰도": "80.0%", "예측 이유": "r"}]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "005930.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)

    monkeypatch.setattr(
        KakaoColabPredictionBot,
        "_collect_live_symbol_events",
        lambda self, symbol, reference_date: pd.DataFrame(
            [{"Date": reference_date, "Symbol": symbol, "source_type": "news", "title": "당일 이슈"}]
        ),
    )

    def _fake_append(pred_df, context_raw_df=None, **kwargs):
        out = pred_df.copy()
        out["오늘 종목 이슈 한줄 요약"] = "요약"
        out["공시 요약"] = "[공시 요약]\n- 없음"
        out["뉴스 요약"] = "[뉴스 요약]\n- 당일 이슈"
        out["종합 판단"] = "중립"
        out["주의사항"] = "참고용"
        out["원문 개수"] = 1
        out["핵심 원문 목록"] = "[]"
        return out

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.append_issue_summary_columns", _fake_append)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-missing-news"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]
    assert "[뉴스 요약]" in text


def test_cached_prediction_still_returns_message_when_issue_summary_raises(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"종목코드": "005930", "종목명": "삼성전자", "권고": "매수", "내일 예상 종가": 71200, "내일 예상 수익률(%)": "1.234%", "상승확률(%)": "78.9%", "예측 신뢰도": "88.0%", "예측 이유": "테스트 사유"}]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    monkeypatch.setattr(bot, "_attach_live_issue_summary", lambda row, symbol: (_ for _ in ()).throw(RuntimeError("boom")))
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-safe"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "권고: 매수" in text


def test_cached_prediction_still_returns_message_when_issue_summary_times_out(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"종목코드": "005930", "종목명": "삼성전자", "권고": "매수", "내일 예상 종가": 71200, "내일 예상 수익률(%)": "1.234%", "상승확률(%)": "78.9%", "예측 신뢰도": "88.0%", "예측 이유": "테스트 사유"}]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "005930.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)

    monkeypatch.setattr(
        "src.chatbot.kakao_colab_bot.append_issue_summary_columns",
        lambda *args, **kwargs: (_ for _ in ()).throw(__import__("concurrent").futures.TimeoutError()),
    )

    bot = make_bot(tmp_path)
    bot.runtime_config.async_issue_summary_on_demand = True
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-timeout"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "공시/뉴스 요약 작업을 진행 중" in text


def test_build_response_truncates_overlong_simpletext_payload(tmp_path: Path):
    bot = make_bot(tmp_path)
    long_text = "A" * 1200
    response = bot._build_response(long_text, quick_replies=[("a", "a")] * 12)
    text = response["template"]["outputs"][0]["simpleText"]["text"]
    quick = response["template"]["quickReplies"]

    assert len(text) <= 900
    assert text.endswith("...(생략)")
    assert len(quick) == 10


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
    assert "--issue-summary-symbols" in command
    assert "000660.KS" in command
    assert "--fetch-investor-context" in command
    assert "--dart-corp-map-csv" in command
    assert "data/dart_corp_map.csv" in command
    assert "--disable-external" in command

    # Secrets must NOT be passed via CLI (visible in `ps`); they go via env.
    assert "--dart-api-key" not in command
    assert "--naver-client-id" not in command
    assert "--naver-client-secret" not in command
    assert "--openai-api-key" not in command
    assert "demo-key" not in command

    env = runner.calls[0].get("env") or {}
    assert env.get("DART_API_KEY") == "demo-key"
    assert env.get("NAVER_CLIENT_ID") == "naver-id"
    assert env.get("NAVER_CLIENT_SECRET") == "naver-secret"

    session_path = tmp_path / "result" / "chatbot_sessions.json"
    assert session_path.exists()
    assert "user-77" in session_path.read_text(encoding="utf-8")


def test_start_bootstrap_job_uses_prewarm_worker_without_large_add_symbols_cli(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"Ticker": "005930", "Symbol": "005930.KS", "Name": "삼성전자", "Market": "KOSPI"},
            {"Ticker": "000660", "Symbol": "000660.KS", "Name": "SK하이닉스", "Market": "KOSPI"},
            {"Ticker": "035420", "Symbol": "035420.KS", "Name": "NAVER", "Market": "KOSPI"},
        ]
    ).to_csv(data_dir / "krx_symbol_name_map.csv", index=False)

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.prewarm_prediction_cache", lambda *args, **kwargs: {"ok": "1"})
    runtime_config = PipelineRuntimeConfig(
        project_root=tmp_path,
        bootstrap_default_symbols=True,
    )
    bot = KakaoColabPredictionBot(
        runtime_config=runtime_config,
        result_simple_path="result/result_simple.csv",
        state_path="result/chatbot_jobs.json",
        session_path="result/chatbot_sessions.json",
    )

    assert bot._start_bootstrap_job(force=True) is True
    assert bot._bootstrap_thread is not None
    bot._bootstrap_thread.join(timeout=1.0)

    bootstrap_state = bot._job_registry[bot.BOOTSTRAP_JOB_KEY]
    assert bootstrap_state["status"] == "completed"
    assert bootstrap_state["command"][0] == "internal:prewarm_prediction_cache"
    assert "--issue-summary-enabled" in bootstrap_state["command"]


def test_request_during_bootstrap_returns_global_progress_and_queues_summary(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"Ticker": "005930", "Symbol": "005930.KS", "Name": "삼성전자", "Market": "KOSPI"},
            {"Ticker": "000660", "Symbol": "000660.KS", "Name": "SK하이닉스", "Market": "KOSPI"},
        ]
    ).to_csv(data_dir / "krx_symbol_name_map.csv", index=False)

    release = threading.Event()

    def _slow_prewarm(*args, **kwargs):
        release.wait()
        return {"ok": "1"}

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.prewarm_prediction_cache", _slow_prewarm)
    runtime_config = PipelineRuntimeConfig(
        project_root=tmp_path,
        python_executable="python",
        input_csv="data/real_ohlcv.csv",
        report_json="pipeline_report_with_context.json",
        figure_dir="figures_with_context",
        bootstrap_default_symbols=True,
    )
    bot = KakaoColabPredictionBot(
        runtime_config=runtime_config,
        result_simple_path="result/result_simple.csv",
        state_path="result/chatbot_jobs.json",
        session_path="result/chatbot_sessions.json",
    )
    bot._start_bootstrap_job(force=True)

    response = bot.handle_kakao_payload({"userRequest": {"utterance": "000660", "user": {"id": "u-bootstrap-progress"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "초기 전체 종목 예측" in text
    assert "요약 요청을 접수" in text
    assert "000660.KS" in bot._queued_summary_symbols
    release.set()


def test_background_summary_uses_non_timeout_live_fetch(tmp_path: Path, monkeypatch):
    bot = make_bot(tmp_path)
    row = pd.Series({"종목코드": "005930", "종목명": "삼성전자"})
    captured = {"flag": None}

    def _fake_attach(target_row, symbol, use_timeout_for_live_fetch=True, **kwargs):
        captured["flag"] = use_timeout_for_live_fetch
        out = target_row.copy()
        out["공시 요약"] = "실제 공시"
        out["뉴스 요약"] = "실제 뉴스"
        return out

    monkeypatch.setattr(bot, "_attach_live_issue_summary", _fake_attach)
    bot._run_issue_summary_background("005930.KS", row)

    assert captured["flag"] is False


def test_background_summary_console_log_does_not_duplicate_section_headers(tmp_path: Path, monkeypatch):
    bot = make_bot(tmp_path)
    row = pd.Series({"종목코드": "005930", "종목명": "삼성전자"})
    logs: list[str] = []

    def _fake_attach(target_row, symbol, **kwargs):
        out = target_row.copy()
        out["공시 요약"] = "[공시 요약]\n- 확인된 핵심 공시 내용 없음"
        out["뉴스 요약"] = "[뉴스 요약]\n- 테스트 뉴스"
        return out

    monkeypatch.setattr(bot, "_attach_live_issue_summary", _fake_attach)
    monkeypatch.setattr(bot, "_console_log", lambda message: logs.append(message))
    bot._run_issue_summary_background("005930.KS", row)

    joined = "\n".join(logs)
    assert joined.count("[공시 요약]") == 1
    assert joined.count("[뉴스 요약]") == 1


def test_additional_symbol_request_generates_summary_when_placeholder_exists(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "종목코드": "000660",
                "종목명": "SK하이닉스",
                "권고": "매도",
                "내일 예상 종가": 842353,
                "내일 예상 수익률(%)": "-8.836%",
                "상승확률(%)": "13.6%",
                "예측 신뢰도": "25.9%",
                "예측 이유": "거래대금 상위",
                "공시 요약": "[공시 요약]\n- 요청 종목에 대해서만 요약을 생성합니다.",
                "뉴스 요약": "[뉴스 요약]\n- 요청 종목에 대해서만 요약을 생성합니다.",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "000660.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)

    calls = {"count": 0}

    def _fake_append(pred_df, **kwargs):
        calls["count"] += 1
        out = pred_df.copy()
        out["공시 요약"] = "[공시 요약]\n- 실제 공시 요약"
        out["뉴스 요약"] = "[뉴스 요약]\n- 실제 뉴스 요약"
        out["오늘 종목 이슈 한줄 요약"] = "요약"
        out["종합 판단"] = "중립"
        out["주의사항"] = "참고"
        out["원문 개수"] = 1
        out["핵심 원문 목록"] = "[]"
        return out

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.append_issue_summary_columns", _fake_append)
    monkeypatch.setattr(KakaoColabPredictionBot, "_collect_live_symbol_events", lambda self, symbol, reference_date: pd.DataFrame())

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "000660", "user": {"id": "u-extra-symbol"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]
    for _ in range(20):
        follow_up = bot.handle_kakao_payload({"userRequest": {"utterance": "000660", "user": {"id": "u-extra-symbol"}}})
        follow_text = follow_up["template"]["outputs"][0]["simpleText"]["text"]
        if "실제 공시 요약" in follow_text:
            break
        time.sleep(0.01)

    assert "실제 공시 요약" in follow_text
    assert "실제 뉴스 요약" in follow_text
    assert calls["count"] == 1


def test_cached_prediction_with_empty_issue_placeholders_triggers_async_summary_progress(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "삼성전자",
                "권고": "매수",
                "내일 예상 종가": 71200,
                "내일 예상 수익률(%)": "1.2%",
                "상승확률(%)": "70.0%",
                "예측 신뢰도": "80.0%",
                "예측 이유": "r",
                "공시 요약": "당일 공시 없음.",
                "뉴스 요약": "당일 뉴스 없음.",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    bot.runtime_config.async_issue_summary_on_demand = True
    calls = {"count": 0}

    def _fake_start(symbol, row):
        calls["count"] += 1

    monkeypatch.setattr(bot, "_start_issue_summary_background", _fake_start)
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-empty-issue"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "공시/뉴스 요약 작업을 진행 중" in text
    assert calls["count"] == 1


def test_repeated_request_while_running_shows_running_message(tmp_path: Path):
    runner = RecordingRunner()
    bot = make_bot(tmp_path, runner=runner)

    first = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-seq"}}})
    second = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-seq"}}})

    first_text = first["template"]["outputs"][0]["simpleText"]["text"]
    second_text = second["template"]["outputs"][0]["simpleText"]["text"]

    assert "005930 예측을 시작합니다" in first_text
    assert "005930 예측이 현재 진행 중입니다" in second_text


def test_stale_running_state_is_downgraded_to_failed_and_prompts_retry(tmp_path: Path):
    bot = make_bot(tmp_path)
    bot._job_registry["005930.KS"] = {"status": "running", "submitted_at": "2026-03-26T00:00:00+00:00"}
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-stale-running"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "예측 작업이 실패했습니다" in text
    assert bot._job_registry["005930.KS"]["status"] == "failed"


def test_completed_without_result_for_long_time_prompts_refresh(tmp_path: Path):
    bot = make_bot(tmp_path)
    bot._job_registry["005930.KS"] = {"status": "completed", "completed_at": "2026-03-26T00:00:00+00:00"}
    bot._session_registry["u-no-result"] = {"last_symbol": "005930.KS", "last_intent": "tracking"}
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "결과", "user": {"id": "u-no-result"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "결과 파일에 반영되지 않았습니다" in text
    assert "뉴스/공시 요약을 다시 실행" in text


def test_refresh_restarts_prediction_even_when_previous_job_is_completed_without_cached_row(tmp_path: Path):
    runner = RecordingRunner()
    bot = make_bot(tmp_path, runner=runner)
    bot._job_registry["005930.KS"] = {"status": "completed", "completed_at": "2026-03-26T00:00:00+00:00"}
    bot._session_registry["u-refresh"] = {"last_symbol": "005930.KS", "last_intent": "tracking"}

    response = bot.handle_kakao_payload({"userRequest": {"utterance": "최신화", "user": {"id": "u-refresh"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "최신 예측을 다시 시작합니다" in text
    assert len(runner.calls) == 1


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

    assert "사유: 거래대금 상위, 외국인/기관 순매수, 나스닥 선물 +1% 이상" in text


def test_cached_prediction_message_hides_reason_when_rule_not_satisfied(tmp_path: Path):
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
                "예측 이유": "",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-no-reason"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "사유:" not in text


def test_cached_prediction_message_renders_issue_summary_block_header(tmp_path: Path):
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
                "예측 이유": "종배수급: 거래대금 상위",
                "오늘 종목 이슈 한줄 요약": "공급계약 공시가 발표되었습니다.",
                "공시 요약": "대형 공급계약 체결",
                "뉴스 요약": "증권사 리포트 호평",
                "종합 판단": "약한 호재",
                "주의사항": "이 문구는 출력 대상이 아님",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-issue-block"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "[공시 요약]" in text
    assert "- 대형 공급계약 체결" in text
    assert "[뉴스 요약]" in text
    assert "- 증권사 리포트 호평" in text
    assert "종합 판단:" not in text
    assert "주의사항:" not in text


def test_cached_prediction_message_splits_issue_summary_into_bullets(tmp_path: Path):
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
                "예측 이유": "종배수급: 거래대금 상위",
                "공시 요약": "[공시 요약]\n- 미국 ADR 상장 추진 관련 보도 / 대규모 현금·자금 조달 및 주주환원 논의; 메모리 가격 강세",
                "뉴스 요약": "복수 기사에서 ADR 상장 추진 공식화 · 업황 측면 메모리 가격 상승 | 중동 리스크로 변동성 우려",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-issue-split"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "[공시 요약]" in text
    assert "- 미국 ADR 상장 추진 관련 보도" in text
    assert "- 대규모 현금·자금 조달 및 주주환원 논의" in text
    assert "- 메모리 가격 강세" in text
    assert "[뉴스 요약]" in text
    assert "- 복수 기사에서 ADR 상장 추진 공식화" in text
    assert "- 업황 측면 메모리 가격 상승" in text
    assert "- 중동 리스크로 변동성 우려" in text


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
        def set_auth_token(self, token, **kwargs):
            calls["auth_token"] = token

        def connect(self, **kwargs):
            calls["kwargs"] = kwargs
            return FakeListener()

    fake_module = ModuleType("pyngrok")
    fake_module.ngrok = FakeNgrok()

    class _FakeConf:
        @staticmethod
        def get_default():
            return object()

        class PyngrokConfig:
            def __init__(self, ngrok_path=None):
                self.ngrok_path = ngrok_path

    fake_module.conf = _FakeConf
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


def test_start_pyngrok_tunnel_disconnects_existing_endpoint_on_err_ngrok_334(monkeypatch):
    calls = {"connect_count": 0, "disconnected": None}

    class FakeListener:
        public_url = "https://new.ngrok-free.app/"

    class FakeNgrok:
        def connect(self, **kwargs):
            calls["connect_count"] += 1
            if calls["connect_count"] == 1:
                raise RuntimeError(
                    "failed to start tunnel: The endpoint 'https://occupied.ngrok-free.dev' is already online. ERR_NGROK_334"
                )
            return FakeListener()

        def disconnect(self, url, **kwargs):
            calls["disconnected"] = url

    fake_module = ModuleType("pyngrok")
    fake_module.ngrok = FakeNgrok()

    class _FakeConf:
        @staticmethod
        def get_default():
            return object()

        class PyngrokConfig:
            def __init__(self, ngrok_path=None):
                self.ngrok_path = ngrok_path

    fake_module.conf = _FakeConf
    monkeypatch.setitem(sys.modules, "pyngrok", fake_module)

    public_url = start_pyngrok_tunnel(PyngrokTunnelConfig(port=8000))

    assert public_url == "https://new.ngrok-free.app"
    assert calls["connect_count"] == 2
    assert calls["disconnected"] == "https://occupied.ngrok-free.dev"


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


def test_launch_colab_kakao_bot_starts_bootstrap_after_server_start(monkeypatch, tmp_path: Path):
    events = []

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True
            events.append("thread_started")

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.create_app", lambda runtime_config=None, bot=None: object())
    monkeypatch.setattr("src.chatbot.kakao_colab_bot.start_pyngrok_tunnel", lambda tunnel_config=None: "https://demo.ngrok")
    monkeypatch.setattr("src.chatbot.kakao_colab_bot.threading.Thread", FakeThread)
    monkeypatch.setattr(
        KakaoColabPredictionBot,
        "_start_bootstrap_job",
        lambda self, force=False: events.append("bootstrap_started") or True,
    )

    launched = launch_colab_kakao_bot(
        runtime_config=PipelineRuntimeConfig(project_root=tmp_path, prewarm_default_predictions=True),
        tunnel_config=PyngrokTunnelConfig(port=8000),
    )

    assert events == ["thread_started", "bootstrap_started"]
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


def test_prewarm_prediction_cache_invalidates_stale_daily_signature(monkeypatch, tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "result_simple.csv").write_text("종목코드,종목명\n005930,삼성전자\n", encoding="utf-8-sig")

    runtime_config = PipelineRuntimeConfig(project_root=tmp_path, input_csv="data/real_ohlcv.csv")
    input_path = tmp_path / "data" / "real_ohlcv.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("Date,Symbol,Open,High,Low,Close,Volume\n", encoding="utf-8")
    universe_path = tmp_path / "data" / "default_universe_kospi50_kosdaq50.csv"
    universe_path.write_text("Symbol\n005930.KS\n", encoding="utf-8")

    stale_signature = _runtime_cache_signature(runtime_config, tmp_path)
    stale_signature["cache_date_kst"] = "2026-03-06"
    meta_path = result_dir / "prewarm_cache_meta.json"
    meta_path.write_text(
        json.dumps({"signature": stale_signature, "signature_hash": _cache_signature_hash(stale_signature)}, ensure_ascii=False),
        encoding="utf-8",
    )

    called = {"count": 0}

    def _fake_run_colab_pipeline(**kwargs):
        called["count"] += 1
        return {"result_simple_csv": str(result_dir / "result_simple.csv")}

    monkeypatch.setattr("colab.stock_predict_colab.run_colab_pipeline", _fake_run_colab_pipeline)

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


def test_finalize_process_logs_completion_without_inline_formatting(tmp_path: Path, monkeypatch):
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

    logs: list[str] = []
    monkeypatch.setattr(bot, "_console_log", lambda message: logs.append(message))

    bot._finalize_process("005930.KS", 0)

    assert any("예측 작업 completed" in log for log in logs)


def test_handle_symbol_request_falls_back_when_cached_message_format_fails(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(
        bot,
        "_format_prediction_message",
        lambda row: (_ for _ in ()).throw(NameError("rationale_block")),
    )

    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u1"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "[005930 삼성전자]" in text
    assert "사유: 거래대금 상위" in text


def test_kakao_webhook_returns_safe_response_when_handler_raises(tmp_path: Path, monkeypatch):
    from src.chatbot.kakao_colab_bot import create_app

    bot = make_bot(tmp_path)
    app = create_app(bot=bot)

    monkeypatch.setattr(
        bot,
        "handle_kakao_payload",
        lambda payload: (_ for _ in ()).throw(NameError("rationale_block")),
    )

    client = app.test_client()
    response = client.post("/kakao/webhook", json={"userRequest": {"utterance": "005930"}})

    assert response.status_code == 200
    body = response.get_json()
    assert "예측 메시지 처리 중 오류가 발생했습니다" in body["template"]["outputs"][0]["simpleText"]["text"]


def test_start_job_response_returns_error_when_process_runner_raises(tmp_path: Path):
    class RaisingRunner:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("runner failed")

    bot = make_bot(tmp_path, runner=RaisingRunner())
    response = bot.handle_kakao_payload(
        {
            "userRequest": {
                "utterance": "005930",
                "user": {"id": "user-runner-fail"},
            }
        }
    )
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "예측 작업 시작 중 오류가 발생했습니다" in text


def test_finalize_process_handles_missing_log_handle(tmp_path: Path, monkeypatch):
    bot = make_bot(tmp_path)
    bot._active_processes["005930.KS"] = {"log_thread": None}
    bot._job_registry["005930.KS"] = {"status": "running"}

    logs: list[str] = []
    monkeypatch.setattr(bot, "_console_log", lambda message: logs.append(message))

    bot._finalize_process("005930.KS", 1)

    assert any("예측 작업 failed" in log for log in logs)


def test_load_cached_result_simple_uses_mtime_cache(monkeypatch, tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "result_simple.csv"
    csv_path.write_text("종목코드,종목명\n005930,삼성전자\n", encoding="utf-8-sig")

    bot = make_bot(tmp_path)
    original_read_csv = pd.read_csv
    calls = {"count": 0}

    def _counting_read_csv(*args, **kwargs):
        calls["count"] += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _counting_read_csv)

    out1 = bot._load_cached_result_simple()
    out2 = bot._load_cached_result_simple()

    assert calls["count"] == 1
    assert not out1.empty and not out2.empty


def test_issue_summary_timeout_does_not_block_webhook_response(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
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
                "공시 요약": "[공시 요약]\n- 요청 종목에 대해서만 요약을 생성합니다.",
                "뉴스 요약": "[뉴스 요약]\n- 요청 종목에 대해서만 요약을 생성합니다.",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)
    pd.DataFrame([{"Symbol": "005930.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)

    def _slow_summary(*args, **kwargs):
        time.sleep(0.2)
        out = args[0].copy()
        out["공시 요약"] = "없음"
        out["뉴스 요약"] = "없음"
        return out

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.append_issue_summary_columns", _slow_summary)

    bot = make_bot(tmp_path)
    bot.runtime_config.async_issue_summary_on_demand = True
    bot.ISSUE_SUMMARY_TIMEOUT_SEC = 0.01

    started = time.perf_counter()
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-timeout"}}})
    elapsed = time.perf_counter() - started

    text = response["template"]["outputs"][0]["simpleText"]["text"]
    assert "공시/뉴스 요약 작업을 진행 중" in text
    assert elapsed < 0.12


def test_cached_prediction_does_not_regenerate_issue_summary_when_summary_exists(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
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
                "공시 요약": "이미 생성된 공시 요약",
                "뉴스 요약": "이미 생성된 뉴스 요약",
            }
        ]
    ).to_csv(result_dir / "result_simple.csv", index=False)

    called = {"count": 0}

    def _should_not_run(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("요약 재생성은 호출되면 안 됩니다.")

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.append_issue_summary_columns", _should_not_run)

    bot = make_bot(tmp_path)
    response = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-summary-exists"}}})
    text = response["template"]["outputs"][0]["simpleText"]["text"]

    assert "이미 생성된 공시 요약" in text
    assert "이미 생성된 뉴스 요약" in text
    assert called["count"] == 0


def test_generated_issue_summary_is_reused_without_refresh(tmp_path: Path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
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
    pd.DataFrame([{"Symbol": "005930.KS", "Date": "2026-03-26"}]).to_csv(result_dir / "result_detail.csv", index=False)

    called = {"count": 0}

    def _fake_summary(pred_df, **kwargs):
        called["count"] += 1
        out = pred_df.copy()
        out["공시 요약"] = "생성된 공시 요약"
        out["뉴스 요약"] = "생성된 뉴스 요약"
        out["오늘 종목 이슈 한줄 요약"] = "요약"
        out["종합 판단"] = "중립"
        out["주의사항"] = "참고"
        out["원문 개수"] = 0
        out["핵심 원문 목록"] = "[]"
        return out

    monkeypatch.setattr("src.chatbot.kakao_colab_bot.append_issue_summary_columns", _fake_summary)
    monkeypatch.setattr(KakaoColabPredictionBot, "_collect_live_symbol_events", lambda self, symbol, reference_date: pd.DataFrame())

    bot = make_bot(tmp_path)
    resp1 = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-reuse-1"}}})
    resp2 = bot.handle_kakao_payload({"userRequest": {"utterance": "005930", "user": {"id": "u-reuse-2"}}})

    text1 = resp1["template"]["outputs"][0]["simpleText"]["text"]
    text2 = resp2["template"]["outputs"][0]["simpleText"]["text"]
    assert "생성된 공시 요약" in text1 and "생성된 뉴스 요약" in text1
    assert "생성된 공시 요약" in text2 and "생성된 뉴스 요약" in text2
    assert called["count"] == 1


def test_load_result_news_merges_result_disclosure(tmp_path: Path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Date": "2026-03-26", "Symbol": "005930.KS", "source_type": "news", "title": "뉴스"}]).to_csv(
        result_dir / "result_news.csv", index=False
    )
    pd.DataFrame([{"Date": "2026-03-26", "Symbol": "005930.KS", "source_type": "disclosure", "title": "공시"}]).to_csv(
        result_dir / "result_disclosure.csv", index=False
    )

    bot = make_bot(tmp_path)
    merged = bot._load_result_news()

    assert len(merged) == 2
    assert set(merged["source_type"].astype(str)) == {"news", "disclosure"}
