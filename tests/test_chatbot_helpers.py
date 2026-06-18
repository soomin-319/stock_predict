from dataclasses import dataclass

from src.chatbot.intent import is_help_utterance, is_status_utterance, normalize_utterance
from src.chatbot.responses import simple_text_response
from src.chatbot.kakao_colab_bot import (
    KakaoColabPredictionBot,
    PipelineRuntimeConfig,
    create_app,
)


def test_normalize_utterance_strips_whitespace():
    assert normalize_utterance("  도움말  ") == "도움말"


def test_status_and_help_intents():
    assert is_status_utterance("상태")
    assert is_status_utterance("status")
    assert is_help_utterance("도움말")
    assert is_help_utterance("help")


def test_simple_text_response_shape():
    assert simple_text_response("hello") == {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": "hello"}}]},
    }


def test_list_card_response_shape_and_quick_replies():
    from src.chatbot.responses import list_card_response

    response = list_card_response(
        header_title="실시간 추천",
        items=[
            {"title": "1. 삼성전자", "description": "005930.KS | 점수 250"},
            {"title": "2. SK하이닉스", "description": "000660.KS | 점수 240"},
        ],
        quick_replies=[("다시 추천", "추천")],
    )

    assert response["version"] == "2.0"
    card = response["template"]["outputs"][0]["listCard"]
    assert card["header"]["title"] == "실시간 추천"
    assert card["items"][0]["title"] == "1. 삼성전자"
    assert response["template"]["quickReplies"][0]["messageText"] == "추천"


@dataclass
class _RecommendationItem:
    rank: int
    name: str
    symbol: str
    final_score: float
    grade: str = "후보"
    first_buy_ratio: float = 0.5
    reasons: tuple[str, ...] = ("테스트",)


class _RecommendationService:
    def get_recommendations(self, **kwargs):
        return [
            _RecommendationItem(1, "삼성전자", "005930.KS", 250.5),
            _RecommendationItem(2, "SK하이닉스", "000660.KS", 240.0),
        ]


def test_recommendation_request_uses_list_card_when_items_available(tmp_path):
    bot = make_test_bot(tmp_path, recommendation_service=_RecommendationService())
    bot._load_latest_manifest = lambda: None

    response = bot.handle_utterance("추천")

    assert "listCard" in response["template"]["outputs"][0]
    card = response["template"]["outputs"][0]["listCard"]
    assert card["items"][0]["title"] == "1. 삼성전자"


def make_test_bot(tmp_path, **kwargs):
    return KakaoColabPredictionBot(
        result_simple_path=tmp_path / "missing.csv",
        state_path=tmp_path / "jobs.json",
        session_path=tmp_path / "sessions.json",
        **kwargs,
    )


def test_runtime_dir_derives_default_state_session_logs(tmp_path):
    runtime_dir = tmp_path / "drive_runtime"
    cfg = PipelineRuntimeConfig(runtime_dir=str(runtime_dir))
    bot = KakaoColabPredictionBot(runtime_config=cfg)

    assert bot.state_path == runtime_dir / "chatbot_jobs.json"
    assert bot.session_path == runtime_dir / "chatbot_sessions.json"
    assert bot.prewarm_meta_path == runtime_dir / "prewarm_cache_meta.json"
    assert bot.log_dir == runtime_dir / "logs"


def test_startup_marks_stale_running_jobs_failed(tmp_path):
    state_path = tmp_path / "jobs.json"
    state_path.write_text(
        '{"005930.KS":{"symbol":"005930.KS","display_code":"005930.KS",'
        '"command":[],"log_path":"result/runtime/logs/old.log",'
        '"submitted_at":"2026-06-18T00:00:00+00:00","status":"running"}}',
        encoding="utf-8",
    )

    bot = KakaoColabPredictionBot(
        result_simple_path=tmp_path / "missing.csv",
        state_path=state_path,
        session_path=tmp_path / "sessions.json",
    )

    state = bot._job_registry["005930.KS"]
    assert state["status"] == "failed"
    assert state["exit_code"] == -2
    assert state["note"] == "stale_after_restart"
    assert state["completed_at"]


def test_intents_match_phrases_and_punctuation():
    assert is_help_utterance("도움말 좀 알려줘!")
    assert is_status_utterance("결과 확인 부탁")


def test_simple_text_response_truncates_long_text():
    response = simple_text_response("가" * 1200, max_text_length=20)
    text = response["template"]["outputs"][0]["simpleText"]["text"]
    assert len(text) <= 20
    assert text.endswith("...(생략)")


def test_extract_stock_code_rejects_noisy_numeric_tokens(tmp_path):
    bot = make_test_bot(tmp_path)
    assert bot._extract_stock_code("005930") == "005930"
    assert bot._extract_stock_code("005930.KS") == "005930.KS"
    assert bot._extract_stock_code("005930 보여줘") == "005930"
    assert bot._extract_stock_code("abc123") is None
    assert bot._extract_stock_code("12345") is None


def test_kakao_webhook_rejects_missing_or_bad_secret(tmp_path):
    bot = make_test_bot(tmp_path)
    app = create_app(bot=bot, runtime_config=PipelineRuntimeConfig(kakao_webhook_secret="secret"))
    client = app.test_client()
    assert client.post("/kakao/webhook", json={}).status_code == 401
    assert client.post("/kakao/webhook", json={}, headers={"X-Webhook-Secret": "bad"}).status_code == 401


def test_kakao_webhook_accepts_matching_secret(tmp_path):
    bot = make_test_bot(tmp_path)
    app = create_app(bot=bot, runtime_config=PipelineRuntimeConfig(kakao_webhook_secret="secret"))
    client = app.test_client()
    response = client.post("/kakao/webhook", json={}, headers={"X-Webhook-Secret": "secret"})
    assert response.status_code == 200


def test_kakao_webhook_rejects_disallowed_remote_addr(tmp_path):
    bot = make_test_bot(tmp_path)
    cfg = PipelineRuntimeConfig(allowed_webhook_cidrs=("10.0.0.0/8",))
    app = create_app(bot=bot, runtime_config=cfg)
    client = app.test_client()

    response = client.post("/kakao/webhook", json={}, environ_base={"REMOTE_ADDR": "127.0.0.1"})

    assert response.status_code == 403


def test_kakao_webhook_accepts_allowed_remote_addr(tmp_path):
    bot = make_test_bot(tmp_path)
    cfg = PipelineRuntimeConfig(allowed_webhook_cidrs=("127.0.0.1/32",))
    app = create_app(bot=bot, runtime_config=cfg)
    client = app.test_client()

    response = client.post("/kakao/webhook", json={}, environ_base={"REMOTE_ADDR": "127.0.0.1"})

    assert response.status_code == 200


class _DummyProcess:
    pid = 123
    stdout = None


def test_start_prediction_job_deduplicates_running_symbol(tmp_path):
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)
        return _DummyProcess()

    bot = make_test_bot(tmp_path, process_runner=runner)
    assert bot._start_prediction_job("005930.KS") is True
    assert bot._start_prediction_job("005930.KS") is True
    assert len(calls) == 1


def test_start_prediction_job_respects_concurrency_limit(tmp_path):
    def runner(*args, **kwargs):
        return _DummyProcess()

    cfg = PipelineRuntimeConfig(max_concurrent_prediction_jobs=1)
    bot = make_test_bot(tmp_path, runtime_config=cfg, process_runner=runner)
    assert bot._start_prediction_job("005930.KS") is True
    assert bot._start_prediction_job("000660.KS") is False
