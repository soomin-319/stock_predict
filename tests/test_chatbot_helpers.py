from src.chatbot.intent import is_help_utterance, is_status_utterance, normalize_utterance
from src.chatbot.responses import simple_text_response


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
