from __future__ import annotations

from typing import Any

DEFAULT_SIMPLE_TEXT_MAX_LENGTH = 900


def _truncate_text(text: str, max_text_length: int) -> str:
    message = str(text or "")
    if max_text_length <= 0 or len(message) <= max_text_length:
        return message
    suffix = "...(생략)"
    if max_text_length <= len(suffix):
        return suffix[:max_text_length]
    return message[: max_text_length - len(suffix)].rstrip() + suffix


def simple_text_response(text: str, max_text_length: int = DEFAULT_SIMPLE_TEXT_MAX_LENGTH) -> dict[str, Any]:
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": _truncate_text(text, max_text_length)}},
            ],
        },
    }


def quick_reply(label: str, message_text: str) -> dict[str, str]:
    return {"action": "message", "label": label, "messageText": message_text}


def attach_quick_replies(response: dict[str, Any], quick_replies: list[tuple[str, str]] | None) -> dict[str, Any]:
    if quick_replies:
        response["template"]["quickReplies"] = [quick_reply(label, message_text) for label, message_text in quick_replies]
    return response
