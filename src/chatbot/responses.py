from __future__ import annotations

from typing import Any


def simple_text_response(text: str) -> dict[str, Any]:
    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": text}}]}}


def quick_reply(label: str, message_text: str) -> dict[str, str]:
    return {"action": "message", "label": label, "messageText": message_text}


def attach_quick_replies(response: dict[str, Any], quick_replies: list[tuple[str, str]] | None) -> dict[str, Any]:
    if quick_replies:
        response["template"]["quickReplies"] = [quick_reply(label, message_text) for label, message_text in quick_replies]
    return response
