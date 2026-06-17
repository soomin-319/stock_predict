from __future__ import annotations

import re

HELP_UTTERANCES = {"도움", "도움말", "help", "?", "사용법", "시작", "안내"}
STATUS_UTTERANCES = {"결과", "상태", "진행상황", "조회", "확인", "status"}


def normalize_utterance(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\.,!?\-_/]+", " ", text)
    return text.strip()


def _contains_keyword(value: object, keywords: set[str]) -> bool:
    normalized = normalize_utterance(value)
    compact = normalized.replace(" ", "")
    return any(keyword in normalized or keyword in compact for keyword in keywords)


def is_status_utterance(value: object) -> bool:
    return _contains_keyword(value, STATUS_UTTERANCES)


def is_help_utterance(value: object) -> bool:
    return _contains_keyword(value, HELP_UTTERANCES)
