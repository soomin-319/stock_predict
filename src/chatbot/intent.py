from __future__ import annotations

HELP_UTTERANCES = {"도움", "도움말", "help", "?", "사용법", "시작", "안내"}
STATUS_UTTERANCES = {"결과", "상태", "진행상황", "조회", "확인", "status"}


def normalize_utterance(value: object) -> str:
    return str(value or "").strip()


def is_status_utterance(value: object) -> bool:
    return normalize_utterance(value).lower() in STATUS_UTTERANCES


def is_help_utterance(value: object) -> bool:
    return normalize_utterance(value).lower() in HELP_UTTERANCES
