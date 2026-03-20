from importlib import import_module
from typing import Any

__all__ = ["KakaoColabPredictionBot", "PipelineRuntimeConfig", "create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("src.chatbot.kakao_colab_bot")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
