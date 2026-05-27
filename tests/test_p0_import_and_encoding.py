from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def test_pipeline_imports_without_market_or_llm_extras():
    """--disable-external/basic import path must not require live API packages."""
    code = r'''
import importlib.abc
import sys

BLOCKED = {"yfinance", "openai", "pykrx", "joblib", "sklearn", "lightgbm"}
for name in list(sys.modules):
    if name.split(".", 1)[0] in BLOCKED:
        del sys.modules[name]

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BLOCKED:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockOptionalImports())
import src.pipeline
'''
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)

    assert result.returncode == 0, result.stderr


def test_recommendation_package_import_is_lazy():
    pkg = importlib.import_module("src.recommendation")

    assert pkg.CloseBettingRecommendation is not None


def test_known_p0_mojibake_strings_are_removed():
    chatbot = Path("src/chatbot/kakao_colab_bot.py").read_text(encoding="utf-8")
    realtime = Path("src/recommendation/realtime_close_betting.py").read_text(encoding="utf-8")

    assert 'row.get("?덉륫 ?댁쑀")' not in chatbot
    assert 'row.get("예측 이유")' in chatbot
    assert '"?? ?? ?? ?? ??"' not in realtime
    assert '"종가 확정 후 다음 거래일 진입"' in realtime
