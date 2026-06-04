from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import tomllib
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
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr


def test_recommendation_package_import_is_lazy():
    pkg = importlib.import_module("src.recommendation")

    assert pkg.CloseBettingRecommendation is not None


def test_known_p0_mojibake_strings_are_removed():
    chatbot = Path("src/chatbot/kakao_colab_bot.py").read_text(encoding="utf-8")
    realtime = Path("src/recommendation/realtime_close_betting.py").read_text(encoding="utf-8")
    bad_prediction_reason = 'row.get("' + "".join(chr(c) for c in (0x3F, 0xB349, 0xB96B, 0x20, 0x3F, 0xB301, 0xC440)) + '")'
    bad_close_betting_message = '"' + " ".join(["??"] * 5) + '"'

    assert bad_prediction_reason not in chatbot
    assert 'row.get("예측 이유")' in chatbot
    assert bad_close_betting_message not in realtime
    assert '"종가 확정 후 다음 거래일 진입"' in realtime


def test_source_test_and_docs_do_not_contain_mojibake_markers():
    bad_prediction_reason = "".join(chr(c) for c in (0x3F, 0xB349, 0xB96B, 0x20, 0x3F, 0xB301, 0xC440))
    bad_close_betting_message = " ".join(["??"] * 5)
    blocked = ("\ufffd", bad_prediction_reason, bad_close_betting_message)
    checked = [
        path
        for root in (Path("src"), Path("tests"), Path("docs"))
        for path in root.rglob("*")
        if path.suffix.lower() in {".py", ".md"}
    ]

    offenders = []
    for path in checked:
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in blocked):
            offenders.append(str(path))

    assert offenders == []


def test_pytest_cache_uses_ignored_result_directory():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = config["tool"]["pytest"]["ini_options"]

    assert str(pytest_options.get("cache_dir", "")).replace("\\", "/") == "result/.pytest_cache"
    assert "--basetemp=result/" not in str(pytest_options.get("addopts", "")).replace("\\", "/")


def test_text_subprocess_calls_pin_utf8_decoding():
    checked = [
        path
        for root in (Path("src"), Path("news_impact"))
        for path in root.rglob("*.py")
    ]
    offenders: list[str] = []
    for path in checked:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            text_arg = keywords.get("text") or keywords.get("universal_newlines")
            if not (isinstance(text_arg, ast.Constant) and text_arg.value is True):
                continue
            if "encoding" not in keywords or "errors" not in keywords:
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == []
