import pandas as pd

from src import pipeline


def test_fallback_symbols_uses_input_symbols(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "load_ohlcv_csv",
        lambda _path: pd.DataFrame({"Symbol": ["005930.KS", "000660.KS", "005930.KS"]}),
    )

    symbols = pipeline._fallback_symbols_from_input_or_default("dummy.csv")

    assert symbols == ["000660.KS", "005930.KS"]


def test_fallback_symbols_uses_default_when_input_unavailable(monkeypatch):
    def _raise(_path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(pipeline, "load_ohlcv_csv", _raise)

    symbols = pipeline._fallback_symbols_from_input_or_default("dummy.csv")

    assert symbols == pipeline.DEFAULT_REAL_SYMBOLS
