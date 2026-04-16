from pathlib import Path

import pandas as pd

from colab import stock_predict_colab as colab_runner


def test_should_bootstrap_default_symbols_for_demo_symbols(tmp_path: Path):
    target = tmp_path / "sample.csv"
    pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"]}).to_csv(target, index=False)

    assert colab_runner._should_bootstrap_default_symbols(target) is True


def test_should_not_bootstrap_default_symbols_for_krx_symbols(tmp_path: Path):
    target = tmp_path / "real.csv"
    pd.DataFrame({"Symbol": ["005930.KS", "000660.KS"]}).to_csv(target, index=False)

    assert colab_runner._should_bootstrap_default_symbols(target) is False


def test_run_colab_pipeline_bootstraps_default_krx_symbols(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(colab_runner, "_should_bootstrap_default_symbols", lambda input_csv: True)
    monkeypatch.setattr(
        colab_runner,
        "_fallback_symbols_from_input_or_default",
        lambda input_csv: ["000270.KS", "005930.KS", "000660.KS"],
    )

    def _fake_save(path, symbols, start):
        captured["save"] = {"path": str(path), "symbols": list(symbols), "start": start}

    def _fake_run_pipeline(**kwargs):
        captured["run_pipeline"] = kwargs

    monkeypatch.setattr(colab_runner, "save_real_ohlcv_csv", _fake_save)
    monkeypatch.setattr(colab_runner, "run_pipeline", _fake_run_pipeline)

    out = colab_runner.run_colab_pipeline(input_csv="data/sample_ohlcv.csv", report_json="demo.json", figure_dir="figs")

    assert captured["save"]["path"].endswith("data/real_ohlcv.csv")
    assert captured["save"]["symbols"] == ["000270.KS", "005930.KS", "000660.KS"]
    assert captured["run_pipeline"]["input_csv"] == "data/real_ohlcv.csv"
    assert out["result_simple_csv"].endswith("result/result_simple.csv")
