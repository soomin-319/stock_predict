from pathlib import Path

import pandas as pd

from colab import stock_predict_colab as colab_runner
from colab.stock_predict_colab import load_published_predictions


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
        return {
            "manifest": {
                "promoted": True,
                "artifacts": [
                    {"relative_path": "csv/result_detail.csv"},
                    {"relative_path": "csv/result_simple.csv"},
                    {"relative_path": "pipeline_report.json"},
                ]
            }
        }

    monkeypatch.setattr(colab_runner, "save_real_ohlcv_csv", _fake_save)
    monkeypatch.setattr(colab_runner, "run_pipeline", _fake_run_pipeline)

    out = colab_runner.run_colab_pipeline(input_csv="data/sample_ohlcv.csv", report_json="demo.json")

    assert captured["save"]["path"].endswith("data/real_ohlcv.csv")
    assert captured["save"]["symbols"] == ["000270.KS", "005930.KS", "000660.KS"]
    assert captured["run_pipeline"]["input_csv"] == "data/real_ohlcv.csv"
    assert "figure_dir" not in captured["run_pipeline"]
    assert "figure_dir" not in out
    assert out["result_simple_csv"].endswith("result/latest/csv/result_simple.csv")


def test_run_colab_pipeline_returns_current_run_paths_when_not_promoted(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(colab_runner, "_should_bootstrap_default_symbols", lambda input_csv: False)
    run_dir = tmp_path / "result" / "runs" / "run-smoke"
    manifest = {
        "promoted": False,
        "artifacts": [
            {"relative_path": "csv/result_detail.csv"},
            {"relative_path": "csv/result_simple.csv"},
            {"relative_path": "pipeline_report.json"},
        ],
    }
    monkeypatch.setattr(
        colab_runner,
        "run_pipeline",
        lambda **kwargs: {"manifest": manifest, "artifacts": {"pipeline_report_json": str(run_dir / "pipeline_report.json")}},
    )

    out = colab_runner.run_colab_pipeline(input_csv="data/sample_ohlcv.csv")

    assert out["result_detail_csv"] == (run_dir / "csv" / "result_detail.csv").as_posix()
    assert out["result_simple_csv"] == (run_dir / "csv" / "result_simple.csv").as_posix()
    assert out["report_json"] == (run_dir / "pipeline_report.json").as_posix()


def test_result_paths_do_not_fallback_to_stale_compatibility_file_when_manifest_omits_artifact(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    report = {
        "manifest": {
            "promoted": True,
            "artifacts": [{"relative_path": "pipeline_report.json"}],
        }
    }

    out = colab_runner._result_paths_from_report(report)

    assert out["result_detail_csv"] == ""
    assert out["result_simple_csv"] == ""
    assert out["report_json"].endswith("result/latest/pipeline_report.json")


def test_run_colab_pipeline_forwards_news_summary_and_scoring_credentials(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(colab_runner, "_should_bootstrap_default_symbols", lambda input_csv: False)
    monkeypatch.setattr(colab_runner, "run_pipeline", lambda **kwargs: captured.setdefault("run", kwargs))

    colab_runner.run_colab_pipeline(
        input_csv="data/real_ohlcv.csv",
        use_investor_context=True,
        enable_investor_disclosure=False,
        openai_api_key="sk-colab",
        openai_model="gpt-colab",
        naver_client_id="naver-colab-id",
        naver_client_secret="naver-colab-secret",
    )

    assert captured["run"]["use_investor_context"] is True
    assert captured["run"]["enable_investor_disclosure"] is False
    assert captured["run"]["openai_api_key"] == "sk-colab"
    assert captured["run"]["openai_model"] == "gpt-colab"
    assert captured["run"]["naver_client_id"] == "naver-colab-id"
    assert captured["run"]["naver_client_secret"] == "naver-colab-secret"


def test_run_colab_pipeline_incremental_refresh_uses_append(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(colab_runner, "_should_bootstrap_default_symbols", lambda input_csv: False)
    monkeypatch.setattr(colab_runner, "_fallback_symbols_from_input_or_default", lambda _input: ["005930.KS"])
    monkeypatch.setattr(colab_runner, "_resolve_incremental_fetch_start", lambda *_args: "2024-01-10")
    monkeypatch.setattr(colab_runner, "save_real_ohlcv_csv", lambda *args, **kwargs: captured.setdefault("save", True))

    def _fake_append(path, symbols, start):
        captured["append"] = {"path": str(path), "symbols": list(symbols), "start": start}

    monkeypatch.setattr(colab_runner, "append_real_ohlcv_csv", _fake_append)
    monkeypatch.setattr(colab_runner, "run_pipeline", lambda **kwargs: captured.setdefault("run", kwargs))

    colab_runner.run_colab_pipeline(
        input_csv="data/real_ohlcv.csv",
        auto_refresh_real=True,
        real_start="2020-01-01",
    )

    assert "save" not in captured
    assert captured["append"]["path"].endswith("data/real_ohlcv.csv")
    assert captured["append"]["symbols"] == ["005930.KS"]
    assert captured["append"]["start"] == "2024-01-10"
    assert captured["run"]["input_csv"] == "data/real_ohlcv.csv"


def _seed_published(root: Path, date: str):
    for sub in ("latest", f"history/{date}"):
        d = root / "published" / sub / "csv"
        d.mkdir(parents=True, exist_ok=True)
        (d / "result_simple.csv").write_text(
            "종목코드,종목명\n005930,삼성전자\n", encoding="utf-8-sig"
        )
        (root / "published" / sub / "publish_meta.json").write_text(
            f'{{"trading_date": "{date}", "news_mode": "gemma", "symbol_count": 1}}',
            encoding="utf-8",
        )
    (root / "published" / "index.json").write_text(
        f'{{"latest": "{date}", "entries": [{{"trading_date": "{date}"}}]}}',
        encoding="utf-8",
    )


def test_load_published_predictions_latest(tmp_path, monkeypatch):
    _seed_published(tmp_path, "2026-06-17")
    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    paths = load_published_predictions(date=None)
    assert Path(paths["result_simple_csv"]).exists()
    assert paths["trading_date"] == "2026-06-17"


def test_load_published_predictions_specific_date(tmp_path, monkeypatch):
    _seed_published(tmp_path, "2026-06-17")
    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    paths = load_published_predictions(date="2026-06-17")
    assert "history/2026-06-17" in Path(paths["result_simple_csv"]).as_posix()


def test_load_published_predictions_missing_returns_empty_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runner, "PROJECT_ROOT", tmp_path)
    paths = load_published_predictions(date="2099-01-01")
    assert paths["result_simple_csv"] == ""
