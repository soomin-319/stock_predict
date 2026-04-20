import pandas as pd

from src import pipeline
from src.data import fetch_real_data
from src.data import universe as universe_module


def test_fallback_symbols_loads_repo_managed_default_universe():
    symbols = pipeline._fallback_symbols_from_input_or_default("dummy.csv")

    assert len(symbols) == 5
    assert symbols[:5] == [
        "005930.KS",
        "000660.KS",
        "373220.KS",
        "207940.KS",
        "005380.KS",
    ]



def test_fallback_symbols_match_default_universe_csv_contents():
    symbols = pipeline._fallback_symbols_from_input_or_default("dummy.csv")

    assert symbols == universe_module.load_default_universe_symbols()[:5]





def test_cli_real_start_defaults_to_today():
    parser = pipeline.build_cli_parser()
    args = parser.parse_args([])

    assert args.real_start == "2020-01-01"
    assert args.auto_refresh_real is False


def test_resolve_incremental_fetch_start_uses_next_day_of_existing_csv(tmp_path):
    target = tmp_path / "real.csv"
    pd.DataFrame({"Date": ["2024-01-05", "2024-01-08"]}).to_csv(target, index=False)

    start = pipeline._resolve_incremental_fetch_start(str(target), "2020-01-01")

    assert start == "2024-01-09"


def test_resolve_incremental_fetch_start_respects_later_requested_start(tmp_path):
    target = tmp_path / "real.csv"
    pd.DataFrame({"Date": ["2024-01-05", "2024-01-08"]}).to_csv(target, index=False)

    start = pipeline._resolve_incremental_fetch_start(str(target), "2024-02-01")

    assert start == "2024-02-01"


def test_resolve_fetch_symbols_uses_universe_csv_when_provided(monkeypatch):
    monkeypatch.setattr(pipeline, "load_universe_symbols", lambda _p: ["A.KS", "B.KS"])

    symbols = pipeline._resolve_fetch_symbols(None, "dummy_universe.csv", "data/real_ohlcv.csv")

    assert symbols == ["A.KS", "B.KS"]


def test_resolve_fetch_symbols_falls_back_to_default_when_universe_load_fails(monkeypatch):
    monkeypatch.setattr(pipeline, "load_universe_symbols", lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(pipeline, "_fallback_symbols_from_input_or_default", lambda _p: ["005930.KS"])

    symbols = pipeline._resolve_fetch_symbols(None, "dummy_universe.csv", "data/real_ohlcv.csv")

    assert symbols == ["005930.KS"]


def test_main_add_symbols_skips_incremental_auto_refresh(monkeypatch):
    calls: dict[str, int] = {"append": 0, "save": 0, "run": 0}

    monkeypatch.setattr(
        pipeline,
        "append_real_ohlcv_csv",
        lambda *args, **kwargs: calls.__setitem__("append", calls["append"] + 1),
    )
    monkeypatch.setattr(
        pipeline,
        "save_real_ohlcv_csv",
        lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1),
    )
    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda *args, **kwargs: calls.__setitem__("run", calls["run"] + 1),
    )

    parser = pipeline.build_cli_parser()
    args = parser.parse_args(["--add-symbols", "005930", "--auto-refresh-real"])
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "build_cli_parser", lambda: parser)

    pipeline.main()

    assert calls["append"] == 1
    assert calls["save"] == 0
    assert calls["run"] == 1


def test_is_default_real_ohlcv_path_supports_relative_and_absolute_paths(tmp_path):
    assert pipeline._is_default_real_ohlcv_path("data/real_ohlcv.csv") is True
    assert pipeline._is_default_real_ohlcv_path(str(tmp_path / "data" / "real_ohlcv.csv")) is True
    assert pipeline._is_default_real_ohlcv_path("data/sample_ohlcv.csv") is False


def test_resolve_fetch_symbols_uses_universe_csv_when_provided(monkeypatch):
    monkeypatch.setattr(pipeline, "load_universe_symbols", lambda _p: ["A.KS", "B.KS"])

    symbols = pipeline._resolve_fetch_symbols(None, "dummy_universe.csv", "data/real_ohlcv.csv")

    assert symbols == ["A.KS", "B.KS"]


def test_resolve_fetch_symbols_falls_back_to_default_when_universe_load_fails(monkeypatch):
    monkeypatch.setattr(pipeline, "load_universe_symbols", lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(pipeline, "_fallback_symbols_from_input_or_default", lambda _p: ["005930.KS"])

    symbols = pipeline._resolve_fetch_symbols(None, "dummy_universe.csv", "data/real_ohlcv.csv")

    assert symbols == ["005930.KS"]

def test_save_real_ohlcv_csv_preserves_existing_optional_columns(tmp_path, monkeypatch):
    target = tmp_path / "real.csv"
    base = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Symbol": ["005930.KS", "005930.KS"],
            "Open": [100, 101],
            "High": [101, 102],
            "Low": [99, 100],
            "Close": [100.5, 101.5],
            "Volume": [1000, 1100],
            "foreign_net_buy": [12345, 23456],
            "pbr": [1.1, 1.2],
            "warning_level": [1, 2],
        }
    )
    base.to_csv(target, index=False)

    fetched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Symbol": ["005930.KS", "005930.KS"],
            "Open": [200, 201],
            "High": [202, 203],
            "Low": [198, 199],
            "Close": [201, 202],
            "Volume": [3000, 3100],
        }
    )
    monkeypatch.setattr(fetch_real_data, "fetch_real_ohlcv", lambda *a, **k: fetched.copy())

    fetch_real_data.save_real_ohlcv_csv(target, symbols=["005930"])

    out = pd.read_csv(target)
    assert out["foreign_net_buy"].tolist() == [12345, 23456]
    assert out["pbr"].tolist() == [1.1, 1.2]
    assert out["warning_level"].tolist() == [1, 2]
    assert out["Close"].tolist() == [201, 202]



def test_append_real_ohlcv_csv_preserves_existing_optional_columns_on_overlap(tmp_path, monkeypatch):
    target = tmp_path / "real.csv"
    base = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Symbol": ["005930.KS"],
            "Open": [100],
            "High": [101],
            "Low": [99],
            "Close": [100.5],
            "Volume": [1000],
            "foreign_net_buy": [12345],
            "program_trading_flow": [777],
            "buyback_flag": [1],
        }
    )
    base.to_csv(target, index=False)

    fetched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Symbol": ["005930.KS", "005930.KS"],
            "Open": [200, 201],
            "High": [202, 203],
            "Low": [198, 199],
            "Close": [201, 202],
            "Volume": [3000, 3100],
        }
    )
    monkeypatch.setattr(fetch_real_data, "fetch_real_ohlcv", lambda *a, **k: fetched.copy())

    fetch_real_data.append_real_ohlcv_csv(target, symbols=["005930"])

    out = pd.read_csv(target).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    assert len(out) == 2
    assert out.loc[0, "foreign_net_buy"] == 12345
    assert out.loc[0, "program_trading_flow"] == 777
    assert out.loc[0, "buyback_flag"] == 1
    assert pd.isna(out.loc[1, "foreign_net_buy"])
