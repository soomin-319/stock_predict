import pandas as pd

from src import pipeline
from src.data import fetch_real_data


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

    assert symbols == ["207940.KS", "034020.KS", "035420.KS", "272210.KS", "042660.KS", "000660.KS", "042700.KS", "006400.KS", "015760.KS", "051910.KS", "005380.KS", "005930.KS"]


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
