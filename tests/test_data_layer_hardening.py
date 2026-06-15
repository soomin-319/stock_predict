from __future__ import annotations

import pandas as pd

from src.data import fetch_real_data as fr
from src.data.cleaners import clean_ohlcv
from src.data.fetch_real_data import normalize_user_symbols
from src.data.loaders import load_ohlcv_csv


def _base_rows() -> dict[str, list]:
    return {
        "Date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
        "Symbol": ["AAA", "AAA", "AAA", "AAA"],
        "Open": [100, 200, 210, 205],
        "High": [101, 205, 215, 210],
        "Low": [99, 195, 205, 200],
        "Close": [100, 200, 210, 205],
        "Volume": [100, 200, 500, 0],
    }


def test_load_ohlcv_csv_accepts_utf8_bom_and_symbol_argument(tmp_path):
    path = tmp_path / "bom.csv"
    frame = pd.DataFrame(_base_rows()).drop(columns="Symbol").iloc[:1]
    frame.to_csv(path, index=False, encoding="utf-8-sig")

    out = load_ohlcv_csv(path, symbol="005930.KS")

    assert out["Symbol"].unique().tolist() == ["005930.KS"]


def test_clean_ohlcv_selects_highest_volume_duplicate_and_flags_quality():
    out = clean_ohlcv(pd.DataFrame(_base_rows()))

    assert out.loc[out["Date"].eq(pd.Timestamp("2024-01-02")), "Volume"].item() == 500
    assert out["is_zero_volume"].tolist() == [False, False, True]
    assert out["is_extreme_return"].tolist() == [False, True, False]


def test_clean_ohlcv_fills_missing_symbol_and_coerces_date():
    frame = pd.DataFrame(_base_rows()).drop(columns="Symbol").iloc[:1]

    out = clean_ohlcv(frame)

    assert out["Symbol"].tolist() == ["UNKNOWN"]
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])


def test_normalize_known_kosdaq_ticker_uses_kq():
    assert normalize_user_symbols(["247540"]) == ["247540.KQ"]


def test_fetch_unknown_korean_ticker_falls_back_to_kq(monkeypatch):
    calls: list[str] = []

    def fake_download(symbol: str, start: str, end: str | None = None):
        calls.append(symbol)
        if symbol.endswith(".KS"):
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
        )
        return frame

    monkeypatch.setattr(fr, "_safe_download_ohlcv", fake_download)

    out = fr.fetch_real_ohlcv(["999999.KS"], start="2024-01-01")

    assert calls == ["999999.KS", "999999.KQ"]
    assert out["Symbol"].unique().tolist() == ["999999.KQ"]


def test_real_download_uses_adjusted_prices_and_retries(monkeypatch):
    calls: list[bool] = []

    class FakeTicker:
        def history(self, **kwargs):
            calls.append(kwargs["auto_adjust"])
            if len(calls) == 1:
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    "Open": [100],
                    "High": [101],
                    "Low": [99],
                    "Close": [100],
                    "Volume": [1000],
                },
                index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
            )

    class FakeYFinance:
        @staticmethod
        def Ticker(_symbol):
            return FakeTicker()

    monkeypatch.setattr(fr, "_get_yfinance", lambda: FakeYFinance())
    monkeypatch.setattr(fr, "_sleep", lambda _seconds: None, raising=False)

    out = fr.fetch_real_ohlcv(["005930.KS"], start="2024-01-01")
    coverage = fr.get_last_fetch_coverage()

    assert calls == [True, True]
    assert not out.empty
    assert coverage["retried_symbols"] == ["005930.KS"]
    assert coverage["total_retry_count"] == 1


def test_fetch_coverage_reports_partial_failure(monkeypatch):
    def fake_download(symbol: str, start: str, end: str | None = None):
        if symbol == "BAD":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
        )

    monkeypatch.setattr(fr, "_safe_download_ohlcv", fake_download)

    fr.fetch_real_ohlcv(["GOOD", "BAD"], start="2024-01-01")
    coverage = fr.get_last_fetch_coverage()

    assert coverage["requested"] == 2
    assert coverage["successful"] == 1
    assert coverage["failed"] == 1
    assert coverage["failed_symbols"] == ["BAD"]
    assert coverage["success_ratio"] == 0.5


def test_fetch_coverage_does_not_count_market_fallback_as_retry(monkeypatch):
    calls: list[str] = []

    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, **_kwargs):
            calls.append(self.symbol)
            if self.symbol.endswith(".KS"):
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    "Open": [100],
                    "High": [101],
                    "Low": [99],
                    "Close": [100],
                    "Volume": [1000],
                },
                index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
            )

    class FakeYFinance:
        @staticmethod
        def Ticker(symbol):
            return FakeTicker(symbol)

    monkeypatch.setattr(fr, "_get_yfinance", lambda: FakeYFinance())
    monkeypatch.setattr(fr, "_sleep", lambda _seconds: None)

    fr.fetch_real_ohlcv(["999999.KS"], start="2024-01-01")
    coverage = fr.get_last_fetch_coverage()

    assert calls == ["999999.KS", "999999.KS", "999999.KS", "999999.KQ"]
    assert coverage["fallback_used"] == 1
    assert coverage["total_retry_count"] == 2


def test_save_real_ohlcv_csv_writes_utf8_bom(tmp_path, monkeypatch):
    path = tmp_path / "real.csv"
    fetched = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02"]),
            "Symbol": ["005930.KS"],
            "Open": [100],
            "High": [101],
            "Low": [99],
            "Close": [100],
            "Volume": [1000],
        }
    )
    monkeypatch.setattr(fr, "fetch_real_ohlcv", lambda *args, **kwargs: fetched.copy())

    fr.save_real_ohlcv_csv(path, ["005930.KS"])

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_cli_real_start_uses_data_layer_default():
    from src import pipeline

    args = pipeline.build_cli_parser().parse_args([])

    assert args.real_start == fr.DEFAULT_REAL_START_DATE


def test_load_pipeline_config_excludes_quality_flagged_rows(tmp_path):
    from src import pipeline

    path = tmp_path / "quality.csv"
    pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "Symbol": ["AAA"] * 4,
            "Open": [100, 200, 202, 203],
            "High": [101, 201, 203, 204],
            "Low": [99, 199, 201, 202],
            "Close": [100, 200, 202, 203],
            "Volume": [1000, 1000, 0, 1000],
        }
    ).to_csv(path, index=False)

    _, _, cleaned, data, _ = pipeline._load_pipeline_config_and_data(
        str(path), None, None, {}
    )

    assert cleaned[["is_zero_volume", "is_extreme_return"]].any(axis=1).sum() == 2
    assert not data[["is_zero_volume", "is_extreme_return"]].any(axis=1).any()


def test_pipeline_cleaning_selects_highest_volume_before_loader_dedup(tmp_path):
    from src import pipeline

    path = tmp_path / "duplicates.csv"
    frame = pd.DataFrame(_base_rows())
    frame.loc[1, "Volume"] = 500
    frame.loc[2, "Volume"] = 200
    frame.to_csv(path, index=False)

    _, _, cleaned, _, _ = pipeline._load_pipeline_config_and_data(
        str(path), None, None, {}
    )

    selected = cleaned.loc[cleaned["Date"].eq(pd.Timestamp("2024-01-02"))]
    assert selected["Volume"].item() == 500
