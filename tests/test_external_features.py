from __future__ import annotations

import pandas as pd

from src.features import external_features



def test_add_external_market_features_handles_multi_column_reset_index_shape(monkeypatch):
    base = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "Symbol": ["AAA", "AAA"]})

    weird = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Adj Close": [100.0, 101.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    weird.index.name = "Date"

    monkeypatch.setattr(external_features, "_safe_download", lambda symbol, start, end: weird.copy())

    out, coverage = external_features.add_external_market_features_with_coverage(base, ["^GSPC"])

    assert coverage["successful"] == 1
    assert "gspc_close" in out.columns
    assert pd.isna(out["gspc_close"].iloc[0])
    assert out["gspc_close"].iloc[1] == 100.0


def test_external_download_uses_adjusted_prices_and_retries(monkeypatch):
    calls: list[bool] = []

    class FakeYFinance:
        @staticmethod
        def download(*args, **kwargs):
            calls.append(kwargs["auto_adjust"])
            if len(calls) == 1:
                return pd.DataFrame()
            return pd.DataFrame(
                {"Close": [100.0]},
                index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
            )

    monkeypatch.setattr(external_features, "_get_yfinance", lambda: FakeYFinance())
    monkeypatch.setattr(external_features, "_sleep", lambda _seconds: None, raising=False)

    out = external_features._safe_download("^GSPC", "2024-01-01", "2024-01-03")

    assert calls == [True, True]
    assert out.tolist() == [100.0]


def _external_series() -> pd.Series:
    return pd.Series(
        [100.0, 101.0],
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date"),
    )


def _base_external_dates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "Symbol": ["AAA"] * 3,
        }
    )


def test_overseas_external_features_are_delayed_one_observation(monkeypatch):
    monkeypatch.setattr(external_features, "_safe_download", lambda symbol, start, end: _external_series())

    out, _ = external_features.add_external_market_features_with_coverage(_base_external_dates(), ["^GSPC"])

    assert pd.isna(out.loc[out["Date"].eq("2024-01-02"), "gspc_close"]).all()
    assert out.loc[out["Date"].eq("2024-01-03"), "gspc_close"].iloc[0] == 100.0


def test_korean_external_features_remain_same_date(monkeypatch):
    monkeypatch.setattr(external_features, "_safe_download", lambda symbol, start, end: _external_series())

    out, _ = external_features.add_external_market_features_with_coverage(_base_external_dates(), ["^KS11"])

    assert out.loc[out["Date"].eq("2024-01-02"), "ks11_close"].iloc[0] == 100.0


def test_external_features_do_not_backfill_leading_dates(monkeypatch):
    series = pd.Series(
        [100.0],
        index=pd.DatetimeIndex(["2024-01-03"], name="Date"),
    )
    monkeypatch.setattr(external_features, "_safe_download", lambda symbol, start, end: series)

    out, _ = external_features.add_external_market_features_with_coverage(_base_external_dates(), ["^KS11"])

    assert pd.isna(out.loc[out["Date"].eq("2024-01-02"), "ks11_close"]).all()
