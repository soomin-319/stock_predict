import pandas as pd

from src.data.krx_universe import _resolve_market_cap_col, _top_by_market_cap, get_kospi200_kosdaq150_symbols


def test_resolve_market_cap_col_with_english_name():
    df = pd.DataFrame({"MarketCap": [3, 2, 1]}, index=["A", "B", "C"])
    assert _resolve_market_cap_col(df) == "MarketCap"


def test_top_by_market_cap_handles_non_korean_schema():
    df = pd.DataFrame({"market_cap": [10, 30, 20]}, index=["000001", "000002", "000003"])
    out = _top_by_market_cap(df, 2)
    assert out == ["000002", "000003"]


def test_get_kospi200_kosdaq150_symbols_with_marketcap_alias(monkeypatch):
    class _Stock:
        def get_market_ohlcv_by_ticker(self, date):
            return pd.DataFrame({"dummy": [1]}, index=["000001"])

        def get_market_cap_by_ticker(self, date, market="KOSPI"):
            count = 220 if market == "KOSPI" else 180
            idx = [f"{i:06d}" for i in range(1, count + 1)]
            vals = list(range(count, 0, -1))
            return pd.DataFrame({"MarketCap": vals}, index=idx)

    monkeypatch.setattr("src.data.krx_universe._import_pykrx_stock", lambda: _Stock())

    symbols = get_kospi200_kosdaq150_symbols()
    assert len(symbols) == 350
    assert symbols[0].endswith(".KS")
    assert any(s.endswith(".KQ") for s in symbols)
