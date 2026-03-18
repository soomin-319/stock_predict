from src.data import krx_universe


def test_get_symbol_name_map_falls_back_to_symbols_when_pykrx_missing(monkeypatch):
    monkeypatch.setattr(krx_universe, "import_pykrx_stock", lambda: None)

    assert krx_universe.get_symbol_name_map(["005930.KS"]) == {"005930.KS": "005930.KS"}


def test_get_symbol_name_map_uses_pykrx_when_available(monkeypatch):
    class _Stock:
        @staticmethod
        def get_market_ticker_name(ticker):
            return {"005930": "삼성전자"}.get(ticker)

    monkeypatch.setattr(krx_universe, "import_pykrx_stock", lambda: _Stock())

    assert krx_universe.get_symbol_name_map(["005930.KS", "000660.KS"]) == {
        "005930.KS": "삼성전자",
        "000660.KS": "000660.KS",
    }
