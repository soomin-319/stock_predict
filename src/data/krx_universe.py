from __future__ import annotations

from src.data.pykrx_support import import_pykrx_stock


def get_symbol_name_map(symbols: list[str]) -> dict[str, str]:
    """Return mapping of yfinance-style Symbol -> Korean company name."""
    stock = import_pykrx_stock()
    if stock is None:
        return {str(symbol): str(symbol) for symbol in symbols}

    out: dict[str, str] = {}
    for symbol in symbols:
        s = str(symbol)
        ticker = s.split(".")[0]
        try:
            out[s] = stock.get_market_ticker_name(ticker) or s
        except Exception:
            out[s] = s
    return out
