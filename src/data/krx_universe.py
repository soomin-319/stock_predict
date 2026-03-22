from __future__ import annotations

from difflib import SequenceMatcher

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


def _normalize_name(text: str) -> str:
    return "".join(str(text).strip().lower().split())


def find_symbol_candidates_by_name(query: str, limit: int = 5) -> list[dict[str, str | float]]:
    stock = import_pykrx_stock()
    normalized_query = _normalize_name(query)
    if stock is None or not normalized_query:
        return []

    records: list[dict[str, str | float]] = []
    for market, suffix in (("KOSPI", ".KS"), ("KOSDAQ", ".KQ")):
        try:
            tickers = stock.get_market_ticker_list(market=market)
        except Exception:
            continue
        for ticker in tickers:
            try:
                name = stock.get_market_ticker_name(ticker)
            except Exception:
                continue
            if not name:
                continue
            normalized_name = _normalize_name(name)
            if normalized_query == normalized_name:
                score = 1.0
            elif normalized_query in normalized_name:
                score = 0.9
            else:
                score = SequenceMatcher(None, normalized_query, normalized_name).ratio()
            if score < 0.45:
                continue
            records.append(
                {
                    "symbol": f"{ticker}{suffix}",
                    "ticker": ticker,
                    "name": str(name),
                    "market": market,
                    "score": float(score),
                }
            )

    deduped: dict[str, dict[str, str | float]] = {}
    for record in sorted(records, key=lambda item: (-float(item["score"]), str(item["name"]), str(item["ticker"]))):
        deduped.setdefault(str(record["ticker"]), record)
    return list(deduped.values())[:limit]
