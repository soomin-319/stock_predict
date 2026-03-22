from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.data.pykrx_support import import_pykrx_stock


KRX_SYMBOL_NAME_CSV = Path(__file__).resolve().parents[2] / "data" / "krx_symbol_name_map.csv"


@lru_cache(maxsize=1)
def _load_krx_symbol_name_df() -> pd.DataFrame:
    path = KRX_SYMBOL_NAME_CSV
    if not path.exists():
        return pd.DataFrame(columns=["Ticker", "Symbol", "Name", "Market"])

    df = pd.read_csv(path)
    expected = {"Ticker", "Symbol", "Name", "Market"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"KRX symbol-name CSV must include columns: {sorted(expected)}")

    out = df[["Ticker", "Symbol", "Name", "Market"]].copy()
    out["Ticker"] = out["Ticker"].astype(str).str.zfill(6)
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["Market"] = out["Market"].astype(str).str.strip()
    out = out[(out["Ticker"] != "") & (out["Symbol"] != "") & (out["Name"] != "")]
    out = out.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
    return out



def _normalize_name(text: str) -> str:
    return "".join(str(text).strip().lower().split())



def _csv_symbol_name_map() -> tuple[dict[str, str], dict[str, str]]:
    df = _load_krx_symbol_name_df()
    by_symbol = dict(zip(df["Symbol"], df["Name"]))
    by_ticker = dict(zip(df["Ticker"], df["Name"]))
    return by_symbol, by_ticker



def get_symbol_name_map(symbols: list[str]) -> dict[str, str]:
    """Return mapping of yfinance-style Symbol -> Korean company name."""
    by_symbol, by_ticker = _csv_symbol_name_map()
    out: dict[str, str] = {}
    unresolved: list[str] = []

    for symbol in symbols:
        s = str(symbol)
        ticker = s.split(".")[0].zfill(6)
        name = by_symbol.get(s) or by_ticker.get(ticker)
        if name:
            out[s] = name
        else:
            unresolved.append(s)

    if not unresolved:
        return out

    stock = import_pykrx_stock()
    if stock is None:
        out.update({str(symbol): str(symbol) for symbol in unresolved})
        return out

    for symbol in unresolved:
        s = str(symbol)
        ticker = s.split(".")[0].zfill(6)
        try:
            out[s] = stock.get_market_ticker_name(ticker) or s
        except Exception:
            out[s] = s
    return out



def _score_name_match(normalized_query: str, normalized_name: str) -> float:
    if normalized_query == normalized_name:
        return 1.0
    if normalized_query in normalized_name:
        return 0.9
    return SequenceMatcher(None, normalized_query, normalized_name).ratio()



def find_symbol_candidates_by_name(query: str, limit: int | None = None) -> list[dict[str, str | float]]:
    normalized_query = _normalize_name(query)
    if not normalized_query:
        return []

    df = _load_krx_symbol_name_df()
    records: list[dict[str, str | float]] = []
    for row in df.itertuples(index=False):
        normalized_name = _normalize_name(row.Name)
        score = _score_name_match(normalized_query, normalized_name)
        if score < 0.45:
            continue
        records.append(
            {
                "symbol": str(row.Symbol),
                "ticker": str(row.Ticker),
                "name": str(row.Name),
                "market": str(row.Market),
                "score": float(score),
            }
        )

    deduped: dict[str, dict[str, str | float]] = {}
    for record in sorted(records, key=lambda item: (-float(item["score"]), str(item["name"]), str(item["ticker"]))):
        deduped.setdefault(str(record["ticker"]), record)

    if deduped:
        candidates = list(deduped.values())
        return candidates if limit is None else candidates[:limit]

    stock = import_pykrx_stock()
    if stock is None:
        return []

    fallback_records: list[dict[str, str | float]] = []
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
            score = _score_name_match(normalized_query, _normalize_name(name))
            if score < 0.45:
                continue
            fallback_records.append(
                {
                    "symbol": f"{ticker}{suffix}",
                    "ticker": str(ticker).zfill(6),
                    "name": str(name),
                    "market": market,
                    "score": float(score),
                }
            )

    deduped_fallback: dict[str, dict[str, str | float]] = {}
    for record in sorted(fallback_records, key=lambda item: (-float(item["score"]), str(item["name"]), str(item["ticker"]))):
        deduped_fallback.setdefault(str(record["ticker"]), record)
    fallback_candidates = list(deduped_fallback.values())
    return fallback_candidates if limit is None else fallback_candidates[:limit]
