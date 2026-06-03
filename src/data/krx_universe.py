from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_KRX_SYMBOL_NAME_CSV = DATA_DIR / "krx_symbol_name_map.csv"
DEFAULT_KOSPI200_SYMBOL_NAME_CSV = DATA_DIR / "kospi200_symbol_name_map.csv"
KRX_SYMBOL_NAME_CSV = DEFAULT_KRX_SYMBOL_NAME_CSV
KOSPI200_SYMBOL_NAME_CSV = DEFAULT_KOSPI200_SYMBOL_NAME_CSV


def _read_symbol_name_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not str(path) or not path.exists() or not path.is_file():
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


@lru_cache(maxsize=8)
def _load_krx_symbol_name_df_cached(path_str: str, kospi200_path_str: str) -> pd.DataFrame:
    primary = _read_symbol_name_csv(Path(path_str))
    fallback = _read_symbol_name_csv(Path(kospi200_path_str))
    combined = pd.concat([primary, fallback], ignore_index=True)
    return combined.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)


def _load_krx_symbol_name_df() -> pd.DataFrame:
    kospi200_path = KOSPI200_SYMBOL_NAME_CSV
    if Path(KRX_SYMBOL_NAME_CSV) != DEFAULT_KRX_SYMBOL_NAME_CSV and kospi200_path == DEFAULT_KOSPI200_SYMBOL_NAME_CSV:
        kospi200_path = Path("")
    return _load_krx_symbol_name_df_cached(str(KRX_SYMBOL_NAME_CSV), str(kospi200_path))


_load_krx_symbol_name_df.cache_clear = _load_krx_symbol_name_df_cached.cache_clear  # type: ignore[attr-defined]
_load_krx_symbol_name_df.cache_info = _load_krx_symbol_name_df_cached.cache_info  # type: ignore[attr-defined]



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

    if unresolved:
        out.update({str(symbol): str(symbol) for symbol in unresolved})
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

    candidates = list(deduped.values())
    return candidates if limit is None else candidates[:limit]
