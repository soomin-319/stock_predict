from __future__ import annotations

from datetime import datetime, timedelta
import warnings

import pandas as pd


def _import_pykrx_stock():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from pykrx import stock

        return stock
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pykrx is required to resolve Korean ticker names") from exc


def _latest_business_day(stock) -> str:
    d = datetime.now().date()
    for _ in range(14):
        s = d.strftime("%Y%m%d")
        try:
            if not stock.get_market_ohlcv_by_ticker(s).empty:
                return s
        except Exception:
            pass
        d -= timedelta(days=1)
    return datetime.now().strftime("%Y%m%d")


def _resolve_market_cap_col(df: pd.DataFrame) -> str | None:
    candidates = ["시가총액", "MarketCap", "market_cap", "MARCAP"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "cap" in str(col).lower() and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _resolve_value_col(df: pd.DataFrame) -> str | None:
    candidates = ["거래대금", "Value", "value", "TradingValue", "trading_value"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "value" in str(col).lower() and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _top_by_market_cap(df: pd.DataFrame, n: int) -> list[str]:
    if df is None or df.empty:
        return []
    cap_col = _resolve_market_cap_col(df)
    if cap_col is None:
        return []
    work = df.copy()
    work[cap_col] = pd.to_numeric(work[cap_col], errors="coerce")
    work = work.dropna(subset=[cap_col])
    if work.empty:
        return []
    return work.sort_values(cap_col, ascending=False).head(n).index.astype(str).tolist()


def _top_by_trading_value(df: pd.DataFrame, n: int) -> list[str]:
    if df is None or df.empty:
        return []
    value_col = _resolve_value_col(df)
    if value_col is None:
        return []
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return []
    return work.sort_values(value_col, ascending=False).head(n).index.astype(str).tolist()


def get_kospi200_kosdaq150_symbols(as_of: str | None = None) -> list[str]:
    """Return 350 yfinance-compatible tickers (.KS/.KQ).

    Source priority:
    1) Official KRX index constituents (KOSPI200/KOSDAQ150).
    2) Market-cap ranking fallback.
    3) Trading-value ranking fallback.
    """
    stock = _import_pykrx_stock()
    date = as_of or _latest_business_day(stock)

    def _index_constituents(index_code: str) -> list[str]:
        try:
            rows = stock.get_index_portfolio_deposit_file(index_code, date)
        except TypeError:
            rows = stock.get_index_portfolio_deposit_file(index_code)
        return [str(row) for row in rows if str(row)]

    kospi_top: list[str] = []
    kosdaq_top: list[str] = []
    errors: list[str] = []

    for code, target in (("1028", "kospi"), ("2203", "kosdaq")):
        try:
            data = _index_constituents(code)
            if target == "kospi":
                kospi_top = data[:200]
            else:
                kosdaq_top = data[:150]
        except Exception as exc:
            errors.append(f"{target}_index:{exc}")

    if len(kospi_top) < 200 or len(kosdaq_top) < 150:
        try:
            if len(kospi_top) < 200:
                kospi_top = _top_by_market_cap(stock.get_market_cap_by_ticker(date, market="KOSPI"), 200)
        except Exception as exc:
            errors.append(f"kospi_market_cap:{exc}")

        try:
            if len(kosdaq_top) < 150:
                kosdaq_top = _top_by_market_cap(stock.get_market_cap_by_ticker(date, market="KOSDAQ"), 150)
        except Exception as exc:
            errors.append(f"kosdaq_market_cap:{exc}")

    if len(kospi_top) < 200 or len(kosdaq_top) < 150:
        try:
            if len(kospi_top) < 200:
                kospi_top = _top_by_trading_value(stock.get_market_ohlcv_by_ticker(date, market="KOSPI"), 200)
        except Exception as exc:
            errors.append(f"kospi_trading_value:{exc}")

        try:
            if len(kosdaq_top) < 150:
                kosdaq_top = _top_by_trading_value(stock.get_market_ohlcv_by_ticker(date, market="KOSDAQ"), 150)
        except Exception as exc:
            errors.append(f"kosdaq_trading_value:{exc}")

    symbols = [f"{ticker}.KS" for ticker in kospi_top] + [f"{ticker}.KQ" for ticker in kosdaq_top]
    deduped = list(dict.fromkeys(symbols))
    if len(deduped) < 350:
        detail = "; ".join(errors) if errors else "unknown"
        raise RuntimeError(f"Universe build incomplete: expected 350, got {len(deduped)} ({detail})")
    return deduped[:350]


def save_universe_csv(path: str, symbols: list[str]):
    pd.DataFrame({"Symbol": symbols}).to_csv(path, index=False)


def get_symbol_name_map(symbols: list[str]) -> dict[str, str]:
    """Return mapping of yfinance-style Symbol -> Korean company name."""
    try:
        stock = _import_pykrx_stock()
    except Exception:
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
