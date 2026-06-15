from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd

warnings.filterwarnings("ignore", module=r"yfinance(\..*)?")

_LOGGER = logging.getLogger(__name__)
_YF = None
MAX_DOWNLOAD_ATTEMPTS = 3
_sleep = time.sleep
SAME_DATE_EXTERNAL_SYMBOLS = frozenset({"^KS11", "^KQ11"})


def _get_yfinance():
    global _YF
    if _YF is None:
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yfinance is required for external market features. "
                "Install market extras or disable external features."
            ) from exc
        # Silence yfinance chatter once after the optional import.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        _YF = yf
    return _YF


def _safe_download(symbol: str, start: str, end: str | None) -> pd.Series:
    """Download close series without leaking provider noise to console."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            yf = _get_yfinance()
            df = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    # Find the MultiIndex level that contains price-type names rather
                    # than blindly taking level 0, which may be the ticker level in
                    # newer yfinance versions and would produce duplicate labels.
                    _price_cols = frozenset({"Open", "High", "Low", "Close", "Adj Close", "Volume"})
                    extracted = None
                    for _lvl in range(df.columns.nlevels):
                        _vals = df.columns.get_level_values(_lvl)
                        if _price_cols.intersection(_vals):
                            extracted = _vals
                            break
                    df.columns = extracted if extracted is not None else df.columns.get_level_values(0)
                if "Close" in df.columns:
                    s = df["Close"].copy()
                    s.index = pd.to_datetime(s.index)
                    return s
        except Exception as exc:
            last_exc = exc
        if attempt < MAX_DOWNLOAD_ATTEMPTS:
            _sleep(2 ** (attempt - 1))

    if last_exc is not None:
        _LOGGER.warning(
            "yfinance external download failed for %s after %d attempts: %s: %s",
            symbol,
            MAX_DOWNLOAD_ATTEMPTS,
            type(last_exc).__name__,
            last_exc,
        )
    return pd.Series(dtype=float)



def _symbol_candidates(symbol: str) -> tuple[str, list[str]]:
    fallback = {
        "^SOX": ("sox", ["^SOX", "SOXX"]),
        "^VIX": ("vix", ["^VIX", "VIXY"]),
        "^KS11": ("ks11", ["^KS11", "EWY"]),
        "^KQ11": ("kq11", ["^KQ11", "KORU"]),
        "^IXIC": ("ixic", ["^IXIC", "QQQ"]),
        "NQ=F": ("nq_f", ["NQ=F", "QQQ"]),
        "^GSPC": ("gspc", ["^GSPC", "SPY"]),
        "KRW=X": ("krw_x", ["KRW=X", "USDKRW=X"]),
        "^TNX": ("tnx", ["^TNX", "IEF"]),
    }
    if symbol in fallback:
        return fallback[symbol]

    alias = symbol.replace("^", "").replace("=", "_").replace("-", "_").lower()
    return alias, [symbol]



def _series_to_external_frame(series_or_df: pd.Series | pd.DataFrame, col_base: str) -> pd.DataFrame:
    if isinstance(series_or_df, pd.Series):
        frame = series_or_df.rename(f"{col_base}_close").reset_index()
    else:
        frame = pd.DataFrame(series_or_df).reset_index()

    if frame.empty or frame.shape[1] < 2:
        return pd.DataFrame(columns=["Date", f"{col_base}_close"])

    date_col = frame.columns[0]
    value_col = None
    for candidate in frame.columns[1:]:
        if pd.api.types.is_numeric_dtype(frame[candidate]):
            value_col = candidate
            break
    if value_col is None:
        value_col = frame.columns[-1]

    out = frame[[date_col, value_col]].copy()
    out.columns = ["Date", f"{col_base}_close"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    # yfinance occasionally emits duplicate Date rows; keep the latest so the
    # downstream `ext.merge(frame, on="Date")` does not fan out and trigger
    # `InvalidIndexError: Reindexing only valid with uniquely valued Index`.
    out = out.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return out



def _download_external_symbol(sym: str, start: str, end: str) -> tuple[str, dict, pd.DataFrame | None]:
    col_base, candidates = _symbol_candidates(sym)

    s = pd.Series(dtype=float)
    used_candidate = None
    for candidate in candidates:
        s = _safe_download(candidate, start=start, end=end)
        if not s.empty:
            used_candidate = candidate
            break

    if s.empty:
        return sym, {"symbol": sym, "status": "failed", "used": None}, None

    e = _series_to_external_frame(s, col_base)
    if e.empty:
        return sym, {"symbol": sym, "status": "failed", "used": None}, None
    e[f"{col_base}_ret_1d"] = e[f"{col_base}_close"].pct_change()
    e[f"{col_base}_ret_5d"] = e[f"{col_base}_close"].pct_change(5)
    e[f"{col_base}_vol_20"] = e[f"{col_base}_ret_1d"].rolling(20).std()
    return sym, {"symbol": sym, "status": "ok", "used": used_candidate}, e


def _apply_availability_lag(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if symbol in SAME_DATE_EXTERNAL_SYMBOLS:
        return frame
    out = frame.copy()
    value_columns = [column for column in out.columns if column != "Date"]
    out[value_columns] = out[value_columns].shift(1)
    return out



def add_external_market_features_with_coverage(df: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    coverage = {
        "requested": len(symbols),
        "successful": 0,
        "failed": 0,
        "fallback_used": 0,
        "details": [],
    }

    if out.empty or not symbols:
        return out, coverage

    start = out["Date"].min().strftime("%Y-%m-%d")
    end = (out["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base_dates = pd.Series(sorted(out["Date"].unique()), name="Date")
    ext = pd.DataFrame({"Date": pd.to_datetime(base_dates)})

    worker_count = min(4, len(symbols))
    if worker_count <= 1:
        downloads = [_download_external_symbol(sym, start, end) for sym in symbols]
    else:
        download_one = partial(_download_external_symbol, start=start, end=end)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            downloads = list(executor.map(download_one, symbols))

    for sym, detail, frame in downloads:
        if detail["status"] == "failed" or frame is None:
            coverage["failed"] += 1
            coverage["details"].append(detail)
            continue

        coverage["successful"] += 1
        if detail["used"] != sym:
            coverage["fallback_used"] += 1
        coverage["details"].append(detail)
        ext = ext.merge(_apply_availability_lag(frame, detail["used"]), on="Date", how="left")

    if coverage["successful"] == 0:
        return out, coverage

    ext = ext.sort_values("Date").ffill()
    out = out.merge(ext, on="Date", how="left")
    return out, coverage



def add_external_market_features(df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    out, _ = add_external_market_features_with_coverage(df, symbols)
    return out
