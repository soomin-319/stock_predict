from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.fetch_real_data import fetch_real_ohlcv
from src.data.krx_universe import get_symbol_name_map
from src.recommendation.close_betting import (
    CloseBettingRecommendation,
    add_technical_indicators,
    add_trade_value_rank,
    latest_rows,
    recommendations_from_candidates,
    score_candidates,
    select_close_betting_candidates,
)

_LOGGER = logging.getLogger(__name__)
DEFAULT_UNIVERSE_CSV = Path(__file__).resolve().parents[2] / "data" / "default_universe_kospi50_kosdaq50.csv"

SymbolsProvider = Callable[[], pd.DataFrame]
OhlcvFetcher = Callable[[list[str], str, str | None], pd.DataFrame]
TodayProvider = Callable[[], date]


class RealTimeCloseBettingRecommendationService:
    def __init__(
        self,
        symbols_provider: SymbolsProvider | None = None,
        ohlcv_fetcher: OhlcvFetcher | None = None,
        today_provider: TodayProvider | None = None,
        universe_csv: str | Path | None = None,
        lookback_days: int = 420,
        universe_limit: int = 200,
        first_buy_ratio: float = 0.6,
        top_trade_value_count: int = 20,
    ):
        self.symbols_provider = symbols_provider or (lambda: self._load_default_symbols(universe_csv, universe_limit))
        self.ohlcv_fetcher = ohlcv_fetcher or fetch_real_ohlcv
        self.today_provider = today_provider or date.today
        self.lookback_days = lookback_days
        self.first_buy_ratio = first_buy_ratio
        self.top_trade_value_count = top_trade_value_count

    def get_recommendations(
        self,
        top_n: int | None = 3,
        min_final_score: int | None = None,
    ) -> list[CloseBettingRecommendation]:
        today = self.today_provider()
        start = (today - timedelta(days=self.lookback_days)).isoformat()
        end = (today + timedelta(days=1)).isoformat()
        symbols_df = self._normalize_symbols(self.symbols_provider())
        symbols = symbols_df["Symbol"].dropna().astype(str).drop_duplicates().tolist()
        if not symbols:
            return []

        try:
            raw = self.ohlcv_fetcher(symbols, start, end)
        except RuntimeError as exc:
            _LOGGER.warning("live OHLCV fetch failed for recommendation scan: %s", exc)
            return []
        standard = self._standardize_ohlcv(raw, symbols_df)
        if standard.empty:
            return []

        ranked = add_trade_value_rank(standard)
        top = ranked[ranked["trade_value_rank"].astype("Int64") <= self.top_trade_value_count].copy()
        technical = add_technical_indicators(top)
        scored = score_candidates(latest_rows(technical), top_trade_value_count=self.top_trade_value_count)
        candidates = select_close_betting_candidates(
            scored,
            top_n=top_n,
            first_buy_ratio=self.first_buy_ratio,
            min_final_score=min_final_score,
        )
        return recommendations_from_candidates(candidates)

    def _load_default_symbols(self, universe_csv: str | Path | None, universe_limit: int) -> pd.DataFrame:
        if universe_csv is None:
            df = pd.read_csv(DEFAULT_UNIVERSE_CSV) if DEFAULT_UNIVERSE_CSV.exists() else pd.DataFrame()
        else:
            df = pd.read_csv(Path(universe_csv))
        if df is None or df.empty or "Symbol" not in df.columns:
            return pd.DataFrame(columns=["Symbol", "Name", "Market", "Bucket"])
        if "Market" in df.columns:
            df = df[df["Market"].astype(str).str.upper() == "KOSPI"].copy()
        if universe_limit > 0:
            df = df.head(universe_limit)
        if df.empty:
            return pd.DataFrame(columns=["Symbol", "Name", "Market", "Bucket"])
        names = get_symbol_name_map(df["Symbol"].astype(str).tolist()) if "Symbol" in df.columns else {}
        if "Name" not in df.columns:
            df["Name"] = df["Symbol"]
        df["Name"] = df["Name"].astype(str)
        mapped_names = df["Symbol"].astype(str).map(names)
        has_real_mapped_name = mapped_names.notna() & (mapped_names.astype(str) != df["Symbol"].astype(str))
        df["Name"] = mapped_names.where(has_real_mapped_name, df["Name"])
        return df

    def _normalize_symbols(self, symbols_df: pd.DataFrame) -> pd.DataFrame:
        if symbols_df is None or symbols_df.empty:
            return pd.DataFrame(columns=["Symbol", "Ticker", "Name", "Market"])
        df = symbols_df.copy()
        if "symbol" in df.columns and "Symbol" not in df.columns:
            df = df.rename(columns={"symbol": "Symbol"})
        if "name" in df.columns and "Name" not in df.columns:
            df = df.rename(columns={"name": "Name"})
        if "market" in df.columns and "Market" not in df.columns:
            df = df.rename(columns={"market": "Market"})
        if "Symbol" not in df.columns:
            raise ValueError("symbols dataframe must include Symbol column")
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["Ticker"] = df["Symbol"].str.split(".").str[0].str.zfill(6)
        if "Name" not in df.columns:
            df["Name"] = df["Ticker"]
        df["Name"] = df["Name"].astype(str).str.strip()
        if "Market" not in df.columns:
            df["Market"] = "KOSPI"
        return df[["Symbol", "Ticker", "Name", "Market"]].drop_duplicates(subset=["Symbol"])

    def _standardize_ohlcv(self, raw: pd.DataFrame, symbols_df: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        rename_map = {
            "Date": "date",
            "Symbol": "symbol_raw",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        required = {"date", "symbol_raw", "open", "high", "low", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")

        symbol_meta = symbols_df.set_index("Symbol")
        df["symbol_raw"] = df["symbol_raw"].astype(str).str.strip().str.upper()
        df["symbol"] = df["symbol_raw"].str.split(".").str[0].str.zfill(6)
        name_by_symbol = symbol_meta["Name"].to_dict()
        name_by_ticker = symbols_df.set_index("Ticker")["Name"].to_dict()
        df["name"] = df["symbol_raw"].map(name_by_symbol).fillna(df["symbol"].map(name_by_ticker)).fillna(df["symbol"])
        df["market"] = "KOSPI"
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)].copy()
        df["trade_value"] = df["close"] * df["volume"]
        df["trade_value_source"] = "close_volume_estimated"
        df["signal_timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        df["data_cutoff_timestamp"] = str(df["date"].max())
        df["execution_assumption"] = "종가 확정 후 다음 거래일 진입"
        df["price_basis"] = "unadjusted"
        df["is_close_confirmed"] = True
        cols = [
            "symbol",
            "name",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_value",
            "trade_value_source",
            "market",
            "signal_timestamp",
            "data_cutoff_timestamp",
            "execution_assumption",
            "price_basis",
            "is_close_confirmed",
        ]
        return df[cols].drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
