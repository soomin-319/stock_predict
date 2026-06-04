from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class GlobalProxyPrice:
    proxy_symbol: str
    price_date: date
    close: float
    collected_at: datetime


@dataclass(frozen=True)
class GlobalProxyReturnResult:
    returns: dict[str, float]
    quality_flags: tuple[str, ...]


class CachedGlobalMarketCollector:
    def __init__(self, prices: list[GlobalProxyPrice]) -> None:
        self._prices = prices

    @classmethod
    def from_csv(cls, path: str | Path) -> "CachedGlobalMarketCollector":
        with Path(path).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        return cls(
            [
                GlobalProxyPrice(
                    proxy_symbol=row["proxy_symbol"],
                    price_date=date.fromisoformat(row["price_date"]),
                    close=float(row["close"]),
                    collected_at=datetime.fromisoformat(row["collected_at"]),
                )
                for row in rows
            ]
        )

    def daily_returns(self, price_date: date) -> dict[str, float]:
        return self.daily_returns_with_flags(price_date).returns

    def daily_returns_with_flags(
        self,
        price_date: date,
        collected_cutoff: str | datetime | None = None,
    ) -> GlobalProxyReturnResult:
        cutoff = _parse_cutoff(collected_cutoff)
        by_symbol = self._prices_by_symbol(cutoff)
        returns: dict[str, float] = {}
        quality_flags: list[str] = []

        for symbol, prices in by_symbol.items():
            current = _price_on(prices, price_date)
            previous = _previous_price(prices, price_date)
            if current is None:
                _append_once(quality_flags, "missing_global_proxy_price")
                continue
            if previous is None:
                _append_once(quality_flags, "missing_previous_global_proxy_price")
                continue
            if previous.close == 0:
                _append_once(quality_flags, "invalid_previous_global_proxy_price")
                continue
            returns[symbol] = round((current.close - previous.close) / previous.close, 10)

        if cutoff is not None and self._has_blocked_future_price(price_date, cutoff):
            _append_once(quality_flags, "global_proxy_future_leakage_blocked")

        return GlobalProxyReturnResult(returns=returns, quality_flags=tuple(quality_flags))

    def _prices_by_symbol(self, cutoff: datetime | None) -> dict[str, list[GlobalProxyPrice]]:
        by_symbol: dict[str, list[GlobalProxyPrice]] = {}
        for price in self._prices:
            if cutoff is not None and price.collected_at > cutoff:
                continue
            by_symbol.setdefault(price.proxy_symbol, []).append(price)
        for prices in by_symbol.values():
            prices.sort(key=lambda item: item.price_date)
        return by_symbol

    def _has_blocked_future_price(self, price_date: date, cutoff: datetime) -> bool:
        return any(
            price.price_date == price_date and price.collected_at > cutoff
            for price in self._prices
        )


def _price_on(prices: list[GlobalProxyPrice], target_date: date) -> GlobalProxyPrice | None:
    for price in prices:
        if price.price_date == target_date:
            return price
    return None


def _previous_price(prices: list[GlobalProxyPrice], target_date: date) -> GlobalProxyPrice | None:
    previous_prices = [price for price in prices if price.price_date < target_date]
    if not previous_prices:
        return None
    return previous_prices[-1]


def _parse_cutoff(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _append_once(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)
