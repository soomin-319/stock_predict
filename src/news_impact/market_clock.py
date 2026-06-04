from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Protocol

from src.news_impact.schema import MarketSession


KST = timezone(timedelta(hours=9), "Asia/Seoul")
REGULAR_OPEN = time(9, 0)
REGULAR_CLOSE = time(15, 30)


class SchemaTimeError(ValueError):
    """Raised when event time cannot be safely used for market-session logic."""


class KrxTradingCalendar(Protocol):
    def is_trading_day(self, day: date) -> bool:
        ...

    def next_trading_day(self, day: date) -> date:
        ...


@dataclass(frozen=True)
class WeekendKrxCalendar:
    holidays: set[date] | frozenset[date] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "holidays", frozenset(self.holidays or ()))

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def next_trading_day(self, day: date) -> date:
        candidate = day + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate


@dataclass(frozen=True)
class NormalizedSignalTime:
    signal_at_kst: datetime
    market_session: MarketSession
    effective_trading_day: date


def classify_market_session(
    signal_at: datetime,
    calendar: KrxTradingCalendar,
) -> MarketSession:
    signal_at_kst = _to_kst(signal_at)
    signal_day = signal_at_kst.date()
    if not calendar.is_trading_day(signal_day):
        return "holiday"
    signal_time = signal_at_kst.time()
    if signal_time < REGULAR_OPEN:
        return "pre_market"
    if signal_time <= REGULAR_CLOSE:
        return "regular"
    return "after_market"


def normalize_signal_time(
    signal_at: datetime,
    calendar: KrxTradingCalendar,
) -> NormalizedSignalTime:
    signal_at_kst = _to_kst(signal_at)
    market_session = classify_market_session(signal_at_kst, calendar)
    signal_day = signal_at_kst.date()
    if market_session in {"after_market", "holiday"}:
        effective_trading_day = calendar.next_trading_day(signal_day)
    else:
        effective_trading_day = signal_day
    return NormalizedSignalTime(
        signal_at_kst=signal_at_kst,
        market_session=market_session,
        effective_trading_day=effective_trading_day,
    )


def _to_kst(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SchemaTimeError("signal_at must be timezone-aware")
    return value.astimezone(KST)
