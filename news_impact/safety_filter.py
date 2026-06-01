from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketSafetyInput:
    ticker: str
    as_of: datetime
    trading_halt: bool = False
    management_issue: bool = False
    delisting_risk: bool = False
    investment_warning: bool = False
    limit_up: bool = False
    limit_down: bool = False
    volatility_interruption: bool = False
    wide_spread: bool = False
    avg_trading_value_20d: float | None = None


@dataclass(frozen=True)
class SafetyResult:
    news_disclosure_score: float
    tradeability_status: str
    risk_flags: tuple[str, ...]
    review_checklist: tuple[str, ...]


def apply_safety_filter(
    news_disclosure_score: float,
    market_input: MarketSafetyInput,
    existing_risk_flags: tuple[str, ...] | list[str] = (),
) -> SafetyResult:
    flags = list(existing_risk_flags)
    checklist = ["not_investment_advice", "verify_tradeability_status"]

    _append_hard_filter_flags(flags, market_input)
    _append_liquidity_flags(flags, market_input)

    tradeability_status = _tradeability_status(flags, market_input)
    if tradeability_status == "unknown":
        _append_once(flags, "tradeability_unknown")

    return SafetyResult(
        news_disclosure_score=news_disclosure_score,
        tradeability_status=tradeability_status,
        risk_flags=tuple(flags),
        review_checklist=tuple(checklist),
    )


def _append_hard_filter_flags(flags: list[str], market_input: MarketSafetyInput) -> None:
    if market_input.trading_halt:
        _append_once(flags, "trading_halt")
    if market_input.management_issue:
        _append_once(flags, "management_issue")
    if market_input.delisting_risk:
        _append_once(flags, "delisting_risk")
    if market_input.investment_warning:
        _append_once(flags, "investment_warning")
    if market_input.limit_up:
        _append_once(flags, "limit_up")
    if market_input.limit_down:
        _append_once(flags, "limit_down")
    if market_input.volatility_interruption:
        _append_once(flags, "volatility_interruption")
    if market_input.wide_spread:
        _append_once(flags, "wide_spread_risk")


def _append_liquidity_flags(flags: list[str], market_input: MarketSafetyInput) -> None:
    if market_input.avg_trading_value_20d is None:
        return
    if market_input.avg_trading_value_20d < 100_000_000:
        _append_once(flags, "very_low_liquidity")
    elif market_input.avg_trading_value_20d < 1_000_000_000:
        _append_once(flags, "low_liquidity")


def _tradeability_status(flags: list[str], market_input: MarketSafetyInput) -> str:
    if "trading_halt" in flags:
        return "halted"
    if "limit_up" in flags or "limit_down" in flags:
        return "limit_locked"
    caution_flags = {
        "management_issue",
        "delisting_risk",
        "investment_warning",
        "volatility_interruption",
        "wide_spread_risk",
        "very_low_liquidity",
        "low_liquidity",
    }
    if any(flag in flags for flag in caution_flags):
        return "caution"
    if market_input.avg_trading_value_20d is None:
        return "unknown"
    return "tradable"


def _append_once(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)
