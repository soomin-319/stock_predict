from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import sqrt


@dataclass(frozen=True)
class BacktestSignal:
    ticker: str
    score: float
    signal_day: date
    effective_trading_day: date
    market_session: str
    tradeability_status: str
    sector: str
    market_cap_bucket: str = "unknown"
    event_type: str = "unknown"


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    trading_day: date
    open: float
    close: float
    market_return: float = 0.0
    sector_return: float = 0.0


@dataclass(frozen=True)
class BacktestObservation:
    ticker: str
    score: float
    signal_day: date
    entry_day: date
    raw_return: float
    after_cost_return: float
    market_excess_return: float
    sector_excess_return: float
    quality_flags: tuple[str, ...]
    sector: str = "unknown"
    market_cap_bucket: str = "unknown"
    event_type: str = "unknown"


@dataclass(frozen=True)
class BacktestMetrics:
    sample_size: int
    ic: float
    rank_ic: float
    hit_ratio: float
    top_bottom_spread: float


@dataclass(frozen=True)
class ScoreVariantComparison:
    baseline: BacktestMetrics
    adjusted: BacktestMetrics
    ic_delta: float
    rank_ic_delta: float
    top_bottom_spread_delta: float


class ExperimentLog:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def record(
        self,
        experiment_id: str,
        description: str,
        accepted: bool,
        metrics: dict[str, float],
    ) -> None:
        self.entries.append(
            {
                "experiment_id": experiment_id,
                "description": description,
                "accepted": accepted,
                "metrics": dict(metrics),
            }
        )


def match_signal_returns(
    signals: list[BacktestSignal] | tuple[BacktestSignal, ...],
    prices: list[PriceBar] | tuple[PriceBar, ...],
    round_trip_cost_bps: float = 26.0,
    include_missing: bool = False,
) -> list[BacktestObservation]:
    price_by_key = {(price.ticker, price.trading_day): price for price in prices}
    observations: list[BacktestObservation] = []
    for signal in signals:
        if signal.tradeability_status in {"halted", "limit_locked"}:
            continue
        entry_day = signal.effective_trading_day
        price = price_by_key.get((signal.ticker, entry_day))
        quality_flags = _session_flags(signal)
        if price is None:
            if include_missing:
                observations.append(
                    BacktestObservation(
                        ticker=signal.ticker,
                        score=signal.score,
                        signal_day=signal.signal_day,
                        entry_day=entry_day,
                        raw_return=0.0,
                        after_cost_return=0.0,
                        market_excess_return=0.0,
                        sector_excess_return=0.0,
                        quality_flags=quality_flags + ("missing_intraday_price",),
                        sector=signal.sector,
                        market_cap_bucket=signal.market_cap_bucket,
                        event_type=signal.event_type,
                    )
                )
            continue
        raw_return = price.close / price.open - 1
        cost = round_trip_cost_bps / 10_000
        observations.append(
            BacktestObservation(
                ticker=signal.ticker,
                score=signal.score,
                signal_day=signal.signal_day,
                entry_day=entry_day,
                raw_return=_round(raw_return),
                after_cost_return=_round(raw_return - cost),
                market_excess_return=_round(raw_return - price.market_return),
                sector_excess_return=_round(raw_return - price.sector_return),
                quality_flags=quality_flags,
                sector=signal.sector,
                market_cap_bucket=signal.market_cap_bucket,
                event_type=signal.event_type,
            )
        )
    return observations


def calculate_metrics(
    observations: list[BacktestObservation] | tuple[BacktestObservation, ...],
    basket_size: int = 5,
) -> BacktestMetrics:
    usable = [item for item in observations if "missing_intraday_price" not in item.quality_flags]
    scores = [item.score for item in usable]
    returns = [item.after_cost_return for item in usable]
    return BacktestMetrics(
        sample_size=len(usable),
        ic=_round(_pearson(scores, returns)),
        rank_ic=_round(_pearson(_ranks(scores), _ranks(returns))),
        hit_ratio=_round(_hit_ratio(usable)),
        top_bottom_spread=_round(_top_bottom_spread(usable, basket_size)),
    )


def summarize_by_bucket(
    observations: list[BacktestObservation] | tuple[BacktestObservation, ...],
    bucket: str,
) -> dict[str, BacktestMetrics]:
    grouped: dict[str, list[BacktestObservation]] = {}
    for observation in observations:
        key = str(getattr(observation, bucket))
        grouped.setdefault(key, []).append(observation)
    return {key: calculate_metrics(values, basket_size=1) for key, values in grouped.items()}


def compare_score_variants(
    baseline_observations: list[BacktestObservation] | tuple[BacktestObservation, ...],
    adjusted_observations: list[BacktestObservation] | tuple[BacktestObservation, ...],
    basket_size: int = 5,
) -> ScoreVariantComparison:
    baseline = calculate_metrics(baseline_observations, basket_size=basket_size)
    adjusted = calculate_metrics(adjusted_observations, basket_size=basket_size)
    return ScoreVariantComparison(
        baseline=baseline,
        adjusted=adjusted,
        ic_delta=_round(adjusted.ic - baseline.ic),
        rank_ic_delta=_round(adjusted.rank_ic - baseline.rank_ic),
        top_bottom_spread_delta=_round(adjusted.top_bottom_spread - baseline.top_bottom_spread),
    )


def _session_flags(signal: BacktestSignal) -> tuple[str, ...]:
    if signal.market_session == "after_market" and signal.effective_trading_day != signal.signal_day:
        return ("after_market_shifted",)
    if signal.market_session == "holiday" and signal.effective_trading_day != signal.signal_day:
        return ("holiday_shifted",)
    return ()


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_denominator = sqrt(sum((x - left_mean) ** 2 for x in left))
    right_denominator = sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_denominator == 0 or right_denominator == 0:
        return 0.0
    return numerator / (left_denominator * right_denominator)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    for rank, (_, index) in enumerate(ordered, start=1):
        ranks[index] = float(rank)
    return ranks


def _hit_ratio(observations: list[BacktestObservation]) -> float:
    if not observations:
        return 0.0
    hits = 0
    for item in observations:
        if item.score > 0 and item.after_cost_return > 0:
            hits += 1
        elif item.score < 0 and item.after_cost_return < 0:
            hits += 1
        elif item.score == 0:
            hits += 1
    return hits / len(observations)


def _top_bottom_spread(observations: list[BacktestObservation], basket_size: int) -> float:
    if not observations:
        return 0.0
    ordered = sorted(observations, key=lambda item: item.score, reverse=True)
    top = ordered[:basket_size]
    bottom = ordered[-basket_size:]
    top_return = sum(item.after_cost_return for item in top) / len(top)
    bottom_return = sum(item.after_cost_return for item in bottom) / len(bottom)
    return top_return - bottom_return


def _round(value: float) -> float:
    return round(value, 6)
