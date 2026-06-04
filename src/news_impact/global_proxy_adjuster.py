from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.news_impact.global_proxy_loader import GlobalMarketProxy


@dataclass(frozen=True)
class GlobalProxyAdjustmentResult:
    global_proxy_adjustment: float
    proxy_count: int
    quality_flags: tuple[str, ...]


def adjust_global_proxy(
    korean_sector: str,
    proxies: Iterable[GlobalMarketProxy],
    proxy_returns: dict[str, float],
    sector_sensitivity: float = 1.0,
    lag_weight: float = 1.0,
    adjustment_cap: float = 15.0,
) -> GlobalProxyAdjustmentResult:
    adjustments: list[float] = []
    quality_flags: list[str] = []
    matched_count = 0

    for proxy in proxies:
        if proxy.korean_sector != korean_sector:
            continue
        matched_count += 1
        if proxy.proxy_symbol not in proxy_returns:
            _append_once(quality_flags, "missing_global_proxy_price")
            continue
        adjustment = (
            proxy_returns[proxy.proxy_symbol]
            * proxy.weight
            * sector_sensitivity
            * lag_weight
            * _direction_sign(proxy)
            * 100
        )
        adjustment = _clamp(
            adjustment,
            -proxy.max_adjustment_abs,
            proxy.max_adjustment_abs,
        )
        adjustments.append(adjustment)

    if _has_conflicting_signs(adjustments):
        _append_once(quality_flags, "conflicting_global_proxy_signals")

    return GlobalProxyAdjustmentResult(
        global_proxy_adjustment=_round_score(
            _clamp(sum(adjustments), -adjustment_cap, adjustment_cap)
        ),
        proxy_count=matched_count,
        quality_flags=tuple(quality_flags),
    )


def _direction_sign(proxy: GlobalMarketProxy) -> int:
    if proxy.direction_transform == "inverse":
        return -1
    return 1


def _has_conflicting_signs(values: list[float]) -> bool:
    positives = [value for value in values if value > 0]
    negatives = [value for value in values if value < 0]
    return bool(positives and negatives)


def _append_once(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _round_score(value: float) -> float:
    return round(value, 6)
