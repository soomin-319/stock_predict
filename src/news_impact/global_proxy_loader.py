from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


class GlobalProxyValidationError(ValueError):
    """Raised when global proxy CSV data violates docs/GLOBAL_MARKET_PROXY_SPEC.md."""


@dataclass(frozen=True)
class GlobalMarketProxy:
    korean_sector: str
    proxy_symbol: str
    proxy_name: str
    proxy_type: str
    relation_type: str
    weight: float
    lag_policy: str
    direction_transform: str
    proxy_currency: str
    proxy_timezone: str
    max_adjustment_abs: float
    notes: str

    def __post_init__(self) -> None:
        _require_non_empty("korean_sector", self.korean_sector)
        _require_non_empty("proxy_symbol", self.proxy_symbol)
        _require_allowed("proxy_type", self.proxy_type, _PROXY_TYPES)
        _require_allowed("relation_type", self.relation_type, _RELATION_TYPES)
        _require_allowed("lag_policy", self.lag_policy, _LAG_POLICIES)
        _require_allowed("direction_transform", self.direction_transform, _DIRECTION_TRANSFORMS)
        _require_allowed("proxy_currency", self.proxy_currency, _CURRENCIES)
        _require_non_empty("proxy_timezone", self.proxy_timezone)
        if not 0.0 <= self.weight <= 1.2:
            raise GlobalProxyValidationError("weight must be between 0.0 and 1.2")
        if not 0.0 <= self.max_adjustment_abs <= 15.0:
            raise GlobalProxyValidationError("max_adjustment_abs must be between 0.0 and 15.0")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "korean_sector": self.korean_sector,
            "proxy_symbol": self.proxy_symbol,
            "proxy_name": self.proxy_name,
            "proxy_type": self.proxy_type,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "lag_policy": self.lag_policy,
            "direction_transform": self.direction_transform,
            "proxy_currency": self.proxy_currency,
            "proxy_timezone": self.proxy_timezone,
            "max_adjustment_abs": self.max_adjustment_abs,
            "notes": self.notes,
        }


def load_global_proxies(path: str | Path) -> list[GlobalMarketProxy]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [_proxy_from_row(row, row_number=index + 2) for index, row in enumerate(rows)]


def _proxy_from_row(row: dict[str, str], row_number: int) -> GlobalMarketProxy:
    try:
        return GlobalMarketProxy(
            korean_sector=row["korean_sector"],
            proxy_symbol=row["proxy_symbol"],
            proxy_name=row["proxy_name"],
            proxy_type=row["proxy_type"],
            relation_type=row["relation_type"],
            weight=float(row["weight"]),
            lag_policy=row["lag_policy"],
            direction_transform=row["direction_transform"],
            proxy_currency=row["proxy_currency"],
            proxy_timezone=row["proxy_timezone"],
            max_adjustment_abs=float(row["max_adjustment_abs"]),
            notes=row["notes"],
        )
    except KeyError as error:
        raise GlobalProxyValidationError(
            f"row {row_number}: missing column {error.args[0]}"
        ) from error
    except ValueError as error:
        raise GlobalProxyValidationError(f"row {row_number}: {error}") from error


def _require_non_empty(field_name: str, value: str) -> None:
    if not value:
        raise GlobalProxyValidationError(f"{field_name} must not be empty")


def _require_allowed(field_name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise GlobalProxyValidationError(f"{field_name} must be one of: {allowed_text}")


_PROXY_TYPES = {"index", "etf", "stock", "commodity", "fx"}
_RELATION_TYPES = {"sector_proxy", "customer", "competitor", "input_cost", "macro", "adr"}
_LAG_POLICIES = {"overnight", "same_day", "previous_close"}
_DIRECTION_TRANSFORMS = {"normal", "inverse"}
_CURRENCIES = {"USD", "KRW", "JPY", "CNY", "EUR", "none"}
