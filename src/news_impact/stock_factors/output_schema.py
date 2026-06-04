from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Direction = Literal["positive", "negative", "mixed", "neutral", "unknown"]
Horizon = Literal["short_term", "medium_term", "long_term"]
Confidence = Literal["low", "medium", "high"]
FactorCode = Literal[
    "US_RISK",
    "US_RATE",
    "FX_KRW",
    "SEMI",
    "EXPORT",
    "CHINA",
    "FLOW",
    "BOK_CREDIT",
    "GOVERNANCE",
    "ACCESS",
    "GEO_OIL",
]


@dataclass(frozen=True)
class SectorImpact:
    sector: str
    direction: Direction
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector": self.sector,
            "direction": self.direction,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StockFactorAnalysis:
    summary: str
    factors: tuple[FactorCode, ...]
    direction: Direction
    horizons: tuple[Horizon, ...]
    affected_markets: tuple[str, ...]
    sector_impacts: tuple[SectorImpact, ...]
    impact_path: tuple[str, ...]
    freshness_required: bool
    freshness_items: tuple[str, ...]
    cautions: tuple[str, ...]
    confidence: Confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "factors": list(self.factors),
            "direction": self.direction,
            "horizons": list(self.horizons),
            "affectedMarkets": list(self.affected_markets),
            "sectorImpacts": [impact.to_dict() for impact in self.sector_impacts],
            "impactPath": list(self.impact_path),
            "freshnessRequired": self.freshness_required,
            "freshnessItems": list(self.freshness_items),
            "cautions": list(self.cautions),
            "confidence": self.confidence,
        }
