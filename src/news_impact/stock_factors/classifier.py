from __future__ import annotations

from src.news_impact.stock_factors.factor_taxonomy import FACTOR_KEYWORDS, FACTOR_LABELS, FACTOR_ORDER
from src.news_impact.stock_factors.freshness import detect_freshness_items, freshness_required
from src.news_impact.stock_factors.impact_rules import (
    build_impact_path,
    determine_affected_markets,
    determine_direction,
    determine_horizons,
    determine_sector_impacts,
)
from src.news_impact.stock_factors.output_schema import Direction, FactorCode, StockFactorAnalysis


_BASE_CAUTIONS = (
    "투자 조언 아님",
    "확정되지 않은 전망을 사실처럼 단정하지 말 것",
)


def analyze_stock_factors(text: str) -> StockFactorAnalysis:
    factors = classify_factors(text)
    direction = determine_direction(text, factors)
    horizons = determine_horizons(text, factors)
    affected_markets = determine_affected_markets(factors, text)
    sector_impacts = determine_sector_impacts(text, factors, direction)
    impact_path = build_impact_path(factors, direction)
    freshness_items = detect_freshness_items(text)
    needs_freshness = freshness_required(text) or bool(freshness_items)
    cautions = _build_cautions(text, factors, needs_freshness)

    return StockFactorAnalysis(
        summary=_build_summary(factors, direction),
        factors=factors,
        direction=direction,
        horizons=horizons,
        affected_markets=affected_markets,
        sector_impacts=sector_impacts,
        impact_path=impact_path,
        freshness_required=needs_freshness,
        freshness_items=freshness_items,
        cautions=cautions,
        confidence=_confidence(factors, direction),
    )


def classify_factors(text: str) -> tuple[FactorCode, ...]:
    normalized = text.casefold()
    factors: list[FactorCode] = []
    for factor in FACTOR_ORDER:
        if any(keyword.casefold() in normalized for keyword in FACTOR_KEYWORDS[factor]):
            factors.append(factor)

    if "GEO_OIL" in factors and "BOK_CREDIT" not in factors and any(needle in text for needle in ("유가 급등", "수입물가", "CPI", "물가")):
        factors.append("BOK_CREDIT")
    if "ACCESS" in factors and "MSCI" in text and "GOVERNANCE" not in factors and any(needle in text for needle in ("공매도", "상법", "밸류업")):
        factors.append("GOVERNANCE")

    return tuple(factor for factor in FACTOR_ORDER if factor in factors)


def _build_summary(factors: tuple[FactorCode, ...], direction: Direction) -> str:
    if not factors:
        return "명확한 한국 주식시장 요인을 식별하기 어려운 입력"
    labels = ", ".join(FACTOR_LABELS[factor] for factor in factors[:3])
    suffix = {
        "positive": "중심의 긍정 이벤트",
        "negative": "중심의 부정 이벤트",
        "mixed": "영향이 엇갈릴 수 있는 혼합 이벤트",
        "neutral": "방향성이 제한적인 이벤트",
        "unknown": "방향성을 판단하기 어려운 이벤트",
    }[direction]
    return f"{labels} {suffix}"


def _build_cautions(text: str, factors: tuple[FactorCode, ...], needs_freshness: bool) -> tuple[str, ...]:
    cautions = list(_BASE_CAUTIONS)
    if needs_freshness:
        cautions.append("최신 수치·정책 일정은 별도 확인 필요")
    if "GEO_OIL" in factors or any(needle in text for needle in ("중동", "호르무즈", "방산", "조선")):
        cautions.append("방산·조선 중동 수혜는 조건부로만 해석하고 확정하지 말 것")
    if "ACCESS" in factors and "MSCI" in text:
        cautions.append("MSCI 편입·워치리스트는 공식 발표 전 확정 금지")
    if "FLOW" in factors:
        cautions.append("외국인 순매수 금액만으로 결론 금지")
    if "SEMI" in factors:
        cautions.append("KOSPI 대형주 집중도는 계산 방식에 따라 달라질 수 있음")
    return tuple(dict.fromkeys(cautions))


def _confidence(factors: tuple[FactorCode, ...], direction: Direction) -> str:
    if not factors or direction == "unknown":
        return "low"
    if len(factors) >= 2:
        return "medium"
    return "medium"
