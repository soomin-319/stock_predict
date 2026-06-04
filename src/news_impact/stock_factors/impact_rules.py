from __future__ import annotations

from src.news_impact.stock_factors.output_schema import Direction, FactorCode, Horizon, SectorImpact


_POSITIVE_SIGNALS = (
    "상승",
    "급등",
    "하락",
    "개선",
    "확대",
    "증가",
    "호조",
    "상향",
    "강세",
    "소각",
    "배당 확대",
    "주주환원",
    "수요 전망",
)
_NEGATIVE_SIGNALS = (
    "급락",
    "악화",
    "둔화",
    "축소",
    "피크아웃",
    "순매도",
    "매도",
    "원화 약세",
    "부실",
    "반대매매",
    "관세",
    "리스크",
)
_MIXED_SIGNALS = ("원화 약세", "유가 상승", "유가가 급등", "중동", "공매도", "MSCI")

_HORIZON_RULES: dict[Horizon, tuple[str, ...]] = {
    "short_term": ("미국 증시", "SOX", "Nvidia", "엔비디아", "USD/KRW", "원/달러", "외국인 선물", "유가", "VIX"),
    "medium_term": ("월별 수출", "월간 수출", "반도체 가격", "실적 가이던스", "BOK", "Fed", "CPI", "기준금리"),
    "long_term": ("ROE", "주주환원", "MSCI", "시장 접근성", "공급망", "지배구조", "상법", "밸류업"),
}

_FACTOR_HORIZONS: dict[FactorCode, tuple[Horizon, ...]] = {
    "US_RISK": ("short_term",),
    "US_RATE": ("medium_term",),
    "FX_KRW": ("short_term",),
    "SEMI": ("medium_term",),
    "EXPORT": ("medium_term",),
    "CHINA": ("medium_term",),
    "FLOW": ("short_term",),
    "BOK_CREDIT": ("medium_term",),
    "GOVERNANCE": ("long_term",),
    "ACCESS": ("long_term",),
    "GEO_OIL": ("short_term", "medium_term"),
}


def determine_direction(text: str, factors: tuple[FactorCode, ...]) -> Direction:
    normalized = text.casefold()
    positive = _has_positive_signal(normalized, factors)
    negative = _has_negative_signal(normalized, factors)
    mixed = any(signal.casefold() in normalized for signal in _MIXED_SIGNALS)

    if _contains_any(normalized, ("순매도", "선물 매도", "현물·선물 동반 매도", "원화 급락")):
        return "negative"
    if "GEO_OIL" in factors and _contains_any(normalized, ("유가", "호르무즈", "중동")):
        return "mixed" if not _contains_any(normalized, ("완화", "안정")) else "positive"
    if "FX_KRW" in factors and _contains_any(normalized, ("환율 급등", "원/달러 환율이 급등", "원화 급락")):
        return "negative"
    if positive and negative:
        return "mixed"
    if mixed:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    if factors:
        return "neutral"
    return "unknown"


def determine_horizons(text: str, factors: tuple[FactorCode, ...]) -> tuple[Horizon, ...]:
    normalized = text.casefold()
    horizons: list[Horizon] = []
    for horizon, keywords in _HORIZON_RULES.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            horizons.append(horizon)
    for factor in factors:
        for horizon in _FACTOR_HORIZONS[factor]:
            if horizon not in horizons:
                horizons.append(horizon)
    return tuple(horizons or ("short_term",))


def determine_affected_markets(factors: tuple[FactorCode, ...], text: str) -> tuple[str, ...]:
    markets: list[str] = []
    if any(factor in factors for factor in ("US_RISK", "US_RATE", "SEMI", "EXPORT", "CHINA", "BOK_CREDIT", "GOVERNANCE", "ACCESS", "GEO_OIL")):
        markets.append("KOSPI")
    if "신용" in text or "KOSDAQ" in text or "반대매매" in text:
        markets.append("KOSDAQ")
    if any(factor in factors for factor in ("FX_KRW", "BOK_CREDIT", "GEO_OIL")):
        markets.append("KRW")
    if any(factor in factors for factor in ("SEMI", "EXPORT", "CHINA", "GOVERNANCE", "ACCESS", "GEO_OIL")):
        markets.append("sector")
    if "FLOW" in factors:
        markets.append("foreign_flow")
    return _dedupe(markets)


def determine_sector_impacts(text: str, factors: tuple[FactorCode, ...], direction: Direction) -> tuple[SectorImpact, ...]:
    impacts: list[SectorImpact] = []
    if "SEMI" in factors:
        impacts.append(
            SectorImpact(
                sector="semiconductor",
                direction="positive" if direction in ("positive", "mixed") else direction,
                reason="SOX, HBM, DRAM/NAND 또는 AI CapEx 관련 반도체 심리 변화",
            )
        )
    if "FX_KRW" in factors and "원화 약세" in text:
        impacts.append(SectorImpact(sector="auto", direction="mixed", reason="원화 약세는 수출 환산이익과 외국인 수급에 엇갈린 영향"))
    if "GEO_OIL" in factors:
        impacts.extend(
            (
                SectorImpact(sector="manufacturing", direction="negative", reason="에너지 비용과 수입물가 부담"),
                SectorImpact(sector="defense", direction="mixed", reason="지정학 테마 가능성은 있으나 확정 수혜 아님"),
            )
        )
    if "GOVERNANCE" in factors:
        impacts.append(SectorImpact(sector="bank", direction="positive", reason="배당·자사주 소각·주주환원 기대"))
    if "CHINA" in factors:
        impacts.append(SectorImpact(sector="battery", direction="negative", reason="중국 경쟁과 G2 공급망 리스크"))
    return tuple(impacts)


def build_impact_path(factors: tuple[FactorCode, ...], direction: Direction) -> tuple[str, ...]:
    path: list[str] = []
    if "US_RISK" in factors:
        path.append("미국 위험선호 변화")
    if "US_RATE" in factors:
        path.append("미국 금리·달러 유동성 변화")
    if "FX_KRW" in factors:
        path.append("원/달러 환율과 외국인 달러 수익률 변화")
    if "SEMI" in factors:
        path.append("국내 반도체 대형주 투자심리와 EPS 기대 변화")
    if "EXPORT" in factors:
        path.append("수출·무역수지 흐름이 기업이익 전망에 반영")
    if "FLOW" in factors:
        path.append("외국인·기관·개인 수급이 가격 변동성에 반영")
    if "BOK_CREDIT" in factors:
        path.append("국내 금리·물가·신용 부담이 밸류에이션에 반영")
    if "GOVERNANCE" in factors:
        path.append("주주환원·지배구조 개선 기대가 코리아 디스카운트에 반영")
    if "ACCESS" in factors:
        path.append("시장 접근성 개선 기대가 장기 자금 유입 가능성에 반영")
    if "GEO_OIL" in factors:
        path.append("유가·지정학 리스크가 수입물가와 위험 프리미엄에 반영")
    if direction == "unknown":
        path.append("식별된 시장 영향 경로 부족")
    return tuple(path)


def _has_positive_signal(normalized: str, factors: tuple[FactorCode, ...]) -> bool:
    if "US_RISK" in factors and _contains_any(normalized, ("sox 지수가 급등", "nasdaq 상승", "나스닥 상승", "vix 하락")):
        return True
    if "US_RATE" in factors and _contains_any(normalized, ("금리 하락", "달러 약세", "dxy 하락")):
        return True
    if "SEMI" in factors and _contains_any(normalized, ("hbm", "상향", "가격 상승", "capex 확대", "수요")):
        return True
    if "EXPORT" in factors and _contains_any(normalized, ("수출 증가", "무역수지 개선", "수출 호조")):
        return True
    if "FLOW" in factors and _contains_any(normalized, ("순매수", "매수 확대")):
        return True
    if "GOVERNANCE" in factors and _contains_any(normalized, ("배당 확대", "자사주 소각", "주주환원", "일반주주 보호")):
        return True
    if "ACCESS" in factors and _contains_any(normalized, ("접근성 개선", "fx 개방", "영문공시", "msci")):
        return True
    return any(signal.casefold() in normalized for signal in _POSITIVE_SIGNALS) and not _contains_any(
        normalized, ("금리 상승", "vix 상승", "유가 급등", "환율 급등")
    )


def _has_negative_signal(normalized: str, factors: tuple[FactorCode, ...]) -> bool:
    if "US_RISK" in factors and _contains_any(normalized, ("미국 기술주 급락", "nasdaq 급락", "나스닥 급락", "sox 급락", "vix 상승")):
        return True
    if "US_RATE" in factors and _contains_any(normalized, ("금리 상승", "실질금리 상승", "달러 강세", "dxy 상승")):
        return True
    if "FX_KRW" in factors and _contains_any(normalized, ("원화 급락", "환율 급등", "원화 약세")):
        return True
    if "FLOW" in factors and _contains_any(normalized, ("순매도", "선물 매도", "반대매매")):
        return True
    if "GEO_OIL" in factors and _contains_any(normalized, ("유가 급등", "호르무즈 리스크", "중동")):
        return True
    return any(signal.casefold() in normalized for signal in _NEGATIVE_SIGNALS)


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle.casefold() in value for needle in needles)


def _dedupe(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)
