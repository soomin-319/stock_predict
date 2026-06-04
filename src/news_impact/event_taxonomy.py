from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisclosureTaxonomyEvent:
    detail_event_type: str
    impact_event_type: str
    base_direction: str
    base_importance: float
    required_numbers: tuple[str, ...]
    risk_flags: tuple[str, ...]


def classify_disclosure_title(title: str) -> DisclosureTaxonomyEvent:
    normalized = _normalize(title)
    if _contains_any(normalized, ("단일판매", "공급계약", "판매공급계약")):
        return DisclosureTaxonomyEvent(
            detail_event_type="large_contract",
            impact_event_type="contract",
            base_direction="positive",
            base_importance=0.9,
            required_numbers=("contract_amount", "contract_period"),
            risk_flags=(),
        )
    if _contains_any(normalized, ("계약해지", "계약취소", "공급계약해지")):
        return DisclosureTaxonomyEvent(
            detail_event_type="contract_cancel",
            impact_event_type="contract",
            base_direction="negative",
            base_importance=0.9,
            required_numbers=("contract_amount",),
            risk_flags=("contract_cancelled",),
        )
    if _contains_any(normalized, ("유상증자", "신주인수권부사채", "전환사채")):
        return DisclosureTaxonomyEvent(
            detail_event_type="capital_raise",
            impact_event_type="capital_raise",
            base_direction="negative",
            base_importance=1.0,
            required_numbers=("issue_amount", "dilution_rate"),
            risk_flags=("dilution_risk",),
        )
    if _contains_any(normalized, ("소송", "횡령", "배임")):
        return DisclosureTaxonomyEvent(
            detail_event_type="litigation",
            impact_event_type="legal",
            base_direction="negative",
            base_importance=1.2,
            required_numbers=("claim_amount",),
            risk_flags=("legal_risk",),
        )
    if _contains_any(normalized, ("매매거래정지", "거래정지")):
        return DisclosureTaxonomyEvent(
            detail_event_type="trading_halt",
            impact_event_type="other",
            base_direction="negative",
            base_importance=1.2,
            required_numbers=(),
            risk_flags=("trading_halt",),
        )
    return DisclosureTaxonomyEvent(
        detail_event_type="other",
        impact_event_type="other",
        base_direction="neutral",
        base_importance=0.3,
        required_numbers=(),
        risk_flags=(),
    )


def _normalize(value: str) -> str:
    return value.replace("ㆍ", "").replace(" ", "")


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)
