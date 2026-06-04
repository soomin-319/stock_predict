from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MappingCandidate:
    ticker: str
    relation_type: str
    relevance: float
    confidence: float
    evidence: str
    risk_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "relation_type": self.relation_type,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "risk_flags": list(self.risk_flags),
        }


class CompanyMapper:
    def __init__(
        self,
        companies: Iterable[dict[str, str]],
        aliases: Iterable[dict[str, str]] = (),
        relationships: Iterable[dict[str, str]] = (),
    ) -> None:
        self._companies = [dict(row) for row in companies]
        self._aliases = [dict(row) for row in aliases]
        self._relationships = [dict(row) for row in relationships]
        self._company_by_ticker = {row["ticker"]: row for row in self._companies}

    def map_text(self, text: str) -> list[MappingCandidate]:
        normalized = _normalize_text(text)
        candidates: list[MappingCandidate] = []
        candidates.extend(self._ticker_matches(normalized))
        candidates.extend(self._corp_code_matches(normalized))
        candidates.extend(self._company_name_matches(normalized))
        candidates.extend(self._alias_matches(normalized))
        return _dedupe_and_rank(candidates)

    def expand_relationships(
        self,
        direct_candidates: Iterable[MappingCandidate],
    ) -> list[MappingCandidate]:
        expanded = list(direct_candidates)
        seen = {candidate.ticker for candidate in expanded}
        for candidate in direct_candidates:
            for relationship in self._relationships:
                if relationship["ticker"] != candidate.ticker:
                    continue
                related_ticker = relationship["related_ticker"]
                if related_ticker in seen or related_ticker not in self._company_by_ticker:
                    continue
                expanded.append(
                    MappingCandidate(
                        ticker=related_ticker,
                        relation_type=relationship["relation_type"],
                        relevance=float(relationship["relation_strength"]),
                        confidence=min(candidate.confidence, 0.7),
                        evidence=f"relationship:{candidate.ticker}",
                        risk_flags=("weak_relation",),
                    )
                )
                seen.add(related_ticker)
        return expanded

    def _ticker_matches(self, normalized: str) -> list[MappingCandidate]:
        candidates: list[MappingCandidate] = []
        for row in self._companies:
            ticker = row["ticker"]
            if _contains_token(normalized, ticker):
                candidates.append(
                    MappingCandidate(
                        ticker=ticker,
                        relation_type="direct",
                        relevance=1.0,
                        confidence=1.0,
                        evidence="ticker",
                    )
                )
        return candidates

    def _corp_code_matches(self, normalized: str) -> list[MappingCandidate]:
        candidates: list[MappingCandidate] = []
        for row in self._companies:
            corp_code = row["corp_code"]
            if _contains_token(normalized, corp_code):
                candidates.append(
                    MappingCandidate(
                        ticker=row["ticker"],
                        relation_type="direct",
                        relevance=1.0,
                        confidence=1.0,
                        evidence="corp_code",
                    )
                )
        return candidates

    def _company_name_matches(self, normalized: str) -> list[MappingCandidate]:
        candidates: list[MappingCandidate] = []
        for row in self._companies:
            company = row["company"]
            if _contains_phrase(normalized, company):
                candidates.append(
                    MappingCandidate(
                        ticker=row["ticker"],
                        relation_type="direct",
                        relevance=1.0,
                        confidence=1.0,
                        evidence=f"company:{company}",
                    )
                )
        return candidates

    def _alias_matches(self, normalized: str) -> list[MappingCandidate]:
        matches_by_alias: dict[str, list[dict[str, str]]] = {}
        for row in self._aliases:
            alias = row["alias"]
            if _contains_phrase(normalized, alias):
                matches_by_alias.setdefault(alias, []).append(row)

        candidates: list[MappingCandidate] = []
        for alias, rows in matches_by_alias.items():
            ambiguous = len({row["ticker"] for row in rows}) > 1
            for row in rows:
                confidence = float(row["confidence"])
                risk_flags: tuple[str, ...] = ()
                if ambiguous:
                    confidence = min(confidence, 0.6)
                    risk_flags = ("ambiguous_mapping",)
                candidates.append(
                    MappingCandidate(
                        ticker=row["ticker"],
                        relation_type="direct",
                        relevance=1.0,
                        confidence=confidence,
                        evidence=f"alias:{alias}",
                        risk_flags=risk_flags,
                    )
                )
        return candidates


def load_company_mapping(
    company_master_path: str | Path,
    aliases_path: str | Path,
    relationships_path: str | Path,
) -> CompanyMapper:
    return CompanyMapper(
        companies=_read_csv(company_master_path),
        aliases=_read_csv(aliases_path),
        relationships=_read_csv(relationships_path),
    )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_token(normalized_text: str, token: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(token.lower())}(?!\w)", normalized_text) is not None


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    return _normalize_text(phrase) in normalized_text


def _dedupe_and_rank(candidates: list[MappingCandidate]) -> list[MappingCandidate]:
    best_by_ticker: dict[str, MappingCandidate] = {}
    for candidate in candidates:
        current = best_by_ticker.get(candidate.ticker)
        if current is None or _candidate_rank(candidate) > _candidate_rank(current):
            best_by_ticker[candidate.ticker] = candidate
    return sorted(
        best_by_ticker.values(),
        key=lambda candidate: (
            candidate.confidence,
            candidate.relevance,
            1 if candidate.evidence == "ticker" else 0,
            candidate.ticker,
        ),
        reverse=True,
    )


def _candidate_rank(candidate: MappingCandidate) -> tuple[float, float, int]:
    evidence_priority = {"ticker": 3, "corp_code": 3}
    priority = evidence_priority.get(candidate.evidence, 2 if candidate.evidence.startswith("company:") else 1)
    return (candidate.confidence, candidate.relevance, priority)
