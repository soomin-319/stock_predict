"""Sector keyword loading and company search query helpers."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ALLOWED_LOCALES = {"en", "ko"}
_ALLOWED_QUERY_LOCALES = _ALLOWED_LOCALES | {"all"}
_HANGUL_RE = re.compile(r"[가-힣]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SectorKeyword:
    sector: str
    keyword: str
    weight: float
    locale: str
    notes: str = ""


class SectorKeywordIndex:
    def __init__(self, keywords: Iterable[SectorKeyword]):
        self._keywords: dict[tuple[str, str], SectorKeyword] = {}
        for keyword in keywords:
            validated = _validate_keyword(keyword)
            key = (_normalize_key(validated.sector), _normalize_key(validated.keyword))
            existing = self._keywords.get(key)
            if existing is None or validated.weight > existing.weight:
                self._keywords[key] = validated

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, str]]) -> "SectorKeywordIndex":
        return cls(_keyword_from_row(row) for row in rows)

    def for_sector(self, sector: str) -> tuple[SectorKeyword, ...]:
        sector_key = _normalize_key(sector)
        matches = [kw for (kw_sector, _), kw in self._keywords.items() if kw_sector == sector_key]
        return tuple(sorted(matches, key=lambda kw: (kw.weight, kw.keyword), reverse=True))


def load_sector_keywords(path, *, missing_ok: bool = False) -> SectorKeywordIndex:
    csv_path = Path(path)
    if not csv_path.exists():
        if missing_ok:
            return SectorKeywordIndex(())
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return SectorKeywordIndex.from_rows(csv.DictReader(f))


def build_company_search_queries(
    company: dict[str, str],
    aliases: Iterable[str],
    keyword_index: SectorKeywordIndex,
    *,
    max_sector_keywords: int | None = None,
    locale: str = "ko",
) -> tuple[str, ...]:
    _validate_query_locale(locale)
    raw_queries: list[str] = []
    raw_queries.extend(
        query
        for query in (company.get("company", ""), *aliases)
        if _matches_query_locale(query, locale)
    )

    sector_keywords = tuple(
        keyword
        for keyword in keyword_index.for_sector(company.get("sector", ""))
        if locale == "all" or keyword.locale == locale
    )
    if max_sector_keywords is not None:
        sector_keywords = sector_keywords[:max_sector_keywords]
    raw_queries.extend(keyword.keyword for keyword in sector_keywords)

    seen: set[str] = set()
    queries: list[str] = []
    for raw_query in raw_queries:
        query = _normalize_text(raw_query)
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            queries.append(query)
    return tuple(queries)


def _keyword_from_row(row: dict[str, str]) -> SectorKeyword:
    weight_raw = _normalize_text(row.get("weight", ""))
    try:
        weight = float(weight_raw)
    except ValueError as exc:
        raise ValueError("weight must be a number") from exc

    return SectorKeyword(
        sector=_normalize_text(row.get("sector", "")),
        keyword=_normalize_text(row.get("keyword", "")),
        weight=weight,
        locale=_normalize_text(row.get("locale", "")),
        notes=_normalize_text(row.get("notes", "")),
    )


def _validate_keyword(keyword: SectorKeyword) -> SectorKeyword:
    sector = _normalize_text(keyword.sector)
    text = _normalize_text(keyword.keyword)
    locale = _normalize_text(keyword.locale)
    notes = _normalize_text(keyword.notes)

    if not sector:
        raise ValueError("sector must not be empty")
    if not text:
        raise ValueError("keyword must not be empty")
    if not isinstance(keyword.weight, (int, float)):
        raise ValueError("weight must be a number")
    weight = float(keyword.weight)
    if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
        raise ValueError("weight must be between 0.0 and 1.0")
    if locale not in _ALLOWED_LOCALES:
        raise ValueError("locale must be one of: en, ko")

    return SectorKeyword(sector, text, weight, locale, notes)


def _validate_query_locale(locale: str) -> None:
    if locale not in _ALLOWED_QUERY_LOCALES:
        allowed = ", ".join(sorted(_ALLOWED_QUERY_LOCALES))
        raise ValueError(f"locale must be one of: {allowed}")


def _matches_query_locale(query: str, locale: str) -> bool:
    if locale == "all":
        return True
    has_hangul = _HANGUL_RE.search(query) is not None
    return has_hangul if locale == "ko" else not has_hangul


def _normalize_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _normalize_key(value: object) -> str:
    return _normalize_text(value).casefold()
