from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.news_impact.mapper import CompanyMapper, MappingCandidate
from src.news_impact.schema import NewsItem


@dataclass(frozen=True)
class NewsFilterDecision:
    item: NewsItem
    mapping_candidates: tuple[MappingCandidate, ...]
    matched_keywords: tuple[str, ...]
    matched_sector_keywords: tuple[str, ...] = ()


_IMPACT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("contract", ("계약", "공급", "수주", "contract", "supply", "order")),
    ("earnings", ("실적", "적자", "흑자", "매출", "영업이익", "earnings", "profit", "loss")),
    ("capital_raise", ("증자", "유상증자", "capital raise", "rights offering")),
    ("legal", ("소송", "제재", "벌금", "lawsuit", "sanction", "fine")),
    ("mna", ("인수", "합병", "m&a", "acquisition", "merger")),
    ("investment", ("투자", "investment")),
    ("disclosure", ("공시", "disclosure")),
    ("recall", ("리콜", "recall")),
)


def filter_news_candidates(
    items: list[NewsItem] | tuple[NewsItem, ...],
    mapper: CompanyMapper,
    sector_keywords: Iterable[str] = (),
) -> list[NewsFilterDecision]:
    sector_keywords = tuple(sector_keywords)
    decisions: list[NewsFilterDecision] = []
    for item in items:
        decision = filter_news_candidate(item, mapper, sector_keywords=sector_keywords)
        if decision is not None:
            decisions.append(decision)
    return decisions


def filter_news_candidate(
    item: NewsItem,
    mapper: CompanyMapper,
    sector_keywords: Iterable[str] = (),
) -> NewsFilterDecision | None:
    title = item.title
    candidates = tuple(mapper.map_text(title))
    matched_keywords = _matched_impact_keywords(title)
    matched_sector_keywords = _matched_sector_keywords(title, sector_keywords)
    if not candidates and not matched_keywords and not matched_sector_keywords:
        return None
    return NewsFilterDecision(
        item=item,
        mapping_candidates=candidates,
        matched_keywords=matched_keywords,
        matched_sector_keywords=matched_sector_keywords,
    )


def should_fetch_article(
    item: NewsItem,
    mapper: CompanyMapper,
    sector_keywords: Iterable[str] = (),
) -> bool:
    return filter_news_candidate(item, mapper, sector_keywords=sector_keywords) is not None


def _matched_impact_keywords(title: str) -> tuple[str, ...]:
    normalized = title.casefold()
    matches: list[str] = []
    for label, keywords in _IMPACT_KEYWORDS:
        if any(keyword.casefold() in normalized for keyword in keywords):
            matches.append(label)
    return tuple(matches)


def _matched_sector_keywords(title: str, sector_keywords: Iterable[str]) -> tuple[str, ...]:
    normalized_title = " ".join(title.casefold().split())
    matches: list[str] = []
    seen: set[str] = set()
    for keyword in sector_keywords:
        cleaned = " ".join(str(keyword).split())
        normalized_keyword = cleaned.casefold()
        if not normalized_keyword or normalized_keyword in seen:
            continue
        seen.add(normalized_keyword)
        if _title_contains_sector_keyword(normalized_title, normalized_keyword):
            matches.append(cleaned)
    return tuple(matches)


def _title_contains_sector_keyword(normalized_title: str, normalized_keyword: str) -> bool:
    if _is_ascii_word_keyword(normalized_keyword):
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
        return re.search(pattern, normalized_title) is not None
    return normalized_keyword in normalized_title


def _is_ascii_word_keyword(keyword: str) -> bool:
    return bool(re.search(r"[a-z0-9]", keyword)) and all(ord(char) < 128 for char in keyword)
