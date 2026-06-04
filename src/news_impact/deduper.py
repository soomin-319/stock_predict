from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, TypeVar

from src.news_impact.schema import DisclosureItem, NewsItem


T = TypeVar("T")


@dataclass(frozen=True)
class ClusteredItem:
    item: NewsItem | DisclosureItem
    cluster_id: str


def dedupe_news_items(items: Iterable[NewsItem]) -> list[NewsItem]:
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in items:
        keys = _news_dedupe_keys(item)
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        unique.append(item)
    return unique


def dedupe_disclosures(items: Iterable[DisclosureItem]) -> list[DisclosureItem]:
    return _dedupe_by_key(items, lambda item: item.receipt_no)


def assign_cluster_ids(items: Iterable[NewsItem | DisclosureItem]) -> list[ClusteredItem]:
    clustered: list[ClusteredItem] = []
    for item in items:
        key = _cluster_key(item)
        cluster_id = "cluster-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        clustered.append(ClusteredItem(item=item, cluster_id=cluster_id))
    return clustered


def _dedupe_by_key(items: Iterable[T], key_fn) -> list[T]:
    seen: set[str] = set()
    unique: list[T] = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _news_dedupe_keys(item: NewsItem) -> tuple[str, ...]:
    keys = [f"title:{_normalize_title(item.title)}"]
    if item.original_url:
        keys.append(f"url:{_normalize_url(item.original_url)}")
    if item.url:
        keys.append(f"url:{_normalize_url(item.url)}")
    return tuple(keys)


def _cluster_key(item: NewsItem | DisclosureItem) -> str:
    if isinstance(item, DisclosureItem):
        return f"disclosure:{item.receipt_no}"
    return f"news:{_normalize_title(item.title)}"


def _normalize_url(value: str) -> str:
    return value.strip().lower().rstrip("/")


def _normalize_title(value: str) -> str:
    cleaned = re.sub(r"[^\w\s가-힣]", " ", value.lower())
    return " ".join(cleaned.split())
