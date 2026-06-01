from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import date, datetime, time
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Protocol
from urllib import request
from urllib.parse import urlencode, urlparse

from news_impact.env_config import load_api_keys
from news_impact.market_clock import KST, KrxTradingCalendar, classify_market_session
from news_impact.schema import DisclosureItem, NewsItem, compute_signal_at


_HANGUL_RE = re.compile(r"[가-힣]")
_ALLOWED_NEWS_LANGUAGES = {"ko", "all"}


class JsonGetTransport(Protocol):
    def get_json(
        self,
        url: str,
        params: dict[str, str | int],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        ...


class UrllibJsonGetTransport:
    def __init__(
        self,
        urlopen: Callable[[request.Request, float], Any] = request.urlopen,
    ) -> None:
        self._urlopen = urlopen

    def get_json(
        self,
        url: str,
        params: dict[str, str | int],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        query = urlencode(params)
        separator = "&" if "?" in url else "?"
        http_request = request.Request(
            url=f"{url}{separator}{query}",
            headers=headers,
            method="GET",
        )
        with self._urlopen(http_request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("HTTP JSON response must be an object")
        return parsed


@dataclass(frozen=True)
class NaverNewsCollector:
    client_id: str
    client_secret: str
    transport: JsonGetTransport
    calendar: KrxTradingCalendar
    base_url: str = "https://openapi.naver.com/v1/search/news.json"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        transport: JsonGetTransport,
        calendar: KrxTradingCalendar,
        env_path: str | Path | None = None,
    ) -> "NaverNewsCollector":
        keys = load_api_keys(env_path)
        if not keys.naver_client_id or not keys.naver_client_secret:
            raise ValueError("Missing NAVER_CLIENT_ID or NAVER_CLIENT_SECRET")
        return cls(
            client_id=keys.naver_client_id,
            client_secret=keys.naver_client_secret,
            transport=transport,
            calendar=calendar,
        )

    def search_news(
        self,
        query: str,
        collected_at: datetime,
        display: int = 10,
        start: int = 1,
        sort: str = "date",
        language: str = "ko",
    ) -> list[NewsItem]:
        _validate_news_language(language)
        response = self.transport.get_json(
            url=self.base_url,
            params={
                "query": query,
                "display": display,
                "start": start,
                "sort": sort,
            },
            headers={
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret,
            },
            timeout_seconds=self.timeout_seconds,
        )
        raw_items = response.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("Naver response field 'items' must be a list")
        items = [self._to_news_item(raw_item, collected_at) for raw_item in raw_items]
        if language == "ko":
            return [item for item in items if _is_korean_news_item(item)]
        return items

    def search_news_for_queries(
        self,
        queries: list[str],
        collected_at: datetime,
        display: int = 10,
        start: int = 1,
        sort: str = "date",
        language: str = "ko",
    ) -> list[NewsItem]:
        _validate_news_language(language)
        seen_queries: set[str] = set()
        seen_urls: set[str] = set()
        results: list[NewsItem] = []
        for query in queries:
            normalized_query = " ".join(query.split())
            if not normalized_query or normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            for item in self.search_news(
                query=normalized_query,
                collected_at=collected_at,
                display=display,
                start=start,
                sort=sort,
                language=language,
            ):
                dedupe_url = item.original_url or item.url
                if dedupe_url in seen_urls:
                    continue
                seen_urls.add(dedupe_url)
                results.append(item)
        return results

    def search_news_for_day(
        self,
        query: str,
        search_date: date,
        collected_at: datetime,
        max_results: int = 100,
        language: str = "ko",
    ) -> list[NewsItem]:
        _validate_max_results(max_results)
        items = self.search_news(
            query=query,
            collected_at=collected_at,
            display=max_results,
            start=1,
            sort="date",
            language=language,
        )
        return [
            item
            for item in items
            if item.published_at.astimezone(KST).date() == search_date
        ]

    def _to_news_item(
        self,
        raw_item: dict[str, Any],
        collected_at: datetime,
    ) -> NewsItem:
        published_at = _parse_naver_pub_date(_require_str(raw_item, "pubDate"))
        signal_at = compute_signal_at(published_at, collected_at)
        original_url = _optional_str(raw_item, "originallink")
        publisher_domain = _publisher_domain(original_url or _optional_str(raw_item, "link"))
        return NewsItem(
            source="naver_news",
            title=_clean_naver_text(_require_str(raw_item, "title")),
            summary=_clean_naver_text(_require_str(raw_item, "description")),
            url=_require_str(raw_item, "link"),
            original_url=original_url,
            publisher_domain=publisher_domain,
            publisher_domain_source="originallink" if original_url else "link",
            publisher_confidence=0.7 if original_url else 0.4,
            published_at=published_at,
            timestamp_source="naver_pubDate",
            collected_at=collected_at,
            signal_at=signal_at,
            market_session=classify_market_session(signal_at, self.calendar),
            raw_text=None,
            storage_policy="metadata_only",
            quality_flags=(
                "timestamp_source_ambiguous",
                "publisher_derived_from_domain",
            ),
        )


@dataclass(frozen=True)
class OpenDartCollector:
    api_key: str
    transport: JsonGetTransport
    calendar: KrxTradingCalendar
    base_url: str = "https://opendart.fss.or.kr/api/list.json"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        transport: JsonGetTransport,
        calendar: KrxTradingCalendar,
        env_path: str | Path | None = None,
    ) -> "OpenDartCollector":
        keys = load_api_keys(env_path)
        if not keys.opendart_api_key:
            raise ValueError("Missing OPENDART_API_KEY")
        return cls(
            api_key=keys.opendart_api_key,
            transport=transport,
            calendar=calendar,
        )

    def list_disclosures(
        self,
        corp_code: str,
        ticker: str,
        start_date: date,
        end_date: date,
        collected_at: datetime,
        page_count: int = 100,
    ) -> list[DisclosureItem]:
        response = self.transport.get_json(
            url=self.base_url,
            params={
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bgn_de": start_date.strftime("%Y%m%d"),
                "end_de": end_date.strftime("%Y%m%d"),
                "last_reprt_at": "Y",
                "sort": "date",
                "sort_mth": "desc",
                "page_no": 1,
                "page_count": page_count,
            },
            headers={},
            timeout_seconds=self.timeout_seconds,
        )
        status = response.get("status")
        if status == "013":
            return []
        if status not in (None, "000"):
            message = response.get("message", "")
            raise ValueError(f"OpenDART request failed: {status} {message}")
        raw_items = response.get("list", [])
        if not isinstance(raw_items, list):
            raise ValueError("OpenDART response field 'list' must be a list")
        return [
            self._to_disclosure_item(raw_item, ticker=ticker, collected_at=collected_at)
            for raw_item in raw_items
        ]

    def list_disclosures_for_day(
        self,
        corp_code: str,
        ticker: str,
        disclosure_date: date,
        collected_at: datetime,
        max_results: int = 100,
    ) -> list[DisclosureItem]:
        _validate_max_results(max_results)
        return self.list_disclosures(
            corp_code=corp_code,
            ticker=ticker,
            start_date=disclosure_date,
            end_date=disclosure_date,
            collected_at=collected_at,
            page_count=max_results,
        )

    def _to_disclosure_item(
        self,
        raw_item: dict[str, Any],
        ticker: str,
        collected_at: datetime,
    ) -> DisclosureItem:
        receipt_no = _require_str(raw_item, "rcept_no")
        disclosure_at = _parse_dart_receipt_date(_require_str(raw_item, "rcept_dt"))
        signal_at = compute_signal_at(disclosure_at, collected_at)
        disclosure_title = _require_str(raw_item, "report_nm")
        return DisclosureItem(
            source="opendart",
            receipt_no=receipt_no,
            corp_code=_require_str(raw_item, "corp_code"),
            ticker=_optional_str(raw_item, "stock_code") or ticker,
            disclosure_title=disclosure_title,
            disclosure_at=disclosure_at,
            collected_at=collected_at,
            signal_at=signal_at,
            is_correction=_is_correction_title(disclosure_title),
            original_receipt_no=None,
            url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}",
            quality_flags=("missing_disclosure_time",),
        )


def _require_str(raw_item: dict[str, Any], key: str) -> str:
    value = raw_item.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"Missing required string field: {key}")
    return value


def _optional_str(raw_item: dict[str, Any], key: str) -> str | None:
    value = raw_item.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _clean_naver_text(value: str) -> str:
    without_tags = re.sub(r"</?b>", "", value)
    return html.unescape(without_tags)


def _validate_max_results(max_results: int) -> None:
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")


def _validate_news_language(language: str) -> None:
    if language not in _ALLOWED_NEWS_LANGUAGES:
        allowed = ", ".join(sorted(_ALLOWED_NEWS_LANGUAGES))
        raise ValueError(f"language must be one of: {allowed}")


def _is_korean_news_item(item: NewsItem) -> bool:
    return _HANGUL_RE.search(f"{item.title} {item.summary}") is not None


def _parse_naver_pub_date(value: str) -> datetime:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def _parse_dart_receipt_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d").replace(tzinfo=KST)


def _publisher_domain(url: str | None) -> str | None:
    if not url:
        return None
    hostname = urlparse(url).hostname
    return hostname.lower() if hostname else None


def _is_correction_title(title: str) -> bool:
    return "정정" in title or "Correction" in title or "correction" in title
