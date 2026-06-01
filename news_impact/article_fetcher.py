from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any, Callable
from urllib import request

from news_impact.schema import NewsItem


@dataclass(frozen=True)
class ArticleFetchResult:
    ok: bool
    url: str
    text: str
    status_code: int | None
    error: str | None
    risk_flags: tuple[str, ...]


class ArticleFetcher:
    def __init__(
        self,
        urlopen: Callable[[request.Request, float], Any] = request.urlopen,
        timeout_seconds: float = 10.0,
        user_agent: str = "stock-news-impact/0.1 research metadata-only",
    ) -> None:
        self._urlopen = urlopen
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def fetch(self, url: str, min_text_chars: int = 200) -> ArticleFetchResult:
        http_request = request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html,*/*;q=0.8"},
            method="GET",
        )
        try:
            with self._urlopen(http_request, self.timeout_seconds) as response:
                body = response.read()
                status_code = getattr(response, "status", None) or getattr(response, "code", None)
                content_type = _header_value(response, "Content-Type")
        except Exception as error:  # network, HTTP, TLS, robot blocks
            return ArticleFetchResult(
                ok=False,
                url=url,
                text="",
                status_code=getattr(error, "code", None),
                error=str(error),
                risk_flags=("article_fetch_failed", "needs_full_text_review"),
            )

        text = extract_article_text(_decode_body(body, content_type))
        flags: tuple[str, ...] = ()
        error: str | None = None
        ok = True
        if len(text) < min_text_chars:
            ok = False
            error = "article text too short"
            flags = ("article_text_too_short", "needs_full_text_review")
        elif _looks_paywalled(text):
            ok = False
            error = "article appears paywalled or blocked"
            flags = ("article_fetch_failed", "needs_full_text_review")

        return ArticleFetchResult(
            ok=ok,
            url=url,
            text=text if ok else "",
            status_code=status_code,
            error=error,
            risk_flags=flags,
        )


def select_article_url(item: NewsItem) -> str:
    return item.original_url or item.url


def extract_article_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()
    return _collapse_whitespace(" ".join(parser.parts))


class _VisibleTextParser(HTMLParser):
    _skip_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = _collapse_whitespace(unescape(data))
        if cleaned:
            self.parts.append(cleaned)


def _decode_body(body: bytes, content_type: str | None) -> str:
    charset = "utf-8"
    if content_type:
        match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
        if match:
            charset = match.group(1)
    try:
        return body.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")


def _header_value(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    if hasattr(headers, "get"):
        return headers.get(name)
    return None


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _looks_paywalled(text: str) -> bool:
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            "subscribe to continue",
            "sign in to continue",
            "구독 후",
            "로그인 후",
            "paywall",
        )
    )
