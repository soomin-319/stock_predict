from __future__ import annotations

import email.message
import json
from urllib.error import HTTPError

import pandas as pd
import pytest

from src.data import investor_context as ic
from src.data.investor_context import (
    DEFAULT_MAX_ARTICLES_PER_SYMBOL,
    DEFAULT_MAX_REQUESTS_PER_SECOND,
    MAX_NEWS_FETCH_ATTEMPTS,
    _fetch_naver_news_items,
    _RateLimiter,
    collect_context_raw_events,
)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeClock:
    """Deterministic monotonic clock; sleeping advances time."""

    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def _make_http_error(code: int, retry_after: str | None = None) -> HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(url="https://x", code=code, msg="boom", hdrs=headers, fp=None)


def test_defaults_match_requested_policy():
    assert DEFAULT_MAX_ARTICLES_PER_SYMBOL == 100
    assert DEFAULT_MAX_REQUESTS_PER_SECOND == 10.0


def test_rate_limiter_paces_to_configured_rate(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(ic, "_monotonic", clock.monotonic)
    monkeypatch.setattr(ic, "_sleep", clock.sleep)

    limiter = _RateLimiter(10.0)  # 10/s => 0.1s min interval
    for _ in range(4):
        limiter.acquire()

    # First call is free; the next three are each paced by ~0.1s.
    assert len(clock.sleeps) == 3
    assert clock.sleeps == pytest.approx([0.1, 0.1, 0.1], abs=1e-6)


def test_rate_limiter_zero_disables_pacing(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(ic, "_monotonic", clock.monotonic)
    monkeypatch.setattr(ic, "_sleep", clock.sleep)

    limiter = _RateLimiter(0.0)
    for _ in range(5):
        limiter.acquire()

    assert clock.sleeps == []


def test_fetch_caps_articles_per_symbol_and_stops_early(monkeypatch):
    monkeypatch.setattr(ic, "_sleep", lambda s: None)
    counter = {"n": 0}

    def _fake_urlopen(req, timeout=15):
        i = counter["n"]
        counter["n"] += 1
        items = [
            {
                "title": f"삼성전자 기사 {i}-{j}",
                "description": "삼성전자 내용",
                "originallink": f"https://ex/{i}-{j}",
                "link": f"https://ex/{i}-{j}",
                "pubDate": "Tue, 24 Mar 2026 09:10:00 +0900",
            }
            for j in range(50)
        ]
        return _FakeResp({"items": items})

    monkeypatch.setattr(ic, "urlopen", _fake_urlopen)

    rows = _fetch_naver_news_items(
        symbol="005930.KS",
        symbol_name="삼성전자",
        start_dt=pd.Timestamp("2026-03-01"),
        end_dt=pd.Timestamp("2026-03-31"),
        client_id="id",
        client_secret="secret",
        errors=[],
        rate_limiter=_RateLimiter(0.0),
        max_articles=100,
    )

    # 50 fresh items per call, cap 100 => exactly 2 calls then early stop.
    assert counter["n"] == 2
    assert len(rows) == 100


def test_fetch_retries_transient_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(ic, "_sleep", lambda s: None)
    calls = {"n": 0}

    def _fake_urlopen(req, timeout=15):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_http_error(429)
        # Success fills the per-symbol cap so the query loop stops early.
        items = [
            {
                "title": f"삼성전자 실적 {j}",
                "description": "삼성전자 기사",
                "originallink": f"https://ex/ok-{j}",
                "link": f"https://ex/ok-{j}",
                "pubDate": "Tue, 24 Mar 2026 09:10:00 +0900",
            }
            for j in range(100)
        ]
        return _FakeResp({"items": items})

    monkeypatch.setattr(ic, "urlopen", _fake_urlopen)
    errors: list[dict] = []

    rows = _fetch_naver_news_items(
        symbol="005930.KS",
        symbol_name="삼성전자",
        start_dt=pd.Timestamp("2026-03-01"),
        end_dt=pd.Timestamp("2026-03-31"),
        client_id="id",
        client_secret="secret",
        errors=errors,
        rate_limiter=_RateLimiter(0.0),
        max_articles=100,
    )

    assert calls["n"] == 2  # one 429, one successful retry
    assert errors == []
    assert len(rows) == 100


def test_fetch_records_error_after_exhausting_retries(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(ic, "_sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def _fake_urlopen(req, timeout=15):
        calls["n"] += 1
        raise _make_http_error(429)

    monkeypatch.setattr(ic, "urlopen", _fake_urlopen)
    errors: list[dict] = []

    rows = _fetch_naver_news_items(
        symbol="005930.KS",
        symbol_name="삼성전자",
        start_dt=pd.Timestamp("2026-03-01"),
        end_dt=pd.Timestamp("2026-03-31"),
        client_id="id",
        client_secret="secret",
        errors=errors,
        rate_limiter=_RateLimiter(0.0),
        max_articles=100,
    )

    # Each of the 6 queries attempts MAX_NEWS_FETCH_ATTEMPTS times.
    assert calls["n"] == 6 * MAX_NEWS_FETCH_ATTEMPTS
    assert rows == []
    assert len(errors) == 6
    assert all("429" in e["message"] for e in errors)
    # Exponential backoff between attempts (2 retries per query => 2 sleeps each).
    assert sleeps and all(s in (1.0, 2.0) for s in sleeps)


def test_fetch_honors_retry_after_header(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(ic, "_sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def _fake_urlopen(req, timeout=15):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_http_error(429, retry_after="2")
        return _FakeResp({"items": []})

    monkeypatch.setattr(ic, "urlopen", _fake_urlopen)

    _fetch_naver_news_items(
        symbol="005930.KS",
        symbol_name="삼성전자",
        start_dt=pd.Timestamp("2026-03-01"),
        end_dt=pd.Timestamp("2026-03-31"),
        client_id="id",
        client_secret="secret",
        errors=[],
        rate_limiter=_RateLimiter(0.0),
        max_articles=100,
    )

    assert 2.0 in sleeps


def test_fetch_does_not_retry_non_transient_error(monkeypatch):
    monkeypatch.setattr(ic, "_sleep", lambda s: None)
    calls = {"n": 0}

    def _fake_urlopen(req, timeout=15):
        calls["n"] += 1
        raise _make_http_error(400)  # client error, not transient

    monkeypatch.setattr(ic, "urlopen", _fake_urlopen)
    errors: list[dict] = []

    _fetch_naver_news_items(
        symbol="005930.KS",
        symbol_name="삼성전자",
        start_dt=pd.Timestamp("2026-03-01"),
        end_dt=pd.Timestamp("2026-03-31"),
        client_id="id",
        client_secret="secret",
        errors=errors,
        rate_limiter=_RateLimiter(0.0),
        max_articles=100,
    )

    # 6 queries, one attempt each (no retry on 400).
    assert calls["n"] == 6
    assert len(errors) == 6


def test_collect_context_raw_events_caps_articles(monkeypatch):
    monkeypatch.setattr(ic, "_sleep", lambda s: None)

    def _fake_urlopen(req, timeout=15):
        items = [
            {
                "title": f"삼성전자 기사 {j}",
                "description": "삼성전자 내용",
                "originallink": f"https://ex/{id(req)}-{j}",
                "link": f"https://ex/{id(req)}-{j}",
                "pubDate": "Tue, 24 Mar 2026 09:10:00 +0900",
            }
            for j in range(50)
        ]
        return _FakeResp({"items": items})

    monkeypatch.setattr(ic, "urlopen", _fake_urlopen)

    out = collect_context_raw_events(
        symbols=["005930.KS"],
        start="2026-03-01",
        end="2026-03-31",
        symbol_name_map={"005930.KS": "삼성전자"},
        naver_client_id="id",
        naver_client_secret="secret",
        max_articles_per_symbol=100,
        max_requests_per_second=0.0,
    )

    news = out[out["source_type"] == "news"]
    assert len(news) <= 100
    assert len(news) == 100
