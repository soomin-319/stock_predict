from __future__ import annotations

import pandas as pd

from src.data.investor_context import _fetch_news_sentiment


class _FakeTicker:
    def __init__(self, news_items=None, get_news_items=None):
        self.news = news_items if news_items is not None else []
        self._get_news_items = get_news_items if get_news_items is not None else []

    def get_news(self, **kwargs):
        return self._get_news_items


def test_fetch_news_sentiment_uses_get_news_fallback_when_news_property_empty(monkeypatch):
    today = pd.Timestamp("2026-03-24")
    payload = [
        {
            "content": {
                "title": "삼성전자, 공급계약 체결",
                "providerPublishTime": int(today.timestamp()),
            }
        }
    ]

    monkeypatch.setattr(
        "src.data.investor_context.yf.Ticker",
        lambda symbol: _FakeTicker(news_items=[], get_news_items=payload),
    )

    news_df, coverage = _fetch_news_sentiment(["005930.KS"], "2026-03-24", "2026-03-24")

    assert coverage["requested"] == 1
    assert coverage["successful"] == 1
    assert coverage["failed"] == 0
    assert not news_df.empty
    assert int(news_df["news_article_count"].sum()) == 1


def test_fetch_news_sentiment_parses_pubdate_shape(monkeypatch):
    payload = [
        {
            "content": {
                "headline": "SK하이닉스 실적 기대감",
                "pubDate": "2026-03-24T06:35:00Z",
            }
        }
    ]
    monkeypatch.setattr(
        "src.data.investor_context.yf.Ticker",
        lambda symbol: _FakeTicker(news_items=payload, get_news_items=[]),
    )

    news_df, coverage = _fetch_news_sentiment(["000660.KS"], "2026-03-24", "2026-03-24")

    assert coverage["requested"] == 1
    assert coverage["successful"] == 1
    assert coverage["failed"] == 0
    assert not news_df.empty
    assert str(news_df.iloc[0]["Date"].date()) == "2026-03-24"
