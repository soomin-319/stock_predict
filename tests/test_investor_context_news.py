from __future__ import annotations

import pandas as pd

from src.data.investor_context import _fetch_news_sentiment, collect_context_raw_events


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


def test_collect_context_raw_events_contains_news_and_disclosure(monkeypatch, tmp_path):
    news_payload = [
        {
            "content": {
                "headline": "삼성전자 수주 공시 관련 뉴스",
                "pubDate": "2026-03-24T05:10:00Z",
                "provider": "Reuters",
                "canonicalUrl": {"url": "https://example.com/news-1"},
            },
            "id": "news-1",
        }
    ]
    monkeypatch.setattr(
        "src.data.investor_context.yf.Ticker",
        lambda symbol: _FakeTicker(news_items=news_payload, get_news_items=[]),
    )
    monkeypatch.setattr(
        "src.data.investor_context._dart_list",
        lambda api_key, corp_code, start, end: {
            "list": [
                {
                    "rcept_dt": "20260324",
                    "report_nm": "주요사항보고서(공급계약체결)",
                    "rcept_no": "20260324000123",
                }
            ]
        },
    )
    corp_map_csv = tmp_path / "corp_map.csv"
    pd.DataFrame([{"Symbol": "005930.KS", "corp_code": "00126380"}]).to_csv(corp_map_csv, index=False)

    out = collect_context_raw_events(
        symbols=["005930.KS"],
        start="2026-03-24",
        end="2026-03-24",
        dart_api_key="demo",
        dart_corp_map_csv=str(corp_map_csv),
    )

    assert set(out["source_type"].tolist()) == {"news", "disclosure"}
    assert "주요사항보고서" in " ".join(out[out["source_type"] == "disclosure"]["title"].tolist())
    assert out[out["source_type"] == "news"]["provider"].iloc[0] == "Reuters"
