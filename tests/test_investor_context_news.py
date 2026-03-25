from __future__ import annotations

import pandas as pd

from src.data.investor_context import (
    _fetch_news_sentiment,
    collect_context_raw_events,
)


def test_fetch_news_sentiment_is_disabled():
    news_df, coverage = _fetch_news_sentiment(["005930.KS"], "2026-03-24", "2026-03-24")

    assert coverage["requested"] == 0
    assert coverage["successful"] == 0
    assert coverage["failed"] == 0
    assert news_df.empty


def test_collect_context_raw_events_contains_naver_news_and_disclosure(monkeypatch, tmp_path):
    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            import json

            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "src.data.investor_context.urlopen",
        lambda req, timeout=15: _FakeResp(
            {
                "items": [
                    {
                        "title": "<b>삼성전자</b> 실적 개선",
                        "description": "삼성전자 실적 관련 기사",
                        "originallink": "https://example.com/news-1",
                        "link": "https://example.com/news-1",
                        "pubDate": "Tue, 24 Mar 2026 09:10:00 +0900",
                    }
                ]
            }
        ),
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
        symbol_name_map={"005930.KS": "삼성전자"},
        naver_client_id="nid",
        naver_client_secret="nsecret",
    )

    assert set(out["source_type"].tolist()) == {"news", "disclosure"}
    assert "주요사항보고서" in " ".join(out[out["source_type"] == "disclosure"]["title"].tolist())
    assert out[out["source_type"] == "news"]["provider"].iloc[0] == "naver_news_api"
