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


def test_collect_context_raw_events_contains_disclosure_only(monkeypatch, tmp_path):
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

    assert set(out["source_type"].tolist()) == {"disclosure"}
    assert "주요사항보고서" in " ".join(out[out["source_type"] == "disclosure"]["title"].tolist())
