import pandas as pd
from pathlib import Path

from src.news_impact.pipeline import (
    _disclosure_item,
    _news_item,
    _read_company_master,
    _read_json_object,
    _read_watchlist_tickers,
)
from src.reports.news_impact_fixture import build_news_impact_fixture


def _context_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "2026-06-16",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 HBM 공급계약 체결",
                "published_at": "2026-06-16T09:30:00+09:00",
                "provider": "naver",
                "url": "https://news.example/1",
                "raw_id": "n1",
            },
            {
                "Date": "2026-06-16",
                "Symbol": "005930.KS",
                "source_type": "disclosure",
                "title": "[정정]단일판매ㆍ공급계약체결",
                "published_at": "",
                "provider": "dart",
                "url": "https://dart.example/2",
                "raw_id": "20260616000123",
            },
        ]
    )


def test_build_fixture_passes_pipeline_readers(tmp_path):
    bundle = build_news_impact_fixture(
        context_raw_df=_context_df(),
        symbols=["005930.KS"],
        symbol_name_map={"005930.KS": "삼성전자"},
        run_date="2026-06-16",
        output_dir=tmp_path,
    )

    fixture = _read_json_object(Path(bundle.fixture_path))
    news = [_news_item(row) for row in fixture["news"]]
    disclosures = [_disclosure_item(row) for row in fixture["disclosures"]]

    assert len(news) == 1
    assert len(disclosures) == 1
    # signal_at 불변식 + tz-aware 가 통과(_news_item/_disclosure_item __post_init__에서 검증)
    assert disclosures[0].ticker == "005930"
    assert disclosures[0].is_correction is True
    assert _read_watchlist_tickers(Path(bundle.watchlist_path)) == ["005930"]
    master = _read_company_master(Path(bundle.company_master_path))
    assert master["005930"]["company"] == "삼성전자"
