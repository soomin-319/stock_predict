from __future__ import annotations

from datetime import date

import pandas as pd

from src.recommendation.close_betting import format_recommendation_message
from src.recommendation.realtime_close_betting import RealTimeCloseBettingRecommendationService

SAMSUNG = "\uc0bc\uc131\uc804\uc790"
HYNIX = "SK\ud558\uc774\ub2c9\uc2a4"


def _history(symbol: str, close: int, volume: int) -> pd.DataFrame:
    rows = []
    for i in range(260):
        current = close + i
        rows.append(
            {
                "Date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "Symbol": symbol,
                "Open": current - 1,
                "High": current + 2,
                "Low": current - 2,
                "Close": current,
                "Volume": volume,
            }
        )
    return pd.DataFrame(rows)


def test_realtime_service_scores_and_ranks_recommendations():
    raw = pd.concat(
        [
            _history("005930.KS", 70000, 1000000),
            _history("000660.KS", 120000, 100000),
        ],
        ignore_index=True,
    )

    service = RealTimeCloseBettingRecommendationService(
        symbols_provider=lambda: pd.DataFrame(
            [
                {"Symbol": "005930.KS", "Name": SAMSUNG, "Market": "KOSPI"},
                {"Symbol": "000660.KS", "Name": HYNIX, "Market": "KOSPI"},
            ]
        ),
        ohlcv_fetcher=lambda symbols, start, end: raw,
        today_provider=lambda: date(2026, 5, 27),
    )

    recommendations = service.get_recommendations(top_n=2)

    assert [item.rank for item in recommendations] == [1, 2]
    assert recommendations[0].symbol == "005930"
    assert recommendations[0].name == SAMSUNG
    assert recommendations[0].final_score >= 100
    assert recommendations[0].reasons


def test_format_recommendation_message_includes_rank_symbol_score_and_reason():
    service = RealTimeCloseBettingRecommendationService(
        symbols_provider=lambda: pd.DataFrame([{"Symbol": "005930.KS", "Name": SAMSUNG, "Market": "KOSPI"}]),
        ohlcv_fetcher=lambda symbols, start, end: _history("005930.KS", 70000, 1000000),
        today_provider=lambda: date(2026, 5, 27),
    )

    text = format_recommendation_message(service.get_recommendations(top_n=1), as_of=date(2026, 5, 27))

    assert "[\uc2e4\uc2dc\uac04 \ucd94\ucc9c]" in text
    assert "\uae30\uc900\uc77c: 2026-05-27" in text
    assert f"1\uc704 {SAMSUNG}(005930)" in text
    assert "\uc810\uc218:" in text
    assert "\uadfc\uac70:" in text
