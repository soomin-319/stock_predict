import pandas as pd

from src.data.investor_context import (
    InvestorContextConfig,
    _fetch_news_sentiment,
    _headline_news_features,
    add_investor_context_with_coverage,
)


def _sample_df():
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Symbol": ["005930.KS", "005930.KS"],
            "Open": [100, 101],
            "High": [101, 102],
            "Low": [99, 100],
            "Close": [100, 101],
            "Volume": [10000, 12000],
        }
    )


def test_investor_context_disabled_returns_defaults():
    out, cov = add_investor_context_with_coverage(_sample_df(), InvestorContextConfig(enabled=False))
    assert cov["enabled"] is False
    for c in ["foreign_net_buy", "institution_net_buy", "disclosure_score", "news_sentiment"]:
        assert c in out.columns


def test_investor_context_enabled_graceful_without_sources(monkeypatch):
    import src.data.investor_context as ic

    monkeypatch.setattr(ic, "_fetch_flow_pykrx", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))
    monkeypatch.setattr(ic, "_fetch_disclosure_scores", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))
    monkeypatch.setattr(ic, "_fetch_news_sentiment", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))

    out, cov = add_investor_context_with_coverage(_sample_df(), InvestorContextConfig(enabled=True))
    assert cov["enabled"] is True
    assert cov["flow"]["failed"] == 1
    assert len(out) == 2
    assert out["foreign_net_buy"].fillna(0).sum() == 0


def test_investor_context_can_disable_only_news(monkeypatch):
    import src.data.investor_context as ic

    called = {"news": 0}

    monkeypatch.setattr(ic, "_fetch_flow_pykrx", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))
    monkeypatch.setattr(ic, "_fetch_disclosure_scores", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))

    def _news(*args, **kwargs):
        called["news"] += 1
        return pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}

    monkeypatch.setattr(ic, "_fetch_news_sentiment", _news)

    out, cov = add_investor_context_with_coverage(
        _sample_df(),
        InvestorContextConfig(enabled=True, enable_news=False),
    )

    assert called["news"] == 0
    assert cov["news"] == {"requested": 0, "successful": 0, "failed": 0}
    assert "news_sentiment" in out.columns


def test_fetch_flow_pykrx_uses_shared_import_helper(monkeypatch):
    import src.data.investor_context as ic

    monkeypatch.setattr(ic, "import_pykrx_stock", lambda: None)

    out, cov = ic._fetch_flow_pykrx(["005930.KS"], "2024-01-01", "2024-01-31")

    assert out.empty
    assert cov == {"requested": 1, "successful": 0, "failed": 1}


def test_headline_news_features_prioritize_price_relevant_titles():
    strong = "삼성전자 대규모 공급계약 체결, 실적 개선 기대"
    weak = "오늘 장마감 시황 브리핑"

    strong_sentiment, strong_relevance, strong_impact = _headline_news_features(strong)
    weak_sentiment, weak_relevance, weak_impact = _headline_news_features(weak)

    assert strong_sentiment > 0.5
    assert strong_relevance > weak_relevance
    assert strong_impact > weak_impact


def test_fetch_news_sentiment_returns_relevance_and_article_count(monkeypatch):
    import src.data.investor_context as ic

    class _Ticker:
        news = [
            {
                "providerPublishTime": 1704153600,  # 2024-01-02 UTC
                "title": "삼성전자 대규모 공급계약 체결",
            },
            {
                "providerPublishTime": 1704153600,
                "title": "삼성전자 장마감 시황 브리핑",
            },
            {
                "providerPublishTime": 1704153600,
                "title": "  삼성전자   대규모   공급계약 체결  ",
            },
        ]

    monkeypatch.setattr(ic.yf, "Ticker", lambda symbol: _Ticker())

    out, cov = _fetch_news_sentiment(["005930.KS"], "2024-01-01", "2024-01-03")

    assert cov["successful"] == 1
    assert out.loc[0, "news_article_count"] == 2
    assert out.loc[0, "news_relevance_score"] > 0
    assert out.loc[0, "news_sentiment"] != 0


def test_headline_news_features_uses_ai_when_available(monkeypatch):
    import src.data.investor_context as ic

    monkeypatch.setattr(ic, "_score_headline_with_openai", lambda *a, **k: (0.9, 0.8, 0.7))

    sentiment, relevance, impact = _headline_news_features(
        "삼성전자 대규모 공급계약 체결",
        InvestorContextConfig(enabled=True, news_scoring_mode="ai", openai_api_key="test", openai_model="demo-model"),
    )

    assert (sentiment, relevance, impact) == (0.9, 0.8, 0.7)
