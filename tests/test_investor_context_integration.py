import pandas as pd

from src.data.investor_context import (
    InvestorContextConfig,
    _fetch_news_sentiment,
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


def test_fetch_news_sentiment_is_disabled_in_all_ranges():
    out, cov = _fetch_news_sentiment(["005930.KS"], "2024-01-01", "2024-01-03")
    assert cov == {"requested": 0, "successful": 0, "failed": 0}
    assert out.empty
