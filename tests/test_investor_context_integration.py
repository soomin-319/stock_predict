import pandas as pd

from src.data.investor_context import (
    InvestorContextConfig,
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

    monkeypatch.setattr(ic, "_fetch_flow", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))
    monkeypatch.setattr(ic, "_fetch_disclosure_scores", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))

    out, cov = add_investor_context_with_coverage(_sample_df(), InvestorContextConfig(enabled=True))
    assert cov["enabled"] is True
    assert cov["flow"]["failed"] == 1
    assert len(out) == 2
    assert out["foreign_net_buy"].fillna(0).sum() == 0


def test_investor_context_news_coverage_is_fixed_zero(monkeypatch):
    import src.data.investor_context as ic

    monkeypatch.setattr(ic, "_fetch_flow", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))
    monkeypatch.setattr(ic, "_fetch_disclosure_scores", lambda *a, **k: (pd.DataFrame(), {"requested": 1, "successful": 0, "failed": 1}))

    out, cov = add_investor_context_with_coverage(_sample_df(), InvestorContextConfig(enabled=True))

    assert cov["news"] == {"requested": 0, "successful": 0, "failed": 0}
    assert "news_sentiment" in out.columns


def test_fetch_flow_returns_empty_when_source_returns_no_data(monkeypatch):
    import src.data.investor_context as ic

    def empty_fetcher(ticker, start, end):
        return pd.DataFrame(columns=["Date", "foreign_net_buy", "institution_net_buy"])

    out, cov = ic._fetch_flow(["005930.KS"], "2024-01-01", "2024-01-31", flow_fetcher=empty_fetcher)

    assert out.empty
    assert cov == {
        "requested": 1,
        "successful": 0,
        "failed": 1,
        "status": "no_data",
        "source": "pykrx",
        "message": "Fetched investor flow for 0/1 symbols via pykrx.",
    }


def test_investor_context_preserves_input_flow_columns_and_reports_source(monkeypatch):
    import src.data.investor_context as ic

    df = _sample_df()
    df["foreign_net_buy"] = [1000, 2000]
    df["institution_net_buy"] = [3000, 4000]
    monkeypatch.setattr(
        ic,
        "_fetch_flow",
        lambda *a, **k: (
            pd.DataFrame(columns=["Date", "Symbol", "foreign_net_buy", "institution_net_buy"]),
            {
                "requested": 1,
                "successful": 0,
                "failed": 1,
                "status": "no_data",
                "source": "pykrx",
                "message": "x",
            },
        ),
    )

    out, cov = ic.add_investor_context_with_coverage(
        df,
        InvestorContextConfig(enabled=True, enable_disclosure=False),
    )

    assert out["foreign_net_buy"].tolist() == [1000, 2000]
    assert out["institution_net_buy"].tolist() == [3000, 4000]
    assert cov["flow"]["status"] == "no_data"
    assert cov["flow"]["source"] == "pykrx"
