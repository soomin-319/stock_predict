import pandas as pd

from src.data import investor_context as ic


def _flow_frame(date, foreign, institution):
    return pd.DataFrame(
        {"Date": pd.to_datetime([date]), "foreign_net_buy": [foreign], "institution_net_buy": [institution]}
    )


def test_fetch_flow_reports_per_symbol_success():
    def fake_fetch(ticker, start, end):
        assert start == "2026-06-24"
        assert end == "2026-06-25"
        if ticker == "000660":
            return pd.DataFrame(columns=["Date", "foreign_net_buy", "institution_net_buy"])
        return _flow_frame("2026-06-25", 100.0, 10.0)

    df, cov = ic._fetch_flow(
        ["005930.KS", "000660.KS"], "2026-06-24", "2026-06-25", flow_fetcher=fake_fetch
    )

    assert cov["requested"] == 2
    assert cov["successful"] == 1
    assert cov["failed"] == 1
    assert cov["source"] == "pykrx"
    assert set(df["Symbol"]) == {"005930.KS"}
    assert str(df["Date"].dtype).startswith("datetime64")


def test_add_investor_context_populates_flow(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["2026-06-25", "2026-06-25"],
            "Symbol": ["005930.KS", "000660.KS"],
            "Close": [356000.0, 2783000.0],
        }
    )

    def fake_flow(symbols, start, end, **kwargs):
        assert symbols == ["000660.KS", "005930.KS"]
        assert start == "2026-06-25"
        assert end == "2026-06-25"
        rows = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-06-25", "2026-06-25"]),
                "Symbol": ["005930.KS", "000660.KS"],
                "foreign_net_buy": [100.0, 200.0],
                "institution_net_buy": [10.0, 20.0],
            }
        )
        return rows, {"requested": 2, "successful": 2, "failed": 0, "status": "ok", "source": "pykrx", "message": "x"}

    monkeypatch.setattr(ic, "_fetch_flow", fake_flow)
    cfg = ic.InvestorContextConfig(enabled=True, enable_disclosure=False)
    out, cov = ic.add_investor_context_with_coverage(df, cfg)

    assert out.loc[out["Symbol"] == "005930.KS", "foreign_net_buy"].iloc[0] == 100.0
    assert out.loc[out["Symbol"] == "000660.KS", "institution_net_buy"].iloc[0] == 20.0
    assert cov["flow"]["successful"] == 2


def test_add_investor_context_replaces_existing_zero_flow_columns(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["2026-06-25"],
            "Symbol": ["005930.KS"],
            "Close": [356000.0],
            "foreign_net_buy": [0.0],
            "institution_net_buy": [0.0],
        }
    )

    def fake_flow(symbols, start, end, **kwargs):
        rows = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-06-25"]),
                "Symbol": ["005930.KS"],
                "foreign_net_buy": [300.0],
                "institution_net_buy": [30.0],
            }
        )
        return rows, {"requested": 1, "successful": 1, "failed": 0, "status": "ok", "source": "pykrx", "message": "x"}

    monkeypatch.setattr(ic, "_fetch_flow", fake_flow)
    out, cov = ic.add_investor_context_with_coverage(
        df,
        ic.InvestorContextConfig(enabled=True, enable_disclosure=False),
    )

    assert out["foreign_net_buy"].tolist() == [300.0]
    assert out["institution_net_buy"].tolist() == [30.0]
    assert cov["flow"]["successful"] == 1
