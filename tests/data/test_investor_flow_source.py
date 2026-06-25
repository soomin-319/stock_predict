import pandas as pd

from src.data.investor_flow_source import fetch_investor_flow_pykrx


class _FakeStock:
    def get_market_trading_value_by_date(self, fromdate, todate, ticker):
        assert fromdate == "20260624"
        assert todate == "20260625"
        assert ticker == "005930"
        idx = pd.to_datetime(["2026-06-24", "2026-06-25"])
        return pd.DataFrame(
            {"기관합계": [10, -5], "개인": [1, 2], "외국인합계": [100, -50], "전체": [111, -53]},
            index=idx,
        )


def test_maps_foreign_and_institution_columns():
    out = fetch_investor_flow_pykrx("005930", "2026-06-24", "2026-06-25", stock_module=_FakeStock())

    assert list(out.columns) == ["Date", "foreign_net_buy", "institution_net_buy"]
    assert out["foreign_net_buy"].tolist() == [100.0, -50.0]
    assert out["institution_net_buy"].tolist() == [10.0, -5.0]
    assert str(out["Date"].dtype).startswith("datetime64")


def test_empty_source_returns_typed_empty_frame():
    class _Empty:
        def get_market_trading_value_by_date(self, *args):
            return pd.DataFrame()

    out = fetch_investor_flow_pykrx("005930", "2026-06-24", "2026-06-25", stock_module=_Empty())

    assert out.empty
    assert list(out.columns) == ["Date", "foreign_net_buy", "institution_net_buy"]
