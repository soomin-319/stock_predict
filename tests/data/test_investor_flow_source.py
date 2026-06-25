import builtins
import os
from types import SimpleNamespace

import pandas as pd

from src.data import investor_flow_source as flow
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


def test_loads_krx_credentials_from_dotenv_before_pykrx_import(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "KRX_ID=test-user",
                "KRX_PW='test password'",
                "OPENAI_API_KEY=ignored",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("KRX_ID", raising=False)
    monkeypatch.delenv("KRX_PW", raising=False)

    flow._load_krx_credentials_from_dotenv(search_roots=[tmp_path])

    assert os.environ["KRX_ID"] == "test-user"
    assert os.environ["KRX_PW"] == "test password"
    assert "OPENAI_API_KEY" not in os.environ


def test_get_pykrx_stock_suppresses_login_stdout(monkeypatch, capsys):
    fake_stock = object()
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pykrx":
            print("KRX login ID: secret-user")
            return SimpleNamespace(stock=fake_stock)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(flow, "_load_krx_credentials_from_dotenv", lambda: None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert flow._get_pykrx_stock() is fake_stock
    assert "secret-user" not in capsys.readouterr().out
