from __future__ import annotations

import pandas as pd

_FOREIGN_KEYS = ("외국인합계", "외국인")
_INSTITUTION_KEYS = ("기관합계", "기관")
_OUT_COLUMNS = ["Date", "foreign_net_buy", "institution_net_buy"]


def _pick_column(columns, keys: tuple[str, ...]):
    for key in keys:
        for col in columns:
            if key in str(col):
                return col
    return None


def _get_pykrx_stock():
    try:
        from pykrx import stock
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("pykrx is required for investor flow fetch; run `pip install pykrx`.") from exc
    return stock


def fetch_investor_flow_pykrx(
    ticker: str,
    start: str,
    end: str,
    *,
    stock_module=None,
) -> pd.DataFrame:
    """Fetch daily KRX foreign/institution net-buy trading value for one ticker."""
    stock = stock_module or _get_pykrx_stock()
    fromdate = pd.to_datetime(start).strftime("%Y%m%d")
    todate = pd.to_datetime(end).strftime("%Y%m%d")
    raw = stock.get_market_trading_value_by_date(fromdate, todate, ticker)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=_OUT_COLUMNS)

    columns = list(raw.columns)
    foreign_col = _pick_column(columns, _FOREIGN_KEYS)
    institution_col = _pick_column(columns, _INSTITUTION_KEYS)
    foreign = pd.to_numeric(raw[foreign_col], errors="coerce") if foreign_col else pd.Series(0.0, index=raw.index)
    institution = (
        pd.to_numeric(raw[institution_col], errors="coerce")
        if institution_col
        else pd.Series(0.0, index=raw.index)
    )
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw.index),
            "foreign_net_buy": foreign.to_numpy(dtype="float64"),
            "institution_net_buy": institution.to_numpy(dtype="float64"),
        }
    )
    out["foreign_net_buy"] = out["foreign_net_buy"].fillna(0.0)
    out["institution_net_buy"] = out["institution_net_buy"].fillna(0.0)
    return out.reset_index(drop=True)
