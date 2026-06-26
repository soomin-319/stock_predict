from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import pandas as pd

_FOREIGN_KEYS = ("외국인합계", "외국인")
_INSTITUTION_KEYS = ("기관합계", "기관")
_OUT_COLUMNS = ["Date", "foreign_net_buy", "institution_net_buy"]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.Series(dtype="datetime64[ns]"),
            "foreign_net_buy": pd.Series(dtype="float64"),
            "institution_net_buy": pd.Series(dtype="float64"),
        },
        columns=_OUT_COLUMNS,
    )


def _iter_dotenv_candidates(search_roots=None):
    roots = list(search_roots or [])
    roots.append(Path.cwd())
    roots.extend(Path(__file__).resolve().parents)
    seen: set[Path] = set()
    for root in roots:
        path = Path(root).resolve() / ".env"
        if path in seen:
            continue
        seen.add(path)
        yield path


def _strip_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _load_krx_credentials_from_dotenv(*, search_roots=None) -> None:
    missing = {key for key in ("KRX_ID", "KRX_PW") if not os.environ.get(key)}
    if not missing:
        return
    for dotenv in _iter_dotenv_candidates(search_roots):
        try:
            lines = dotenv.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, raw_value = stripped.split("=", 1)
            name = name.strip()
            if name in missing and not os.environ.get(name):
                os.environ[name] = _strip_env_value(raw_value)
        missing = {key for key in missing if not os.environ.get(key)}
        if not missing:
            return


def _pick_column(columns, keys: tuple[str, ...]):
    for key in keys:
        for col in columns:
            if key in str(col):
                return col
    return None


def _get_pykrx_stock():
    _load_krx_credentials_from_dotenv()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
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
        return _empty_frame()

    columns = list(raw.columns)
    foreign_col = _pick_column(columns, _FOREIGN_KEYS)
    institution_col = _pick_column(columns, _INSTITUTION_KEYS)
    if foreign_col is None or institution_col is None:
        raise ValueError(f"required investor flow columns not found: {columns}")
    foreign = pd.to_numeric(raw[foreign_col], errors="coerce")
    institution = pd.to_numeric(raw[institution_col], errors="coerce")
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
