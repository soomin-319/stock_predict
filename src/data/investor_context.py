from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yfinance as yf


POSITIVE_NEWS_KEYWORDS = {
    "beat", "surge", "record", "upgrade", "partnership", "contract", "approval", "growth",
    "호재", "수주", "실적개선", "상향", "승인", "증가", "신고가",
}
NEGATIVE_NEWS_KEYWORDS = {
    "miss", "drop", "downgrade", "lawsuit", "delay", "fraud", "decline",
    "악재", "소송", "지연", "하향", "감소", "적자",
}


@dataclass
class InvestorContextConfig:
    enabled: bool = False
    enable_disclosure: bool = True
    enable_news: bool = True
    enable_flow: bool = True
    dart_api_key: str | None = None
    dart_corp_map_csv: str | None = None


def _symbol_to_ticker(symbol: str) -> str | None:
    s = str(symbol)
    m = re.match(r"(\d{6})", s)
    return m.group(1) if m else None


def _empty_context(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    for c in ["foreign_net_buy", "institution_net_buy", "disclosure_score", "news_sentiment"]:
        if c not in out.columns:
            out[c] = 0.0
    return out, {
        "enabled": False,
        "flow": {"requested": 0, "successful": 0, "failed": 0},
        "disclosure": {"requested": 0, "successful": 0, "failed": 0},
        "news": {"requested": 0, "successful": 0, "failed": 0},
    }


def _fetch_flow_pykrx(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    coverage = {"requested": len(symbols), "successful": 0, "failed": 0}
    try:
        from pykrx import stock
    except Exception:
        coverage["failed"] = len(symbols)
        return pd.DataFrame(columns=["Date", "Symbol", "foreign_net_buy", "institution_net_buy"]), coverage

    rows = []
    for symbol in symbols:
        ticker = _symbol_to_ticker(symbol)
        if not ticker:
            coverage["failed"] += 1
            continue
        try:
            frame = stock.get_market_trading_value_by_date(start.replace("-", ""), end.replace("-", ""), ticker)
            if frame is None or frame.empty:
                coverage["failed"] += 1
                continue
            frame = frame.reset_index().rename(columns={"날짜": "Date"})
            if "기관합계" not in frame.columns or "외국인합계" not in frame.columns:
                coverage["failed"] += 1
                continue
            part = pd.DataFrame(
                {
                    "Date": pd.to_datetime(frame["Date"]),
                    "Symbol": symbol,
                    "foreign_net_buy": pd.to_numeric(frame["외국인합계"], errors="coerce"),
                    "institution_net_buy": pd.to_numeric(frame["기관합계"], errors="coerce"),
                }
            )
            rows.append(part)
            coverage["successful"] += 1
        except Exception:
            coverage["failed"] += 1

    if not rows:
        return pd.DataFrame(columns=["Date", "Symbol", "foreign_net_buy", "institution_net_buy"]), coverage
    return pd.concat(rows, ignore_index=True), coverage


def _load_dart_corp_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "Symbol" not in df.columns or "corp_code" not in df.columns:
        return {}
    return {str(r.Symbol): str(r.corp_code) for r in df.itertuples(index=False)}


def _dart_list(api_key: str, corp_code: str, start: str, end: str):
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bgn_de": start.replace("-", ""),
        "end_de": end.replace("-", ""),
        "page_no": 1,
        "page_count": 100,
    }
    url = "https://opendart.fss.or.kr/api/list.json?" + urlencode(params)
    with urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data


def _fetch_disclosure_scores(symbols: list[str], start: str, end: str, api_key: str | None, corp_map_csv: str | None):
    coverage = {"requested": len(symbols), "successful": 0, "failed": 0}
    if not api_key:
        coverage["failed"] = len(symbols)
        return pd.DataFrame(columns=["Date", "Symbol", "disclosure_score"]), coverage

    corp_map = _load_dart_corp_map(corp_map_csv)
    rows = []
    positive_kw = {"수주", "계약", "증가", "실적", "합병", "approval", "contract", "growth"}

    for symbol in symbols:
        corp = corp_map.get(symbol)
        if not corp:
            coverage["failed"] += 1
            continue
        try:
            payload = _dart_list(api_key, corp, start, end)
            items = payload.get("list", []) if isinstance(payload, dict) else []
            if not items:
                coverage["failed"] += 1
                continue
            recs = []
            for item in items:
                d = item.get("rcept_dt")
                if not d:
                    continue
                dt = datetime.strptime(d, "%Y%m%d")
                name = str(item.get("report_nm", ""))
                score = 0.2
                if any(k in name for k in positive_kw):
                    score += 0.3
                recs.append((dt, min(score, 1.0)))
            if not recs:
                coverage["failed"] += 1
                continue
            part = pd.DataFrame(recs, columns=["Date", "disclosure_score"]).groupby("Date", as_index=False).sum()
            part["disclosure_score"] = part["disclosure_score"].clip(0.0, 1.0)
            part["Symbol"] = symbol
            rows.append(part)
            coverage["successful"] += 1
        except Exception:
            coverage["failed"] += 1

    if not rows:
        return pd.DataFrame(columns=["Date", "Symbol", "disclosure_score"]), coverage
    return pd.concat(rows, ignore_index=True), coverage


def _headline_sentiment(text: str) -> float:
    t = str(text).lower()
    pos = sum(1 for k in POSITIVE_NEWS_KEYWORDS if k in t)
    neg = sum(1 for k in NEGATIVE_NEWS_KEYWORDS if k in t)
    score = 0.5 + 0.2 * (pos - neg)
    return max(0.0, min(1.0, score))


def _fetch_news_sentiment(symbols: list[str], start: str, end: str):
    coverage = {"requested": len(symbols), "successful": 0, "failed": 0}
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    rows = []

    for symbol in symbols:
        try:
            items = yf.Ticker(symbol).news or []
            if not items:
                coverage["failed"] += 1
                continue
            recs = []
            for it in items:
                ts = it.get("providerPublishTime")
                title = it.get("title", "")
                if ts is None:
                    continue
                dt = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize()
                if dt < start_dt or dt > end_dt:
                    continue
                recs.append((dt, _headline_sentiment(title)))
            if not recs:
                coverage["failed"] += 1
                continue
            part = pd.DataFrame(recs, columns=["Date", "news_sentiment"]).groupby("Date", as_index=False).mean()
            part["Symbol"] = symbol
            rows.append(part)
            coverage["successful"] += 1
        except Exception:
            coverage["failed"] += 1

    if not rows:
        return pd.DataFrame(columns=["Date", "Symbol", "news_sentiment"]), coverage
    return pd.concat(rows, ignore_index=True), coverage


def add_investor_context_with_coverage(df: pd.DataFrame, cfg: InvestorContextConfig) -> tuple[pd.DataFrame, dict]:
    if df.empty or not cfg.enabled:
        return _empty_context(df)

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    symbols = sorted(out["Symbol"].dropna().astype(str).unique().tolist())
    start = out["Date"].min().strftime("%Y-%m-%d")
    end = out["Date"].max().strftime("%Y-%m-%d")

    coverage = {
        "enabled": True,
        "flow": {"requested": 0, "successful": 0, "failed": 0},
        "disclosure": {"requested": 0, "successful": 0, "failed": 0},
        "news": {"requested": 0, "successful": 0, "failed": 0},
    }

    if cfg.enable_flow:
        flow_df, flow_cov = _fetch_flow_pykrx(symbols, start, end)
        coverage["flow"] = flow_cov
        if not flow_df.empty:
            out = out.merge(flow_df, on=["Date", "Symbol"], how="left")

    if cfg.enable_disclosure:
        disc_df, disc_cov = _fetch_disclosure_scores(symbols, start, end, cfg.dart_api_key, cfg.dart_corp_map_csv)
        coverage["disclosure"] = disc_cov
        if not disc_df.empty:
            out = out.merge(disc_df, on=["Date", "Symbol"], how="left")

    if cfg.enable_news:
        news_df, news_cov = _fetch_news_sentiment(symbols, start, end)
        coverage["news"] = news_cov
        if not news_df.empty:
            out = out.merge(news_df, on=["Date", "Symbol"], how="left")

    for c in ["foreign_net_buy", "institution_net_buy", "disclosure_score", "news_sentiment"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out, coverage
