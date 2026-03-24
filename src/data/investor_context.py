from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

from src.data.pykrx_support import import_pykrx_stock




@dataclass
class InvestorContextConfig:
    enabled: bool = False
    enable_disclosure: bool = True
    enable_news: bool = True
    enable_flow: bool = True
    dart_api_key: str | None = None
    dart_corp_map_csv: str | None = None
    news_scoring_mode: str = "auto"
    openai_api_key: str | None = None
    openai_model: str | None = None


def _symbol_to_ticker(symbol: str) -> str | None:
    s = str(symbol)
    m = re.match(r"(\d{6})", s)
    return m.group(1) if m else None


def _empty_context(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    for c in [
        "foreign_net_buy",
        "institution_net_buy",
        "disclosure_score",
        "news_sentiment",
        "news_relevance_score",
        "news_impact_score",
        "news_article_count",
    ]:
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
    stock = import_pykrx_stock()
    if stock is None:
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


def _normalize_news_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title).strip()).lower()


def _fetch_news_sentiment(symbols: list[str], start: str, end: str, cfg: InvestorContextConfig | None = None):
    _ = cfg
    coverage = {"requested": len(symbols), "successful": 0, "failed": 0}
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

    for symbol in symbols:
        try:
            items = _load_yfinance_news_items(symbol)
            if not items:
                coverage["failed"] += 1
                continue
            seen_titles: set[tuple[pd.Timestamp, str]] = set()
            for it in items:
                dt, title = _parse_news_datetime_and_title(it)
                if dt is None:
                    continue
                if dt < start_dt or dt > end_dt:
                    continue
                normalized_title = _normalize_news_title(title)
                if not normalized_title:
                    continue
                dedupe_key = (dt, normalized_title)
                if dedupe_key in seen_titles:
                    continue
                seen_titles.add(dedupe_key)
            if not seen_titles:
                coverage["failed"] += 1
                continue
            coverage["successful"] += 1
        except Exception:
            coverage["failed"] += 1

    return pd.DataFrame(
        columns=["Date", "Symbol", "news_sentiment", "news_relevance_score", "news_impact_score", "news_article_count"]
    ), coverage


def _load_yfinance_news_items(symbol: str) -> list[dict]:
    ticker = yf.Ticker(symbol)
    merged: list[dict] = []

    direct_news = getattr(ticker, "news", None)
    if isinstance(direct_news, list):
        merged.extend(item for item in direct_news if isinstance(item, dict))

    get_news = getattr(ticker, "get_news", None)
    if callable(get_news):
        for kwargs in ({}, {"count": 100}):
            try:
                fetched = get_news(**kwargs)
            except TypeError:
                continue
            except Exception:
                break
            if isinstance(fetched, list):
                merged.extend(item for item in fetched if isinstance(item, dict))
                if fetched:
                    break

    unique: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in merged:
        dt, title = _parse_news_datetime_and_title(item)
        key = (str(dt), _normalize_news_title(title))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(item)
    return unique


def _parse_news_datetime_and_title(item: dict) -> tuple[pd.Timestamp | None, str]:
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = str(
        content.get("title")
        or content.get("headline")
        or item.get("title")
        or item.get("headline")
        or ""
    ).strip()

    ts_value = (
        content.get("providerPublishTime")
        or content.get("pubDate")
        or content.get("published")
        or content.get("publish_time")
        or item.get("providerPublishTime")
        or item.get("pubDate")
        or item.get("published")
        or item.get("publish_time")
    )
    if ts_value is None:
        return None, title

    if isinstance(ts_value, (int, float)):
        unit = "ms" if float(ts_value) > 1_000_000_000_000 else "s"
        dt = pd.to_datetime(ts_value, unit=unit, utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(dt):
        return None, title
    return dt.tz_localize(None).normalize(), title


def _load_yfinance_news_items(symbol: str) -> list[dict]:
    ticker = yf.Ticker(symbol)
    merged: list[dict] = []

    direct_news = getattr(ticker, "news", None)
    if isinstance(direct_news, list):
        merged.extend(item for item in direct_news if isinstance(item, dict))

    get_news = getattr(ticker, "get_news", None)
    if callable(get_news):
        for kwargs in ({}, {"count": 100}):
            try:
                fetched = get_news(**kwargs)
            except TypeError:
                continue
            except Exception:
                break
            if isinstance(fetched, list):
                merged.extend(item for item in fetched if isinstance(item, dict))
                if fetched:
                    break

    unique: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in merged:
        dt, title = _parse_news_datetime_and_title(item)
        key = (str(dt), _normalize_news_title(title))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(item)
    return unique


def _parse_news_datetime_and_title(item: dict) -> tuple[pd.Timestamp | None, str]:
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = str(
        content.get("title")
        or content.get("headline")
        or item.get("title")
        or item.get("headline")
        or ""
    ).strip()

    ts_value = (
        content.get("providerPublishTime")
        or content.get("pubDate")
        or content.get("published")
        or content.get("publish_time")
        or item.get("providerPublishTime")
        or item.get("pubDate")
        or item.get("published")
        or item.get("publish_time")
    )
    if ts_value is None:
        return None, title

    if isinstance(ts_value, (int, float)):
        unit = "ms" if float(ts_value) > 1_000_000_000_000 else "s"
        dt = pd.to_datetime(ts_value, unit=unit, utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(dt):
        return None, title
    return dt.tz_localize(None).normalize(), title


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
        news_df, news_cov = _fetch_news_sentiment(symbols, start, end, cfg=cfg)
        coverage["news"] = news_cov
        if not news_df.empty:
            out = out.merge(news_df, on=["Date", "Symbol"], how="left")

    for c in [
        "foreign_net_buy",
        "institution_net_buy",
        "disclosure_score",
        "news_sentiment",
        "news_relevance_score",
        "news_impact_score",
        "news_article_count",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out, coverage


def collect_context_raw_events(
    symbols: list[str],
    start: str,
    end: str,
    dart_api_key: str | None = None,
    dart_corp_map_csv: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

    for symbol in symbols:
        try:
            news_items = _load_yfinance_news_items(symbol)
        except Exception:
            news_items = []
        for item in news_items:
            dt, title = _parse_news_datetime_and_title(item)
            if dt is None or dt < start_dt or dt > end_dt:
                continue
            content = item.get("content") if isinstance(item.get("content"), dict) else item
            provider = str(
                content.get("provider")
                or content.get("publisher")
                or item.get("provider")
                or item.get("publisher")
                or "yfinance"
            ).strip()
            url = (
                content.get("link")
                or content.get("url")
                or item.get("link")
                or item.get("url")
                or ""
            )
            if isinstance(content.get("canonicalUrl"), dict):
                url = content.get("canonicalUrl", {}).get("url") or url
            rows.append(
                {
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Symbol": symbol,
                    "source_type": "news",
                    "title": title,
                    "published_at": dt.isoformat(),
                    "provider": provider,
                    "url": str(url or ""),
                    "raw_id": str(item.get("id") or ""),
                }
            )

    if dart_api_key:
        corp_map = _load_dart_corp_map(dart_corp_map_csv)
        for symbol in symbols:
            corp = corp_map.get(symbol)
            if not corp:
                continue
            try:
                payload = _dart_list(dart_api_key, corp, start, end)
                items = payload.get("list", []) if isinstance(payload, dict) else []
            except Exception:
                items = []
            for item in items:
                rcept_dt = str(item.get("rcept_dt") or "").strip()
                if len(rcept_dt) != 8:
                    continue
                dt = pd.to_datetime(rcept_dt, format="%Y%m%d", errors="coerce")
                if pd.isna(dt):
                    continue
                dt = dt.normalize()
                if dt < start_dt or dt > end_dt:
                    continue
                rcept_no = str(item.get("rcept_no") or "").strip()
                dart_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}" if rcept_no else ""
                rows.append(
                    {
                        "Date": dt.strftime("%Y-%m-%d"),
                        "Symbol": symbol,
                        "source_type": "disclosure",
                        "title": str(item.get("report_nm") or "").strip(),
                        "published_at": dt.isoformat(),
                        "provider": "DART",
                        "url": dart_url,
                        "raw_id": rcept_no,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["Date", "Symbol", "source_type", "title", "published_at", "provider", "url", "raw_id"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["Date", "Symbol", "source_type", "title", "raw_id"]).reset_index(drop=True)
    out = out.sort_values(["Date", "Symbol", "source_type", "title"]).reset_index(drop=True)
    return out
