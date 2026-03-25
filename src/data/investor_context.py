from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

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
    naver_client_id: str | None = None
    naver_client_secret: str | None = None


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


def _fetch_news_sentiment(symbols: list[str], start: str, end: str, cfg: InvestorContextConfig | None = None):
    _ = (symbols, start, end, cfg)
    # 뉴스 점수화/뉴스 수집 기능은 제거되었습니다.
    # 컬럼 호환성 유지를 위해 빈 프레임과 0-coverage를 반환합니다.
    coverage = {"requested": 0, "successful": 0, "failed": 0}
    return pd.DataFrame(
        columns=["Date", "Symbol", "news_sentiment", "news_relevance_score", "news_impact_score", "news_article_count"]
    ), coverage


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


def _build_news_queries(symbol_name: str) -> list[str]:
    name = str(symbol_name or "").strip()
    if not name:
        return []
    keywords = ["주가", "실적", "공시", "계약", "전망"]
    return [name] + [f"{name} {kw}" for kw in keywords]


def _fetch_naver_news_items(
    *,
    symbol: str,
    symbol_name: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    client_id: str | None,
    client_secret: str | None,
) -> list[dict]:
    if not client_id or not client_secret:
        return []
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for query in _build_news_queries(symbol_name):
        params = urlencode({"query": query, "display": 50, "start": 1, "sort": "date"})
        req = Request(
            f"https://openapi.naver.com/v1/search/news.json?{params}",
            headers={
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret,
            },
        )
        try:
            with urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue

        for item in payload.get("items", []) if isinstance(payload, dict) else []:
            title = _strip_html(item.get("title", ""))
            description = _strip_html(item.get("description", ""))
            pub_raw = str(item.get("pubDate", "")).strip()
            pub_dt = pd.to_datetime(pub_raw, utc=True, errors="coerce")
            if pd.isna(pub_dt):
                continue
            pub_dt = pub_dt.tz_localize(None)
            if pub_dt < start_dt or pub_dt > (end_dt + pd.Timedelta(days=1)):
                continue
            if symbol_name and symbol_name not in f"{title} {description}":
                continue
            origin = str(item.get("originallink") or item.get("link") or "").strip()
            dedupe_key = (origin, title)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append(
                {
                    "Date": pub_dt.normalize().strftime("%Y-%m-%d"),
                    "Symbol": symbol,
                    "source_type": "news",
                    "title": title,
                    "body": description,
                    "published_at": pub_dt.isoformat(),
                    "provider": "naver_news_api",
                    "url": str(item.get("originallink") or item.get("link") or ""),
                    "raw_id": origin,
                    "query": query,
                    "symbol_name": symbol_name,
                    "source": "naver_news_api",
                }
            )
    return rows


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
    symbol_name_map: dict[str, str] | None = None,
    naver_client_id: str | None = None,
    naver_client_secret: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

    for symbol in symbols:
        symbol_name = (symbol_name_map or {}).get(symbol, "")
        rows.extend(
            _fetch_naver_news_items(
                symbol=symbol,
                symbol_name=symbol_name,
                start_dt=start_dt,
                end_dt=end_dt,
                client_id=naver_client_id,
                client_secret=naver_client_secret,
            )
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
                        "body": "",
                        "published_at": dt.isoformat(),
                        "provider": "DART",
                        "url": dart_url,
                        "raw_id": rcept_no,
                        "query": "",
                        "symbol_name": (symbol_name_map or {}).get(symbol, ""),
                        "source": "dart",
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Date",
                "Symbol",
                "source_type",
                "title",
                "body",
                "published_at",
                "provider",
                "url",
                "raw_id",
                "query",
                "symbol_name",
                "source",
            ]
        )

    out = pd.DataFrame(rows).drop_duplicates(subset=["Date", "Symbol", "source_type", "title", "raw_id"]).reset_index(drop=True)
    out = out.sort_values(["Date", "Symbol", "source_type", "title"]).reset_index(drop=True)
    return out
