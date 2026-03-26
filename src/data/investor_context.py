from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
    out: dict[str, str] = {}
    for r in df.itertuples(index=False):
        sym = str(r.Symbol)
        corp = str(r.corp_code)
        out[sym] = corp
        ticker = _symbol_to_ticker(sym)
        if ticker:
            out[ticker] = corp
    return out


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


def _dart_items(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    status = str(payload.get("status", "")).strip()
    if status and status != "000":
        message = str(payload.get("message", "")).strip()
        print(f"[INVESTOR_CONTEXT] DART 응답 비정상 status={status} message={message}")
        return []
    items = payload.get("list", [])
    return items if isinstance(items, list) else []


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
            items = _dart_items(payload)
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
    # 운영 안정성을 위해 최근 구간(지연 없는 실시간 구간)은 비활성화합니다.
    end_dt = pd.to_datetime(end, errors="coerce")
    cutoff = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=30)
    if pd.isna(end_dt) or end_dt.normalize() >= cutoff:
        coverage = {"requested": 0, "successful": 0, "failed": 0}
        return pd.DataFrame(
            columns=["Date", "Symbol", "news_sentiment", "news_relevance_score", "news_impact_score", "news_article_count"]
        ), coverage

    coverage = {"requested": len(symbols), "successful": 0, "failed": 0}
    start_dt = pd.to_datetime(start, errors="coerce").normalize()
    end_dt = end_dt.normalize()
    rows: list[dict] = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            items = getattr(ticker, "news", []) or []
        except Exception:
            coverage["failed"] += 1
            continue

        per_day: dict[pd.Timestamp, list[tuple[float, float, float]]] = {}
        seen_titles: set[str] = set()
        for item in items:
            title = " ".join(str(item.get("title", "")).split())
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            pub_raw = item.get("providerPublishTime")
            pub_dt = pd.to_datetime(pub_raw, unit="s", errors="coerce")
            if pd.isna(pub_dt):
                continue
            pub_day = pub_dt.normalize()
            if pub_day < start_dt or pub_day > end_dt:
                continue
            sentiment, relevance, impact = _headline_news_features(title, cfg=cfg)
            per_day.setdefault(pub_day, []).append((sentiment, relevance, impact))

        if not per_day:
            coverage["failed"] += 1
            continue

        for day, vals in per_day.items():
            s = pd.Series([v[0] for v in vals], dtype=float)
            r = pd.Series([v[1] for v in vals], dtype=float)
            i = pd.Series([v[2] for v in vals], dtype=float)
            rows.append(
                {
                    "Date": day,
                    "Symbol": symbol,
                    "news_sentiment": float(s.mean()),
                    "news_relevance_score": float(r.mean()),
                    "news_impact_score": float(i.mean()),
                    "news_article_count": int(len(vals)),
                }
            )
        coverage["successful"] += 1

    if not rows:
        return pd.DataFrame(
            columns=["Date", "Symbol", "news_sentiment", "news_relevance_score", "news_impact_score", "news_article_count"]
        ), coverage
    out = pd.DataFrame(rows).groupby(["Date", "Symbol"], as_index=False).agg(
        news_sentiment=("news_sentiment", "mean"),
        news_relevance_score=("news_relevance_score", "mean"),
        news_impact_score=("news_impact_score", "mean"),
        news_article_count=("news_article_count", "sum"),
    )
    return out, coverage


def _score_headline_with_openai(headline: str, cfg: InvestorContextConfig) -> tuple[float, float, float]:
    _ = (headline, cfg)
    return 0.0, 0.0, 0.0


def _headline_news_features(headline: str, cfg: InvestorContextConfig | None = None) -> tuple[float, float, float]:
    title = str(headline or "").strip()
    if not title:
        return 0.0, 0.0, 0.0

    use_ai = (
        cfg is not None
        and cfg.news_scoring_mode == "ai"
        and bool(cfg.openai_api_key)
        and bool(cfg.openai_model)
    )
    if use_ai:
        return _score_headline_with_openai(title, cfg)

    pos_kw = {"공급계약", "수주", "실적", "개선", "성장", "호조", "증가", "상향", "흑자"}
    neg_kw = {"하향", "적자", "감소", "소송", "리콜", "부진", "경고", "악화"}
    rel_kw = {"공급계약", "실적", "공시", "가이던스", "매출", "영업이익", "수주", "주가", "전망"}
    impact_kw = {"대규모", "급등", "급락", "M&A", "합병", "신사업", "수출", "규제"}

    pos_score = sum(1 for kw in pos_kw if kw in title)
    neg_score = sum(1 for kw in neg_kw if kw in title)
    rel_score = sum(1 for kw in rel_kw if kw in title)
    imp_score = sum(1 for kw in impact_kw if kw in title)

    sentiment = 0.5 + 0.18 * pos_score - 0.18 * neg_score
    relevance = min(1.0, 0.1 + 0.25 * rel_score + 0.1 * max(pos_score, 0))
    impact = min(1.0, 0.1 + 0.25 * imp_score + 0.15 * (1 if "공급계약" in title else 0))
    return float(max(0.0, min(1.0, sentiment))), float(max(0.0, relevance)), float(max(0.0, impact))


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
    start_date = pd.to_datetime(start_dt).normalize()
    end_date = pd.to_datetime(end_dt).normalize()
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
            pub_kst = pub_dt.tz_convert("Asia/Seoul")
            pub_date = pub_kst.tz_localize(None).normalize()
            if pub_date < start_date or pub_date > end_date:
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
                    "Date": pub_date.strftime("%Y-%m-%d"),
                    "Symbol": symbol,
                    "source_type": "news",
                    "title": title,
                    "body": description,
                    "published_at": pub_kst.isoformat(),
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
                items = _dart_items(payload)
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
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out[(out["Date"] >= start_dt.normalize()) & (out["Date"] <= end_dt.normalize())].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.sort_values(["Date", "Symbol", "source_type", "title"]).reset_index(drop=True)
    return out
