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


POSITIVE_NEWS_KEYWORDS = {
    "beat", "surge", "record", "upgrade", "partnership", "contract", "approval", "growth",
    "호재", "수주", "실적개선", "상향", "승인", "증가", "신고가",
}
NEGATIVE_NEWS_KEYWORDS = {
    "miss", "drop", "downgrade", "lawsuit", "delay", "fraud", "decline",
    "악재", "소송", "지연", "하향", "감소", "적자",
}
STRONG_POSITIVE_NEWS_KEYWORDS = {
    "흑자전환", "어닝서프라이즈", "대규모", "공급계약", "수주", "자사주", "소각", "승인", "record", "beat",
}
STRONG_NEGATIVE_NEWS_KEYWORDS = {
    "유상증자", "전환사채", "bw", "cb", "감사의견", "거절", "횡령", "배임", "하한가", "lawsuit", "fraud",
}
PRICE_IMPACT_NEWS_KEYWORDS = {
    "실적", "가이던스", "수주", "공급계약", "계약", "승인", "허가", "합병", "인수", "매각",
    "유상증자", "무상증자", "전환사채", "bw", "cb", "배당", "자사주", "소각", "최대주주",
    "소송", "횡령", "배임", "감사의견", "거래정지", "단기과열", "투자경고", "투자위험",
    "earnings", "guidance", "contract", "approval", "acquisition", "lawsuit",
}
LOW_SIGNAL_NEWS_KEYWORDS = {
    "market wrap", "preview", "opinion", "column", "브리핑", "장마감", "장전시황", "시황", "리포트 요약",
}
UNCERTAINTY_NEWS_KEYWORDS = {
    "검토", "추진", "가능성", "예정", "설", "rumor", "reportedly", "may", "could",
}


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


def _headline_sentiment(text: str) -> float:
    t = str(text).lower()
    pos = sum(1 for k in POSITIVE_NEWS_KEYWORDS if k in t)
    neg = sum(1 for k in NEGATIVE_NEWS_KEYWORDS if k in t)
    strong_pos = sum(1 for k in STRONG_POSITIVE_NEWS_KEYWORDS if k in t)
    strong_neg = sum(1 for k in STRONG_NEGATIVE_NEWS_KEYWORDS if k in t)
    uncertainty = sum(1 for k in UNCERTAINTY_NEWS_KEYWORDS if k in t)
    score = 0.5 + 0.12 * (pos - neg) + 0.18 * (strong_pos - strong_neg) - 0.05 * uncertainty
    return max(0.0, min(1.0, score))


def _headline_relevance(text: str) -> float:
    t = str(text).lower()
    impact_hits = sum(1 for k in PRICE_IMPACT_NEWS_KEYWORDS if k in t)
    low_signal_hits = sum(1 for k in LOW_SIGNAL_NEWS_KEYWORDS if k in t)
    uncertainty_hits = sum(1 for k in UNCERTAINTY_NEWS_KEYWORDS if k in t)

    score = 0.25 + 0.25 * min(impact_hits, 2) - 0.15 * low_signal_hits - 0.05 * uncertainty_hits
    return max(0.0, min(1.0, score))


def _headline_news_features_rule_based(text: str) -> tuple[float, float, float]:
    sentiment = _headline_sentiment(text)
    relevance = _headline_relevance(text)
    weighted_sentiment = 0.5 + (sentiment - 0.5) * relevance
    impact = (weighted_sentiment - 0.5) * 2.0
    return weighted_sentiment, relevance, impact


def _normalize_news_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title).strip()).lower()


def _resolve_news_ai_settings(cfg: InvestorContextConfig | None) -> tuple[str, str | None, str | None]:
    if cfg is None:
        mode = os.getenv("NEWS_SCORING_MODE", "auto")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")
        return mode.lower(), api_key, model

    mode = str(cfg.news_scoring_mode or os.getenv("NEWS_SCORING_MODE", "auto")).lower()
    api_key = cfg.openai_api_key if cfg.openai_api_key is not None else os.getenv("OPENAI_API_KEY")
    model = cfg.openai_model if cfg.openai_model is not None else os.getenv("OPENAI_MODEL")
    return mode, api_key, model


def _score_headline_with_openai(title: str, api_key: str | None, model: str | None) -> tuple[float, float, float] | None:
    if not api_key or not model or not str(title).strip():
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    prompt = (
        "You are a financial news analyst. Read the stock-news headline and score its likely short-term price impact "
        "(1-5 trading days) for the referenced company. Return JSON only with these numeric keys: "
        "sentiment_score, relevance_score, impact_score. "
        "sentiment_score must be between 0 and 1 where 0 is strongly bearish, 0.5 is neutral, and 1 is strongly bullish. "
        "relevance_score must be between 0 and 1 and represent how directly the headline should affect the company's stock price. "
        "impact_score must be between -1 and 1 and represent the expected stock-price impact direction and magnitude. "
        "If the headline is ambiguous, lower relevance_score and keep sentiment_score near 0.5."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": str(title)}]},
            ],
        )
        raw = getattr(response, "output_text", "") or ""
        payload = json.loads(raw)
        sentiment = float(payload["sentiment_score"])
        relevance = float(payload["relevance_score"])
        impact = float(payload["impact_score"])
    except Exception:
        return None

    sentiment = max(0.0, min(1.0, sentiment))
    relevance = max(0.0, min(1.0, relevance))
    impact = max(-1.0, min(1.0, impact))
    return sentiment, relevance, impact


def _headline_news_features(text: str, cfg: InvestorContextConfig | None = None) -> tuple[float, float, float]:
    mode, api_key, model = _resolve_news_ai_settings(cfg)
    if mode in {"auto", "ai"}:
        ai_result = _score_headline_with_openai(text, api_key=api_key, model=model)
        if ai_result is not None:
            return ai_result
    return _headline_news_features_rule_based(text)


def _fetch_news_sentiment(symbols: list[str], start: str, end: str, cfg: InvestorContextConfig | None = None):
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
            seen_titles: set[tuple[pd.Timestamp, str]] = set()
            for it in items:
                ts = it.get("providerPublishTime")
                title = it.get("title", "")
                if ts is None:
                    continue
                dt = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).normalize()
                if dt < start_dt or dt > end_dt:
                    continue
                normalized_title = _normalize_news_title(title)
                if not normalized_title:
                    continue
                dedupe_key = (dt, normalized_title)
                if dedupe_key in seen_titles:
                    continue
                seen_titles.add(dedupe_key)
                sentiment, relevance, impact = _headline_news_features(title, cfg=cfg)
                recs.append((dt, sentiment, relevance, impact, 1))
            if not recs:
                coverage["failed"] += 1
                continue
            part = (
                pd.DataFrame(
                    recs,
                    columns=["Date", "news_sentiment", "news_relevance_score", "news_impact_score", "news_article_count"],
                )
                .groupby("Date", as_index=False)
                .agg(
                    {
                        "news_sentiment": "mean",
                        "news_relevance_score": "mean",
                        "news_impact_score": "mean",
                        "news_article_count": "sum",
                    }
                )
            )
            part["Symbol"] = symbol
            rows.append(part)
            coverage["successful"] += 1
        except Exception:
            coverage["failed"] += 1

    if not rows:
        return pd.DataFrame(
            columns=["Date", "Symbol", "news_sentiment", "news_relevance_score", "news_impact_score", "news_article_count"]
        ), coverage
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
