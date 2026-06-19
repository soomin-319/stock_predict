from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd



@dataclass
class InvestorContextConfig:
    enabled: bool = False
    enable_disclosure: bool = True
    dart_api_key: str | None = None
    dart_corp_map_csv: str | None = None
    raw_event_n_jobs: int = 4


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


def _fetch_flow(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    coverage = {
        "requested": len(symbols),
        "successful": 0,
        "failed": 0,
        "status": "not_configured",
        "source": "input_csv_only",
        "message": "Investor flow source is not configured; using input CSV values only.",
    }
    _ = (start, end)
    return pd.DataFrame(columns=["Date", "Symbol", "foreign_net_buy", "institution_net_buy"]), coverage


def _load_dart_corp_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
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
    errors: list[dict] | None = None,
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
        except Exception as exc:
            if errors is not None:
                errors.append(
                    {
                        "source": "naver_news_api",
                        "symbol": symbol,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
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

    flow_df, flow_cov = _fetch_flow(symbols, start, end)
    coverage["flow"] = flow_cov
    if not flow_df.empty:
        out = out.merge(flow_df, on=["Date", "Symbol"], how="left")

    if cfg.enable_disclosure:
        disc_df, disc_cov = _fetch_disclosure_scores(symbols, start, end, cfg.dart_api_key, cfg.dart_corp_map_csv)
        coverage["disclosure"] = disc_cov
        if not disc_df.empty:
            out = out.merge(disc_df, on=["Date", "Symbol"], how="left")

    coverage["news"] = {"requested": 0, "successful": 0, "failed": 0}

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
    raw_event_n_jobs: int = 4,
    return_status: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    errors: list[dict] = []
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    symbols = [str(symbol) for symbol in symbols]
    max_workers = min(max(1, int(raw_event_n_jobs or 1)), max(1, len(symbols)))

    def _news_rows_for_symbol(symbol: str) -> list[dict]:
        symbol_name = (symbol_name_map or {}).get(symbol, "")
        return _fetch_naver_news_items(
            symbol=symbol,
            symbol_name=symbol_name,
            start_dt=start_dt,
            end_dt=end_dt,
            client_id=naver_client_id,
            client_secret=naver_client_secret,
            errors=errors,
        )

    if naver_client_id and naver_client_secret and symbols:
        if max_workers == 1:
            for symbol in symbols:
                rows.extend(_news_rows_for_symbol(symbol))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for part in executor.map(_news_rows_for_symbol, symbols):
                    rows.extend(part)

    if dart_api_key:
        corp_map = _load_dart_corp_map(dart_corp_map_csv)

        def _dart_rows_for_symbol(symbol: str) -> list[dict]:
            part: list[dict] = []
            corp = corp_map.get(symbol)
            if not corp:
                return part
            try:
                payload = _dart_list(dart_api_key, corp, start, end)
                items = _dart_items(payload)
            except Exception as exc:
                errors.append(
                    {
                        "source": "dart",
                        "symbol": symbol,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
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
                part.append(
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
            return part

        if max_workers == 1:
            for symbol in symbols:
                rows.extend(_dart_rows_for_symbol(symbol))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for part in executor.map(_dart_rows_for_symbol, symbols):
                    rows.extend(part)

    def _with_status(frame: pd.DataFrame) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
        if not return_status:
            return frame
        details = sorted(errors, key=lambda error: (str(error["symbol"]), str(error["source"]), str(error["error_type"])))
        failed_symbols = sorted({str(error["symbol"]) for error in details})
        error_types = sorted({str(error["error_type"]) for error in details})
        if details:
            status = "partial_failure" if not frame.empty or len(failed_symbols) < len(symbols) else "collection_failed"
        else:
            status = "no_events" if frame.empty else "success"
        return frame, {
            "status": status,
            "requested": len(symbols),
            "collected": int(len(frame)),
            "failed_symbols": failed_symbols,
            "error_types": error_types,
            "details": details,
        }

    if not rows:
        empty = pd.DataFrame(
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
        return _with_status(empty)

    out = pd.DataFrame(rows).drop_duplicates(subset=["Date", "Symbol", "source_type", "title", "raw_id"]).reset_index(drop=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out[(out["Date"] >= start_dt.normalize()) & (out["Date"] <= end_dt.normalize())].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.sort_values(["Date", "Symbol", "source_type", "title"]).reset_index(drop=True)
    return _with_status(out)
