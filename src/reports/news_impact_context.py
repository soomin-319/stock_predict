from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from news_impact.report import REPORT_DISCLAIMER
from news_impact.schema import ImpactEvent
from news_impact.scorer import aggregate_scores
from news_impact.stock_factors.classifier import analyze_stock_factors

NEWS_IMPACT_COLUMNS = {
    "run_id": "news_impact_run_id",
    "final_score": "news_impact_final_score",
    "news_disclosure_score": "news_impact_final_score",
    "sector_neutral_score": "news_impact_sector_neutral_score",
    "positive_score": "news_impact_positive_score",
    "negative_score": "news_impact_negative_score",
    "uncertainty_score": "news_impact_uncertainty_score",
    "confidence": "news_impact_confidence",
    "event_count": "news_impact_event_count",
    "top_event_type": "news_impact_top_event_type",
    "top_reason": "news_impact_top_reason",
    "why_may_be_wrong": "news_impact_why_may_be_wrong",
    "risk_flags": "news_impact_risk_flags",
    "tradeability_status": "news_impact_tradeability_status",
    "review_checklist": "news_impact_review_checklist",
    "top_evidence_url": "news_impact_top_evidence_url",
}


def append_news_impact_context(pred_df: pd.DataFrame, report_path: str | Path | None) -> pd.DataFrame:
    if pred_df.empty or not report_path:
        return pred_df
    path = Path(report_path)
    if not path.exists():
        return pred_df
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pred_df
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return pred_df
    return append_news_impact_rows(pred_df, rows)


def append_news_impact_rows(pred_df: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if pred_df.empty or not rows:
        return pred_df
    context = pd.DataFrame(rows)
    if not {"date", "ticker"}.issubset(set(context.columns)):
        return pred_df
    out = pred_df.copy()
    join = context.copy()
    join["Date"] = pd.to_datetime(join["date"], errors="coerce").dt.normalize()
    join["Symbol"] = join["ticker"].astype(str).map(_ticker_to_symbol)
    keep = ["Date", "Symbol"]
    for source, target in NEWS_IMPACT_COLUMNS.items():
        if source in join.columns:
            join[target] = join[source]
            if target not in keep:
                keep.append(target)
    join = join[keep].dropna(subset=["Date", "Symbol"]).drop_duplicates(subset=["Date", "Symbol"], keep="last")
    if join.empty:
        return out
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out.merge(join, on=["Date", "Symbol"], how="left")
    return _append_display_columns(out)


def append_generated_news_impact_context(
    pred_df: pd.DataFrame,
    context_raw_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Append display-only news/disclosure impact scores built from raw context rows."""
    if pred_df.empty or context_raw_df is None or context_raw_df.empty:
        return pred_df
    if not {"Date", "Symbol", "source_type", "title"}.issubset(set(context_raw_df.columns)):
        return pred_df

    pred_keys = pred_df[["Date", "Symbol"]].copy()
    pred_keys["Date"] = pd.to_datetime(pred_keys["Date"], errors="coerce").dt.normalize()
    pred_keys["Symbol"] = pred_keys["Symbol"].astype(str)

    raw = context_raw_df.copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce").dt.normalize()
    raw["Symbol"] = raw["Symbol"].astype(str)
    raw = raw.merge(pred_keys.drop_duplicates(), on=["Date", "Symbol"], how="inner")
    if raw.empty:
        return pred_df

    symbol_names = {
        str(row.get("Symbol")): str(row.get("symbol_name") or row.get("종목명") or row.get("Symbol"))
        for _, row in pred_df.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for (date_value, symbol), group in raw.groupby(["Date", "Symbol"], sort=False):
        ticker = _symbol_to_ticker(symbol)
        if ticker is None:
            continue
        company = symbol_names.get(str(symbol), str(symbol))
        events = [
            event
            for event in (
                _raw_event_to_impact_event(item, ticker=ticker, company=company)
                for _, item in group.iterrows()
            )
            if event is not None
        ]
        if not events:
            continue
        summary = aggregate_scores(events)
        top_event = max(events, key=lambda event: abs(float(event.impact_score)))
        risk_flags = tuple(dict.fromkeys(flag for event in events for flag in event.risk_flags))
        evidence_urls = [url for event in events for url in event.evidence_urls if url]
        final_score = float(summary.news_disclosure_score)
        rows.append(
            {
                "date": pd.to_datetime(date_value).strftime("%Y-%m-%d"),
                "run_id": _generated_run_id(date_value, symbol, events),
                "ticker": ticker,
                "company": company,
                "market": _market_from_symbol(symbol),
                "sector": top_event.sector,
                "final_score": final_score,
                "news_disclosure_score": final_score,
                "global_proxy_adjustment": 0.0,
                "sector_neutral_score": final_score,
                "positive_score": float(summary.positive_score),
                "negative_score": float(summary.negative_score),
                "uncertainty_score": float(summary.uncertainty_score),
                "confidence": round(sum(float(e.confidence) for e in events) / len(events), 3),
                "event_count": int(summary.event_count),
                "llm_failed_count": int(summary.llm_failed_count),
                "top_event_type": top_event.event_type,
                "top_reason": top_event.reason,
                "why_may_be_wrong": top_event.why_may_be_wrong,
                "risk_flags": ";".join(risk_flags),
                "already_reflected_price_move": 0.0,
                "price_change_since_news": 0.0,
                "volume_change": 0.0,
                "tradeability_status": "review_only",
                "review_checklist": "원문 확인;예측값 미반영 확인",
                "top_evidence_url": evidence_urls[0] if evidence_urls else "",
                "disclaimer": REPORT_DISCLAIMER,
            }
        )
    if not rows:
        return pred_df
    return append_news_impact_rows(pred_df, rows)


def _ticker_to_symbol(ticker: str) -> str:
    value = str(ticker).strip()
    if "." in value:
        return value
    if len(value) == 6 and value.isdigit():
        return f"{value}.KS"
    return value


def _append_display_columns(out: pd.DataFrame) -> pd.DataFrame:
    if "news_impact_final_score" not in out.columns:
        return out
    score = pd.to_numeric(out["news_impact_final_score"], errors="coerce")
    out["뉴스/공시 영향 점수"] = score.map(_format_impact_score)
    reason = out.get("news_impact_top_reason", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    out["뉴스/공시 영향 요약"] = reason.mask(reason == "", "-")
    out["뉴스/공시 영향 참고"] = score.map(lambda value: "-" if pd.isna(value) else "참고용·예측값 미반영")
    return out


def _format_impact_score(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):+.1f}점"


def _raw_event_to_impact_event(row: pd.Series, *, ticker: str, company: str) -> ImpactEvent | None:
    title = str(row.get("title") or "").strip()
    body = str(row.get("body") or "").strip()
    text = f"{title} {body}".strip()
    if not text:
        return None
    source_type = str(row.get("source_type") or "").strip().lower()
    analysis = analyze_stock_factors(text)
    score = _heuristic_impact_score(text, source_type, analysis.direction)
    direction = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    url = str(row.get("url") or "").strip()
    risk_flags = ["heuristic_display_only"]
    if source_type == "news" and not body:
        risk_flags.append("news_title_only")
    if analysis.freshness_required:
        risk_flags.append("freshness_review_required")
    return ImpactEvent(
        event_id=_stable_id("event", ticker, source_type, title, url),
        cluster_id=_stable_id("cluster", ticker, source_type, title),
        ticker=ticker,
        company=company,
        sector=_sector_from_factors(analysis.factors),
        event_type=_event_type_from_text(text, source_type),
        impact_direction=direction,
        impact_strength=round(min(abs(score) / 100.0, 1.0), 6),
        impact_score=score,
        time_horizon="next_day",
        confidence=_confidence_value(analysis.confidence, source_type),
        expectedness="unknown",
        novelty_score=1.0,
        already_reflected_price_move=0.0,
        reason=_reason_text(title, analysis.summary),
        why_may_be_wrong="원문 제목·요약 기반 휴리스틱 점수라 실제 재무 영향과 다를 수 있습니다.",
        risk_flags=tuple(risk_flags),
        evidence_urls=(url,) if url else (),
    )


def _heuristic_impact_score(text: str, source_type: str, direction: str) -> float:
    normalized = text.casefold()
    positive = _contains_any(
        normalized,
        ("수주", "계약", "공급계약", "증가", "개선", "호조", "상향", "흑자", "자사주", "배당", "승인", "hbm", "수요"),
    )
    negative = _contains_any(
        normalized,
        ("소송", "제재", "적자", "감소", "악화", "하락", "급락", "유상증자", "전환사채", "불성실", "횡령", "배임"),
    )
    if direction == "positive":
        base = 30.0
    elif direction == "negative":
        base = -30.0
    elif positive and not negative:
        base = 25.0
    elif negative and not positive:
        base = -25.0
    elif positive and negative:
        base = 0.0
    else:
        base = 10.0 if source_type == "disclosure" else 5.0
    if source_type == "disclosure":
        base *= 1.25
    return round(max(min(base, 100.0), -100.0), 6)


def _event_type_from_text(text: str, source_type: str) -> str:
    normalized = text.casefold()
    if _contains_any(normalized, ("실적", "영업이익", "매출", "순이익")):
        return "earnings"
    if _contains_any(normalized, ("계약", "수주", "공급계약")):
        return "contract"
    if _contains_any(normalized, ("유상증자", "전환사채", "cb", "bw")):
        return "capital_raise"
    if _contains_any(normalized, ("소송", "제재", "횡령", "배임")):
        return "legal"
    if _contains_any(normalized, ("정책", "규제", "정부")):
        return "policy"
    if source_type == "disclosure":
        return "other"
    return "sector" if _contains_any(normalized, ("반도체", "hbm", "수출", "업황")) else "other"


def _sector_from_factors(factors: tuple[str, ...]) -> str:
    if "SEMI" in factors:
        return "semiconductor"
    if "EXPORT" in factors:
        return "export"
    if "GOVERNANCE" in factors:
        return "governance"
    return ""


def _confidence_value(confidence: str, source_type: str) -> float:
    base = {"high": 0.8, "medium": 0.65, "low": 0.45}.get(str(confidence), 0.5)
    if source_type == "disclosure":
        base += 0.1
    return round(min(base, 1.0), 6)


def _reason_text(title: str, summary: str) -> str:
    title = str(title or "").strip()
    summary = str(summary or "").strip()
    if title and summary:
        return f"{title} ({summary})"
    return title or summary or "-"


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle.casefold() in value for needle in needles)


def _symbol_to_ticker(symbol: str) -> str | None:
    value = str(symbol).strip()
    ticker = value.split(".", 1)[0]
    return ticker if len(ticker) == 6 and ticker.isdigit() else None


def _market_from_symbol(symbol: str) -> str:
    value = str(symbol).upper()
    if value.endswith(".KQ"):
        return "KOSDAQ"
    return "KOSPI"


def _stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _generated_run_id(date_value: object, symbol: str, events: list[ImpactEvent]) -> str:
    return "generated-" + _stable_id(pd.to_datetime(date_value).strftime("%Y-%m-%d"), symbol, len(events))
