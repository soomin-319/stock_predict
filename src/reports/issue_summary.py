from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI


@dataclass
class SymbolIssueSummary:
    one_line_summary: str
    disclosure_summary: str
    news_summary: str
    overall_judgment: str
    caution: str
    source_count: int
    key_sources: list[str]


def _overall_judgment(disclosure_score: float, news_impact_score: float, news_count: int) -> str:
    weighted = disclosure_score * 0.55 + ((news_impact_score + 1.0) / 2.0) * 0.45
    if news_count <= 0 and disclosure_score <= 0:
        return "중립"
    if weighted >= 0.66:
        return "호재"
    if weighted <= 0.4:
        return "악재"
    return "중립"


def _disclosure_summary(disclosure_score: float) -> str:
    if disclosure_score >= 0.7:
        return "공시 강도가 높은 편으로 단기 투자심리에 의미 있는 변수로 해석됩니다."
    if disclosure_score >= 0.35:
        return "공시가 존재하지만 강도는 중간 수준으로, 단독 재료로 보기엔 제한적입니다."
    return "유의미한 공시 신호가 약해 공시 단독 영향은 제한적으로 보입니다."


def _news_summary(news_count: int, news_relevance: float, news_impact: float) -> str:
    if news_count <= 0:
        return "수집된 뉴스가 없어 뉴스 기반 해석 정보가 제한적입니다."
    tone = "긍정"
    if news_impact < -0.15:
        tone = "부정"
    elif abs(news_impact) <= 0.15:
        tone = "중립"
    return f"뉴스 {news_count}건 기준으로 관련도 {news_relevance:.2f}, 단기 톤은 {tone}으로 해석됩니다."


def _one_line_summary(judgment: str, disclosure_score: float, news_count: int) -> str:
    return (
        f"오늘 이슈 요약: 공시 강도 {disclosure_score:.2f}, 뉴스 {news_count}건을 종합하면 단기 해석은 '{judgment}'입니다."
    )


def summarize_symbol_issue(row: pd.Series) -> SymbolIssueSummary:
    disclosure_score = float(pd.to_numeric(row.get("disclosure_score", 0.0), errors="coerce") or 0.0)
    news_count = int(pd.to_numeric(row.get("news_article_count", 0), errors="coerce") or 0)
    news_relevance = float(pd.to_numeric(row.get("news_relevance_score", 0.0), errors="coerce") or 0.0)
    news_impact = float(pd.to_numeric(row.get("news_impact_score", 0.0), errors="coerce") or 0.0)

    judgment = _overall_judgment(disclosure_score, news_impact, news_count)
    disclosure_summary = _disclosure_summary(disclosure_score)
    news_summary = _news_summary(news_count, news_relevance, news_impact)
    one_line = _one_line_summary(judgment, disclosure_score, news_count)
    caution = "본 이슈 해석은 예측값 설명용 참고 정보이며, 예측 모델 입력/산출에는 반영되지 않습니다."

    key_sources = []
    if disclosure_score > 0:
        key_sources.append("disclosure")
    if news_count > 0:
        key_sources.append("news")

    return SymbolIssueSummary(
        one_line_summary=one_line,
        disclosure_summary=disclosure_summary,
        news_summary=news_summary,
        overall_judgment=judgment,
        caution=caution,
        source_count=(1 if disclosure_score > 0 else 0) + (1 if news_count > 0 else 0),
        key_sources=key_sources,
    )


def _build_llm_prompt(symbol: str, symbol_name: str, disclosures: list[str], news_titles: list[str]) -> str:
    disclosure_text = "\n".join(f"- {x}" for x in disclosures[:10]) or "- 없음"
    news_text = "\n".join(f"- {x}" for x in news_titles[:15]) or "- 없음"
    return (
        f"종목코드: {symbol}\n"
        f"종목명: {symbol_name}\n\n"
        "[공시 목록]\n"
        f"{disclosure_text}\n\n"
        "[뉴스 목록]\n"
        f"{news_text}\n\n"
        "위 정보를 바탕으로 반드시 JSON만 반환하세요.\n"
        "키는 one_line_summary, disclosure_summary, news_summary, overall_judgment, caution 이어야 합니다.\n"
        "overall_judgment는 '호재', '악재', '중립' 중 하나만 사용하세요."
    )


def _extract_json_dict(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _call_llm_json(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 400) -> dict | None:
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        raw = getattr(response, "output_text", "") or ""
        parsed = _extract_json_dict(raw)
        if parsed is not None:
            return parsed
    except Exception:
        pass

    try:
        chat_resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        chat_raw = (chat_resp.choices[0].message.content or "").strip()
        return _extract_json_dict(chat_raw)
    except Exception:
        return None


def _categorize_disclosure_title(title: str) -> str:
    t = str(title)
    if any(k in t for k in ("공급계약", "계약")):
        return "contract"
    if any(k in t for k in ("잠정실적", "실적")):
        return "earnings"
    if any(k in t for k in ("유상증자", "무상증자")):
        return "equity_financing"
    if any(k in t for k in ("전환사채", "신주인수권부사채", "cb", "bw")):
        return "convertible_financing"
    if any(k in t for k in ("자기주식", "자사주", "소각")):
        return "shareholder_return"
    if any(k in t for k in ("최대주주", "경영권")):
        return "control_change"
    if any(k in t for k in ("합병", "분할")):
        return "mna_restructuring"
    if any(k in t for k in ("조회공시", "불성실공시")):
        return "compliance_query"
    return "general_disclosure"


def _normalize_title_key(title: str) -> str:
    normalized = re.sub(r"\s+", " ", str(title).strip()).lower()
    normalized = re.sub(r"[^\w\s가-힣]", "", normalized)
    return normalized


def _build_structured_events(symbol: str, symbol_name: str, events: pd.DataFrame) -> dict:
    if events.empty:
        return {
            "symbol": symbol,
            "symbol_name": symbol_name,
            "date_kst": "",
            "disclosures": [],
            "news_clusters": [],
        }

    ev = events.copy()
    ev["title"] = ev["title"].astype(str).str.strip()
    ev["published_at"] = ev["published_at"].astype(str)
    ev = ev[ev["title"] != ""]
    date_kst = str(ev["Date"].astype(str).max()) if "Date" in ev.columns and not ev.empty else ""

    disclosures_df = ev[ev["source_type"] == "disclosure"].copy()
    disclosures_df = disclosures_df.drop_duplicates(subset=["title", "published_at"])
    disclosures = [
        {
            "title": row["title"],
            "published_at": row.get("published_at", ""),
            "category": _categorize_disclosure_title(row["title"]),
            "summary_hint": f"{_categorize_disclosure_title(row['title'])} 관련 공시",
        }
        for _, row in disclosures_df.head(10).iterrows()
    ]

    news_df = ev[ev["source_type"] == "news"].copy()
    news_df["cluster_key"] = news_df["title"].map(_normalize_title_key)
    clustered = (
        news_df.groupby("cluster_key", as_index=False)
        .agg(
            article_count=("title", "count"),
            representative_title=("title", "first"),
            representative_titles=("title", lambda x: list(dict.fromkeys([str(v) for v in x]))[:3]),
        )
        .sort_values("article_count", ascending=False)
    )
    news_clusters = [
        {
            "cluster_topic": row["representative_title"],
            "article_count": int(row["article_count"]),
            "representative_titles": row["representative_titles"],
            "novelty_score": round(1.0 / max(int(row["article_count"]), 1), 3),
        }
        for _, row in clustered.head(12).iterrows()
    ]

    return {
        "symbol": symbol,
        "symbol_name": symbol_name,
        "date_kst": date_kst,
        "disclosures": disclosures,
        "news_clusters": news_clusters,
    }


def _llm_symbol_issue_summary(
    *,
    symbol: str,
    symbol_name: str,
    events: pd.DataFrame,
    api_key: str,
    model: str,
) -> SymbolIssueSummary | None:
    structured_payload = _build_structured_events(symbol, symbol_name, events)

    raw_news_count = int((events["source_type"] == "news").sum()) if "source_type" in events.columns else 0
    dedup_news_count = int(sum(int(c.get("article_count", 0)) for c in structured_payload.get("news_clusters", [])))
    cluster_count = int(len(structured_payload.get("news_clusters", [])))

    stage1_prompt = (
        "너는 한국 주식시장 뉴스/공시를 해석하는 금융 보조 AI다.\n"
        "입력 데이터만 사용하고 원문에 없는 사실/숫자를 만들지 마라.\n"
        "투자 권유(매수/매도 추천)나 확정적 주가 예측 표현을 금지한다.\n"
        "아래 구조화 데이터로 핵심 사실을 추출해 JSON만 반환하라.\n"
        "키는 article_facts 이고 각 원소는 topic, fact_type(fact/forecast/opinion), evidence_strength(high/medium/low), contains_new_fact(true/false), is_repackaged_story(true/false) 를 포함한다.\n\n"
        f"{json.dumps(structured_payload, ensure_ascii=False)}"
    )
    client = OpenAI(api_key=api_key)
    stage1 = _call_llm_json(client, model, stage1_prompt, max_output_tokens=500)
    if stage1 is None:
        stage1 = {"article_facts": []}

    stage2_prompt = (
        "너는 한국 주식 종목 뉴스/공시 해석 AI다.\n"
        "입력으로 제공된 데이터만 사용하고 과장/권유 표현 없이 신중하게 요약하라.\n"
        "공시 사실이 뉴스 해석보다 우선한다.\n"
        "반드시 JSON만 반환하라.\n"
        "필드: headline, key_topics, news_summary, overall_signal(positive/negative/neutral), risk_note, evidence_count.\n"
        f"raw_news_count={raw_news_count}, deduplicated_news_count={dedup_news_count}, cluster_count={cluster_count}\n\n"
        "[구조화 데이터]\n"
        f"{json.dumps(structured_payload, ensure_ascii=False)}\n\n"
        "[1단계 핵심사실]\n"
        f"{json.dumps(stage1, ensure_ascii=False)}"
    )
    try:
        payload = _call_llm_json(client, model, stage2_prompt, max_output_tokens=500)
        if payload is None:
            raise ValueError("LLM JSON 파싱 실패")
    except Exception as exc:
        print(f"[ISSUE SUMMARY] LLM 요약 실패 ({symbol}): {type(exc).__name__}: {exc}")
        return None

    signal = str(payload.get("overall_signal", "neutral")).strip().lower()
    judgment = "중립"
    if signal in {"positive", "bullish", "호재"}:
        judgment = "호재"
    elif signal in {"negative", "bearish", "악재"}:
        judgment = "악재"

    key_topics = payload.get("key_topics", [])
    if isinstance(key_topics, list):
        topics_text = ", ".join(str(x) for x in key_topics[:4] if str(x).strip())
    else:
        topics_text = ""

    return SymbolIssueSummary(
        one_line_summary=str(payload.get("headline", "")).strip() or "오늘 이슈 요약을 생성하지 못했습니다.",
        disclosure_summary=(
            str(payload.get("disclosure_summary", "")).strip()
            or (f"공시 핵심 이슈: {topics_text}" if topics_text else "당일 공시 관련 핵심 이슈를 요약했습니다.")
        ),
        news_summary=str(payload.get("news_summary", "")).strip() or "뉴스 요약을 생성하지 못했습니다.",
        overall_judgment=judgment,
        caution=str(payload.get("risk_note", "")).strip()
        or "본 이슈 해석은 예측값 설명용 참고 정보이며, 예측 모델 입력/산출에는 반영되지 않습니다.",
        source_count=int(payload.get("evidence_count", len(events)) or len(events)),
        key_sources=sorted(events["source_type"].dropna().astype(str).unique().tolist()),
    )


def append_issue_summary_columns(
    pred_df: pd.DataFrame,
    context_raw_df: pd.DataFrame | None = None,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
) -> pd.DataFrame:
    if pred_df.empty:
        out = pred_df.copy()
        for col in [
            "오늘 종목 이슈 한줄 요약",
            "공시 요약",
            "뉴스 요약",
            "종합 판단",
            "주의사항",
            "원문 개수",
            "핵심 원문 목록",
        ]:
            out[col] = []
        return out

    out = pred_df.copy()
    context = context_raw_df.copy() if isinstance(context_raw_df, pd.DataFrame) and not context_raw_df.empty else None
    if context is not None and "Symbol" in context.columns:
        context["Symbol"] = context["Symbol"].astype(str)
    resolved_model = openai_model or ("gpt-5" if openai_api_key else None)
    use_llm = bool(openai_api_key and resolved_model and context is not None and "source_type" in context.columns)

    summaries: list[SymbolIssueSummary] = []
    for _, row_series in out.iterrows():
        if use_llm:
            symbol = str(row_series.get("Symbol", ""))
            symbol_name = str(row_series.get("종목명", symbol))
            events = context[context["Symbol"] == symbol]
            if not events.empty:
                llm_summary = _llm_symbol_issue_summary(
                    symbol=symbol,
                    symbol_name=symbol_name,
                    events=events,
                    api_key=str(openai_api_key),
                    model=str(resolved_model),
                )
                if llm_summary is not None:
                    summaries.append(llm_summary)
                    continue
        summaries.append(summarize_symbol_issue(row_series))

    summaries = pd.Series(summaries)
    out["오늘 종목 이슈 한줄 요약"] = summaries.map(lambda s: s.one_line_summary)
    out["공시 요약"] = summaries.map(lambda s: s.disclosure_summary)
    out["뉴스 요약"] = summaries.map(lambda s: s.news_summary)
    out["종합 판단"] = summaries.map(lambda s: s.overall_judgment)
    out["주의사항"] = summaries.map(lambda s: s.caution)
    out["원문 개수"] = summaries.map(lambda s: s.source_count)
    out["핵심 원문 목록"] = summaries.map(lambda s: json.dumps(s.key_sources, ensure_ascii=False))
    return out


__all__ = ["append_issue_summary_columns", "summarize_symbol_issue", "SymbolIssueSummary"]
