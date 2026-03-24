from __future__ import annotations

import json
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


def _llm_symbol_issue_summary(
    *,
    symbol: str,
    symbol_name: str,
    events: pd.DataFrame,
    api_key: str,
    model: str,
) -> SymbolIssueSummary | None:
    disclosures = events.loc[events["source_type"] == "disclosure", "title"].dropna().astype(str).tolist()
    news_titles = events.loc[events["source_type"] == "news", "title"].dropna().astype(str).tolist()
    prompt = _build_llm_prompt(symbol, symbol_name, disclosures, news_titles)
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=300,
        )
        raw = getattr(response, "output_text", "") or ""
        payload = json.loads(raw)
    except Exception:
        return None

    judgment = str(payload.get("overall_judgment", "중립")).strip()
    if judgment not in {"호재", "악재", "중립"}:
        judgment = "중립"
    return SymbolIssueSummary(
        one_line_summary=str(payload.get("one_line_summary", "")).strip() or "오늘 이슈 요약을 생성하지 못했습니다.",
        disclosure_summary=str(payload.get("disclosure_summary", "")).strip()
        or "공시 요약을 생성하지 못했습니다.",
        news_summary=str(payload.get("news_summary", "")).strip() or "뉴스 요약을 생성하지 못했습니다.",
        overall_judgment=judgment,
        caution=str(payload.get("caution", "")).strip()
        or "본 이슈 해석은 예측값 설명용 참고 정보이며, 예측 모델 입력/산출에는 반영되지 않습니다.",
        source_count=int(len(events)),
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
    use_llm = bool(openai_api_key and openai_model and context is not None and "source_type" in context.columns)

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
                    model=str(openai_model),
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
