from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd


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


def append_issue_summary_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
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
    summaries = out.apply(summarize_symbol_issue, axis=1)
    out["오늘 종목 이슈 한줄 요약"] = summaries.map(lambda s: s.one_line_summary)
    out["공시 요약"] = summaries.map(lambda s: s.disclosure_summary)
    out["뉴스 요약"] = summaries.map(lambda s: s.news_summary)
    out["종합 판단"] = summaries.map(lambda s: s.overall_judgment)
    out["주의사항"] = summaries.map(lambda s: s.caution)
    out["원문 개수"] = summaries.map(lambda s: s.source_count)
    out["핵심 원문 목록"] = summaries.map(lambda s: json.dumps(s.key_sources, ensure_ascii=False))
    return out


__all__ = ["append_issue_summary_columns", "summarize_symbol_issue", "SymbolIssueSummary"]
