import pandas as pd

from src.reports.issue_summary import SymbolIssueSummary, append_issue_summary_columns
from src.reports.result_formatter import build_result_simple


def test_append_issue_summary_columns_keeps_prediction_values_unchanged():
    base = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "predicted_return": 1.23,
                "predicted_close": 71000.0,
                "up_probability": 0.77,
                "recommendation": "매수",
                "portfolio_action": "BUY",
                "trading_gate": "open",
                "risk_flag": "normal",
                "prediction_reason": "테스트",
                "confidence_score": 0.8,
                "history_direction_accuracy": 0.7,
                "disclosure_score": 0.8,
                "news_impact_score": 0.4,
                "news_relevance_score": 0.7,
                "news_article_count": 5,
            }
        ]
    )

    out = append_issue_summary_columns(base)

    assert out.loc[0, "predicted_return"] == base.loc[0, "predicted_return"]
    assert out.loc[0, "predicted_close"] == base.loc[0, "predicted_close"]
    assert out.loc[0, "up_probability"] == base.loc[0, "up_probability"]
    assert out.loc[0, "종합 판단"] in {"호재", "악재", "중립"}
    assert "예측 모델 입력/산출에는 반영되지 않습니다" in out.loc[0, "주의사항"]


def test_build_result_simple_includes_issue_columns_when_present():
    df = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "recommendation": "매수",
                "portfolio_action": "BUY",
                "trading_gate": "open",
                "risk_flag": "normal",
                "predicted_close": 71000.0,
                "predicted_return": 1.23,
                "up_probability": 0.77,
                "prediction_reason": "테스트",
                "confidence_score": 0.8,
                "history_direction_accuracy": 0.7,
                "오늘 종목 이슈 한줄 요약": "이슈 요약",
                "공시 요약": "공시",
                "뉴스 요약": "뉴스",
                "종합 판단": "중립",
                "주의사항": "참고용",
                "원문 개수": 2,
                "핵심 원문 목록": '["disclosure","news"]',
            }
        ]
    )

    simple = build_result_simple(df)

    assert "오늘 종목 이슈 한줄 요약" in simple.columns
    assert "공시 요약" in simple.columns
    assert simple.loc[0, "종합 판단"] == "중립"


def test_append_issue_summary_columns_uses_llm_for_summary_only(monkeypatch):
    base = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "종목명": "삼성전자",
                "predicted_return": 0.5,
                "predicted_close": 70000.0,
                "up_probability": 0.61,
                "disclosure_score": 0.0,
                "news_impact_score": 0.0,
                "news_relevance_score": 0.0,
                "news_article_count": 0,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "Date": "2026-03-24",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 수주 기대감",
                "published_at": "2026-03-24T00:00:00",
                "provider": "yfinance",
                "url": "",
                "raw_id": "n1",
            }
        ]
    )

    def _fake_llm(**kwargs):
        return SymbolIssueSummary(
            one_line_summary="LLM 한줄 요약",
            disclosure_summary="LLM 공시 요약",
            news_summary="LLM 뉴스 요약",
            overall_judgment="호재",
            caution="LLM 주의사항",
            source_count=1,
            key_sources=["news"],
        )

    monkeypatch.setattr("src.reports.issue_summary._llm_symbol_issue_summary", _fake_llm)

    out = append_issue_summary_columns(
        base,
        context_raw_df=events,
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
    )

    assert out.loc[0, "오늘 종목 이슈 한줄 요약"] == "LLM 한줄 요약"
    assert out.loc[0, "종합 판단"] == "호재"
    assert out.loc[0, "predicted_return"] == base.loc[0, "predicted_return"]
