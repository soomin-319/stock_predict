import pandas as pd

from src.reports.issue_summary import (
    SymbolIssueSummary,
    _build_structured_events,
    _extract_json_dict,
    _llm_symbol_issue_summary,
    append_issue_summary_columns,
)
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


def test_build_result_simple_excludes_removed_columns_and_keeps_issue_summary_columns():
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

    assert "오늘 종목 이슈 한줄 요약" not in simple.columns
    assert "공시 요약" in simple.columns
    assert "종합 판단" not in simple.columns
    assert "주의사항" not in simple.columns


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


def test_build_structured_events_clusters_news_and_categorizes_disclosures():
    events = pd.DataFrame(
        [
            {
                "Date": "2026-03-24",
                "Symbol": "005930.KS",
                "source_type": "disclosure",
                "title": "주요사항보고서(공급계약체결)",
                "published_at": "2026-03-24T08:15:00+09:00",
            },
            {
                "Date": "2026-03-24",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자, 메모리 반등 기대감 확대",
                "published_at": "2026-03-24T09:00:00+09:00",
            },
            {
                "Date": "2026-03-24",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자  메모리 반등 기대감 확대",
                "published_at": "2026-03-24T09:10:00+09:00",
            },
        ]
    )

    structured = _build_structured_events("005930.KS", "삼성전자", events)

    assert structured["symbol"] == "005930.KS"
    assert structured["disclosures"][0]["category"] == "contract"
    assert structured["news_clusters"][0]["article_count"] == 2


def test_append_issue_summary_columns_uses_default_model_when_only_openai_key(monkeypatch):
    base = pd.DataFrame([{"Symbol": "005930.KS", "종목명": "삼성전자"}])
    events = pd.DataFrame([{"Symbol": "005930.KS", "source_type": "news", "title": "테스트", "Date": "2026-03-24"}])

    captured = {}

    def _fake_llm(**kwargs):
        captured["model"] = kwargs["model"]
        return SymbolIssueSummary(
            one_line_summary="ok",
            disclosure_summary="ok",
            news_summary="ok",
            overall_judgment="중립",
            caution="ok",
            source_count=1,
            key_sources=["news"],
        )

    monkeypatch.setattr("src.reports.issue_summary._llm_symbol_issue_summary", _fake_llm)
    out = append_issue_summary_columns(base, context_raw_df=events, openai_api_key="sk-test", openai_model=None)

    assert captured["model"] == "gpt-5-mini"
    assert out.loc[0, "오늘 종목 이슈 한줄 요약"] == "ok"


def test_append_issue_summary_columns_limits_summary_to_requested_symbols(monkeypatch):
    base = pd.DataFrame(
        [
            {"Symbol": "005930.KS", "종목명": "삼성전자"},
            {"Symbol": "000660.KS", "종목명": "SK하이닉스"},
        ]
    )
    events = pd.DataFrame(
        [
            {"Symbol": "005930.KS", "source_type": "news", "title": "테스트", "Date": "2026-03-24"},
            {"Symbol": "000660.KS", "source_type": "news", "title": "테스트2", "Date": "2026-03-24"},
        ]
    )

    monkeypatch.setattr(
        "src.reports.issue_summary._llm_symbol_issue_summary",
        lambda **kwargs: SymbolIssueSummary(
            one_line_summary="요약 생성",
            disclosure_summary="[공시 요약]\n- 없음",
            news_summary="[뉴스 요약]\n- 있음",
            overall_judgment="중립",
            caution="c",
            source_count=1,
            key_sources=["news"],
        ),
    )

    out = append_issue_summary_columns(
        base,
        context_raw_df=events,
        openai_api_key="sk-test",
        summarize_symbols=["005930.KS"],
    )

    assert out.loc[0, "오늘 종목 이슈 한줄 요약"] == "요약 생성"
    assert out.loc[1, "오늘 종목 이슈 한줄 요약"] == "요약 비활성화"


def test_extract_json_dict_parses_wrapped_json():
    raw = "```json\n{\"overall_judgment\":\"호재\",\"one_line_summary\":\"ok\"}\n```"
    out = _extract_json_dict(raw)
    assert isinstance(out, dict)
    assert out["overall_judgment"] == "호재"


def test_llm_symbol_issue_summary_uses_fallback_news_lines_when_llm_returns_empty_marker(monkeypatch):
    events = pd.DataFrame(
        [
            {
                "Date": "2026-03-26",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 메모리 투자 확대 보도",
                "published_at": "2026-03-26T09:00:00+09:00",
            }
        ]
    )

    monkeypatch.setattr("src.reports.issue_summary.OpenAI", lambda api_key: object())

    def _fake_call_llm_text(client, model, prompt, max_output_tokens=700):
        if "[입력 뉴스 데이터]" in prompt:
            return "[뉴스 요약]\n- 확인된 핵심 뉴스 내용 없음"
        return "[공시 요약]\n- 확인된 핵심 공시 내용 없음"

    monkeypatch.setattr("src.reports.issue_summary._call_llm_text", _fake_call_llm_text)

    out = _llm_symbol_issue_summary(
        symbol="005930.KS",
        symbol_name="삼성전자",
        events=events,
        api_key="sk-test",
        model="gpt-5-mini",
    )

    assert out is not None
    assert "확인된 핵심 뉴스 내용 없음" not in out.news_summary
    assert "삼성전자 메모리 투자 확대 보도" in out.news_summary
