import pandas as pd

from src.reports.issue_summary import (
    SymbolIssueSummary,
    _build_structured_events,
    _extract_json_dict,
    _llm_symbol_issue_summary,
    append_issue_summary_columns,
)
from src.reports.result_formatter import (
    RESULT_SIMPLE_OPTIONAL_COLUMNS,
    RESULT_SIMPLE_REQUIRED_COLUMNS,
    build_result_simple,
    validate_result_simple_schema,
)


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
    assert "원문 개수" not in simple.columns
    assert "핵심 원문 목록" not in simple.columns
    assert list(simple.columns) == [*RESULT_SIMPLE_REQUIRED_COLUMNS, *RESULT_SIMPLE_OPTIONAL_COLUMNS]


def test_validate_result_simple_schema_requires_current_contract():
    ok, missing = validate_result_simple_schema(pd.DataFrame(columns=list(RESULT_SIMPLE_REQUIRED_COLUMNS)))
    assert ok is True
    assert missing == []

    ok2, missing2 = validate_result_simple_schema(pd.DataFrame(columns=["종목코드", "종목명"]))
    assert ok2 is False
    assert "권고" in missing2


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


def test_append_issue_summary_columns_limits_default_llm_work(monkeypatch):
    base = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "predicted_return": 0.5,
                "predicted_close": 70000.0,
                "up_probability": 0.61,
            },
            {
                "Symbol": "000660.KS",
                "symbol_name": "SK하이닉스",
                "predicted_return": 0.4,
                "predicted_close": 120000.0,
                "up_probability": 0.59,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "Date": "2026-03-24",
                "Symbol": symbol,
                "source_type": "news",
                "title": f"{name} 신규 수주",
                "published_at": "2026-03-24T00:00:00",
            }
            for symbol, name in [("005930.KS", "삼성전자"), ("000660.KS", "SK하이닉스")]
        ]
    )
    calls: list[str] = []

    def _fake_llm(**kwargs):
        calls.append(kwargs["symbol"])
        return SymbolIssueSummary(
            one_line_summary=f"{kwargs['symbol']} LLM",
            disclosure_summary="[공시 요약]\n- 없음",
            news_summary="[뉴스 요약]\n- LLM 뉴스",
            overall_judgment="중립",
            caution="참고용",
            source_count=1,
            key_sources=["news"],
        )

    monkeypatch.setattr("src.reports.issue_summary._llm_symbol_issue_summary", _fake_llm)

    out = append_issue_summary_columns(
        base,
        context_raw_df=events,
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
        max_llm_symbols=1,
    )

    assert calls == ["005930.KS"]
    assert out.loc[0, "오늘 종목 이슈 한줄 요약"] == "005930.KS LLM"
    assert "기준 당일 공시" in out.loc[1, "오늘 종목 이슈 한줄 요약"]
    assert out.loc[1, "predicted_return"] == base.loc[1, "predicted_return"]


def test_append_issue_summary_columns_reuses_llm_cache(monkeypatch, tmp_path):
    base = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "predicted_return": 0.5,
                "predicted_close": 70000.0,
                "up_probability": 0.61,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "Date": "2026-03-24",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 신규 수주",
                "published_at": "2026-03-24T00:00:00",
            }
        ]
    )
    calls = 0

    def _fake_llm(**kwargs):
        nonlocal calls
        calls += 1
        return SymbolIssueSummary(
            one_line_summary="cached LLM",
            disclosure_summary="[공시 요약]\n- 없음",
            news_summary="[뉴스 요약]\n- LLM 뉴스",
            overall_judgment="중립",
            caution="참고용",
            source_count=1,
            key_sources=["news"],
        )

    monkeypatch.setattr("src.reports.issue_summary._llm_symbol_issue_summary", _fake_llm)

    first = append_issue_summary_columns(
        base,
        context_raw_df=events,
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
        llm_cache_dir=tmp_path / "issue-cache",
    )
    second = append_issue_summary_columns(
        base,
        context_raw_df=events,
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
        llm_cache_dir=tmp_path / "issue-cache",
    )

    assert calls == 1
    assert first.loc[0, "오늘 종목 이슈 한줄 요약"] == "cached LLM"
    assert second.loc[0, "오늘 종목 이슈 한줄 요약"] == "cached LLM"


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


def test_append_issue_summary_columns_handles_duplicate_input_columns_when_summary_disabled():
    base = pd.DataFrame(
        [["000660.KS", "first", "second"]],
        columns=["Symbol", "prediction_reason", "prediction_reason"],
    )

    out = append_issue_summary_columns(base, summarize_symbols=["005930.KS"])

    summary_col = [c for c in out.columns if c not in base.columns][0]
    assert out.loc[0, summary_col] == "요약 비활성화"


def test_append_issue_summary_columns_parallel_preserves_input_order(monkeypatch):
    base = pd.DataFrame(
        [
            {"Symbol": "A", "symbol_name": "Alpha"},
            {"Symbol": "B", "symbol_name": "Beta"},
            {"Symbol": "C", "symbol_name": "Gamma"},
        ]
    )
    events = pd.DataFrame(
        [
            {"Symbol": "A", "source_type": "news", "title": "A news", "Date": "2026-03-24"},
            {"Symbol": "B", "source_type": "news", "title": "B news", "Date": "2026-03-24"},
            {"Symbol": "C", "source_type": "news", "title": "C news", "Date": "2026-03-24"},
        ]
    )

    def _fake_llm(**kwargs):
        symbol = kwargs["symbol"]
        return SymbolIssueSummary(
            one_line_summary=f"{symbol}-summary",
            disclosure_summary="d",
            news_summary="n",
            overall_judgment="neutral",
            caution="c",
            source_count=1,
            key_sources=["news"],
        )

    monkeypatch.setattr("src.reports.issue_summary._llm_symbol_issue_summary", _fake_llm)

    out = append_issue_summary_columns(
        base,
        context_raw_df=events,
        openai_api_key="sk-test",
        summary_n_jobs=3,
    )
    summary_col = [c for c in out.columns if c not in base.columns][0]

    assert out["Symbol"].tolist() == ["A", "B", "C"]
    assert out[summary_col].tolist() == ["A-summary", "B-summary", "C-summary"]


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

# Provider switch tests added for shared LLM config.
from src.reports import issue_summary as _issue_summary_mod


class _FakeOpenAIChat:
    class _Completions:
        def __init__(self, calls):
            self._calls = calls

        def create(self, **kwargs):
            self._calls["chat"] += 1
            self._calls["chat_kwargs"] = kwargs

            class _Message:
                content = "chat text"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    def __init__(self, calls):
        self.completions = self._Completions(calls)


class _FakeOpenAIResponses:
    def __init__(self, calls, *, fail=False):
        self._calls = calls
        self._fail = fail

    def create(self, **kwargs):
        self._calls["responses"] += 1
        if self._fail:
            raise RuntimeError("responses API not implemented")

        class _Response:
            output_text = "responses text"

        return _Response()


class _FakeOpenAIClient:
    def __init__(self, calls, *, fail_responses=False):
        self.responses = _FakeOpenAIResponses(calls, fail=fail_responses)
        self.chat = _FakeOpenAIChat(calls)


def test_gemma_provider_skips_responses_and_uses_chat():
    calls = {"responses": 0, "chat": 0}
    client = _FakeOpenAIClient(calls, fail_responses=True)

    text = _issue_summary_mod._call_llm_text(client, "gemma-4-26b-a4b", "hello", provider="llama_cpp")

    assert text == "chat text"
    assert calls["responses"] == 0
    assert calls["chat"] == 1


def test_openai_provider_uses_responses_first():
    calls = {"responses": 0, "chat": 0}
    client = _FakeOpenAIClient(calls)

    text = _issue_summary_mod._call_llm_text(client, "gpt-5-mini", "hello", provider="openai")

    assert text == "responses text"
    assert calls["responses"] == 1
    assert calls["chat"] == 0


def test_llm_symbol_issue_summary_passes_base_url_and_dummy_key_for_gemma(monkeypatch):
    captured = {}

    class _ClientFactory:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.responses = _FakeOpenAIResponses({"responses": 0, "chat": 0}, fail=True)
            self.chat = _FakeOpenAIChat({"responses": 0, "chat": 0})

    events = pd.DataFrame(
        [{"Date": "2026-06-25", "Symbol": "005930.KS", "source_type": "news", "title": "memory rebound"}]
    )
    monkeypatch.setattr("src.reports.issue_summary.OpenAI", _ClientFactory)
    monkeypatch.setattr("src.reports.issue_summary._call_llm_text", lambda *args, **kwargs: "- ok")

    out = _issue_summary_mod._llm_symbol_issue_summary(
        symbol="005930.KS",
        symbol_name="Samsung",
        events=events,
        api_key=None,
        model="gemma-4-26b-a4b",
        provider="llama_cpp",
        base_url="http://localhost:8001/v1",
    )

    assert out is not None
    assert captured["api_key"] == "not-needed"
    assert captured["base_url"] == "http://localhost:8001/v1"


def test_issue_summary_gemma_enabled_without_api_key(monkeypatch):
    base = pd.DataFrame([{"Symbol": "005930.KS", "symbol_name": "Samsung"}])
    events = pd.DataFrame(
        [{"Date": "2026-06-25", "Symbol": "005930.KS", "source_type": "news", "title": "memory rebound"}]
    )
    captured = {}

    def _fake_llm(**kwargs):
        captured.update(kwargs)
        return SymbolIssueSummary(
            one_line_summary="llm",
            disclosure_summary="d",
            news_summary="n",
            overall_judgment="neutral",
            caution="c",
            source_count=1,
            key_sources=["news"],
        )

    monkeypatch.setattr("src.reports.issue_summary._llm_symbol_issue_summary", _fake_llm)

    out = append_issue_summary_columns(
        base,
        context_raw_df=events,
        openai_api_key=None,
        openai_model="gemma-4-26b-a4b",
        provider="llama_cpp",
        base_url="http://localhost:8001/v1",
    )

    added_cols = [col for col in out.columns if col not in base.columns]
    assert captured["api_key"] is None
    assert captured["provider"] == "llama_cpp"
    assert captured["base_url"] == "http://localhost:8001/v1"
    assert out.loc[0, added_cols[0]] == "llm"


def test_issue_summary_cache_key_includes_provider_and_base_url():
    events = pd.DataFrame(
        [{"Date": "2026-06-25", "Symbol": "005930.KS", "source_type": "news", "title": "memory rebound"}]
    )

    gemma_key = _issue_summary_mod._issue_summary_cache_key(
        model="gemma-4-26b-a4b",
        provider="llama_cpp",
        base_url="http://localhost:8001/v1",
        symbol="005930.KS",
        symbol_name="Samsung",
        events=events,
    )
    openai_key = _issue_summary_mod._issue_summary_cache_key(
        model="gemma-4-26b-a4b",
        provider="openai",
        base_url="https://api.openai.com/v1",
        symbol="005930.KS",
        symbol_name="Samsung",
        events=events,
    )

    assert gemma_key != openai_key
