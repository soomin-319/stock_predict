import json
import types
from pathlib import Path

import pandas as pd

from src.domain.signal_policy import recommendation_from_signal
from src.reports.news_impact_context import (
    append_generated_news_impact_context,
    append_llm_news_impact_context,
    append_news_impact_context,
)
from src.reports.result_formatter import build_result_simple
from src.pipeline import _classify_context_export


def test_context_export_classifies_event_summary_and_no_data():
    metadata = {"environment": "production", "data_mode": "real"}
    event = _classify_context_export(
        pd.DataFrame([{"Symbol": "005930.KS", "source_type": "news", "title": "event"}]),
        pd.DataFrame(),
        source_type="news",
        metadata=metadata,
    )
    summary = _classify_context_export(
        pd.DataFrame(),
        pd.DataFrame([{"Symbol": "005930.KS", "뉴스 요약": "summary"}]),
        source_type="news",
        metadata=metadata,
    )
    no_data = _classify_context_export(
        pd.DataFrame(),
        pd.DataFrame(),
        source_type="news",
        metadata=metadata,
        no_data_reason="context_date_gap_exceeded",
    )

    assert event.iloc[0]["record_type"] == "event"
    assert summary.iloc[0]["record_type"] == "summary"
    assert no_data.iloc[0]["record_type"] == "no_data"
    assert no_data.iloc[0]["collection_status"] == "excluded"
    assert no_data.iloc[0]["no_data_reason"] == "context_date_gap_exceeded"


def test_append_news_impact_context_joins_display_only_columns_without_changing_recommendation(tmp_path):
    report_path = tmp_path / "news_impact.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "stock-news-impact.report.v1",
                "rows": [
                    {
                        "date": "2026-05-15",
                        "ticker": "005930",
                        "run_id": "run-1",
                        "final_score": 95.0,
                        "sector_neutral_score": 80.0,
                        "uncertainty_score": 12.0,
                        "top_event_type": "contract",
                        "top_reason": "large supply contract",
                        "why_may_be_wrong": "margin unknown",
                        "risk_flags": "low_liquidity",
                        "tradeability_status": "tradable",
                        "review_checklist": "not_advice;verify_url",
                        "top_evidence_url": "https://example.com/evidence",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pred_df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-05-15"),
                "Symbol": "005930.KS",
                "predicted_return": -2.5,
                "recommendation": recommendation_from_signal(None, -2.5),
            }
        ]
    )

    out = append_news_impact_context(pred_df, report_path)

    assert out.loc[0, "news_impact_final_score"] == 95.0
    assert out.loc[0, "news_impact_top_reason"] == "large supply contract"
    assert out.loc[0, "recommendation"] == recommendation_from_signal(None, -2.5)


def test_news_impact_context_preserves_expected_return_ranking_and_policy_columns(tmp_path):
    report_path = tmp_path / "news_impact.json"
    report_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"date": "2026-05-15", "ticker": "005930", "final_score": 100.0, "top_reason": "positive news"},
                    {"date": "2026-05-15", "ticker": "000660", "final_score": -100.0, "top_reason": "negative news"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pred_df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-05-15"),
                "Symbol": "005930.KS",
                "predicted_return": -3.0,
                "signal_score": 0.1,
                "recommendation": recommendation_from_signal(0.1, -3.0),
            },
            {
                "Date": pd.Timestamp("2026-05-15"),
                "Symbol": "000660.KS",
                "predicted_return": 3.0,
                "signal_score": 0.9,
                "recommendation": recommendation_from_signal(0.9, 3.0),
            },
        ]
    )
    before_order = pred_df.sort_values("predicted_return", ascending=False)["Symbol"].tolist()

    out = append_news_impact_context(pred_df, report_path)
    after_order = out.sort_values("predicted_return", ascending=False)["Symbol"].tolist()

    assert out["predicted_return"].tolist() == pred_df["predicted_return"].tolist()
    assert out["signal_score"].tolist() == pred_df["signal_score"].tolist()
    assert out["recommendation"].tolist() == pred_df["recommendation"].tolist()
    assert after_order == before_order == ["000660.KS", "005930.KS"]


def test_append_news_impact_context_missing_or_invalid_report_is_noop(tmp_path):
    pred_df = pd.DataFrame([{"Date": pd.Timestamp("2026-05-15"), "Symbol": "005930.KS"}])

    out = append_news_impact_context(pred_df, tmp_path / "missing.json")

    assert out.equals(pred_df)


def test_pipeline_exposes_optional_news_impact_report_flag_and_parameter():
    from inspect import signature

    from src.pipeline import build_cli_parser, run_pipeline

    parser = build_cli_parser()
    args = parser.parse_args(["--news-impact-report", "impact.json"])

    assert args.news_impact_report == "impact.json"
    assert "news_impact_report" in signature(run_pipeline).parameters


def test_append_generated_news_impact_context_scores_raw_news_and_disclosures_without_changing_policy():
    pred_df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-03-26"),
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "predicted_return": -1.5,
                "recommendation": recommendation_from_signal(None, -1.5),
            }
        ]
    )
    context_raw_df = pd.DataFrame(
        [
            {
                "Date": "2026-03-26",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 HBM 수요 증가",
                "body": "반도체 수요 개선",
                "url": "https://example.com/news",
            },
            {
                "Date": "2026-03-26",
                "Symbol": "005930.KS",
                "source_type": "disclosure",
                "title": "단일판매ㆍ공급계약체결",
                "body": "",
                "url": "https://example.com/dart",
            },
        ]
    )

    out = append_generated_news_impact_context(pred_df, context_raw_df)

    assert out.loc[0, "predicted_return"] == pred_df.loc[0, "predicted_return"]
    assert out.loc[0, "recommendation"] == pred_df.loc[0, "recommendation"]
    assert out.loc[0, "news_impact_final_score"] > 0
    assert out.loc[0, "news_impact_event_count"] == 2
    assert "HBM" in out.loc[0, "news_impact_top_reason"] or "공급계약" in out.loc[0, "news_impact_top_reason"]
    assert out.loc[0, "뉴스/공시 영향 점수"].endswith("점")
    assert "예측값 미반영" in out.loc[0, "뉴스/공시 영향 참고"]


def test_result_simple_includes_news_impact_display_columns_when_available():
    pred_df = pd.DataFrame(
        [
            {
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "recommendation": "보유",
                "predicted_close": 70000,
                "predicted_return": 1.23,
                "up_probability": 0.55,
                "confidence_score": 0.6,
                "history_direction_accuracy": 0.5,
                "뉴스/공시 영향 점수": "+42.0점",
                "뉴스/공시 영향 요약": "HBM 수요 증가",
                "뉴스/공시 영향 참고": "참고용",
            }
        ]
    )

    simple = build_result_simple(pred_df)

    assert simple.loc[0, "뉴스/공시 영향 점수"] == "+42.0점"
    assert simple.loc[0, "뉴스/공시 영향 요약"] == "HBM 수요 증가"
    assert simple.loc[0, "뉴스/공시 영향 참고"] == "참고용"


def _gemma_pred_df() -> pd.DataFrame:
    return pd.DataFrame([{"Date": "2026-06-16", "Symbol": "005930.KS", "종목명": "삼성전자"}])


def _gemma_context_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "2026-06-16",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 HBM 공급계약",
                "published_at": "2026-06-16T09:30:00+09:00",
                "provider": "naver",
                "url": "https://news.example/1",
                "raw_id": "n1",
            }
        ]
    )


def test_append_llm_news_impact_uses_report_json(tmp_path):
    def fake_run(inputs):
        report = Path(inputs.output_dir) / "report.json"
        report.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "date": "2026-06-16",
                            "ticker": "005930",
                            "news_disclosure_score": 42.0,
                            "top_reason": "공급계약",
                            "event_count": 1,
                            "risk_flags": "llm_judged",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return types.SimpleNamespace(artifact_paths={"report.json": report})

    out = append_llm_news_impact_context(
        _gemma_pred_df(),
        _gemma_context_df(),
        llm_config_path="configs/news_impact.gemma.example.json",
        symbols=["005930.KS"],
        symbol_name_map={"005930.KS": "삼성전자"},
        run_date="2026-06-16",
        _run_daily_pipeline=fake_run,
    )
    assert "news_impact_final_score" in out.columns
    assert float(out.iloc[0]["news_impact_final_score"]) == 42.0


def test_append_llm_news_impact_falls_back_on_error():
    def boom(inputs):
        raise RuntimeError("gemma server down")

    out = append_llm_news_impact_context(
        _gemma_pred_df(),
        _gemma_context_df(),
        llm_config_path="configs/news_impact.gemma.example.json",
        symbols=["005930.KS"],
        symbol_name_map={"005930.KS": "삼성전자"},
        run_date="2026-06-16",
        _run_daily_pipeline=boom,
    )
    # 폴백: 규칙 기반 표시 컬럼이 생성됨
    assert "뉴스/공시 영향 점수" in out.columns


def test_pipeline_accepts_news_impact_llm_config_flag():
    from inspect import signature

    from src.pipeline import build_cli_parser, run_pipeline

    parser = build_cli_parser()
    args = parser.parse_args(["--news-impact-llm-config", "configs/news_impact.gemma.example.json"])
    assert args.news_impact_llm_config == "configs/news_impact.gemma.example.json"
    assert "news_impact_llm_config" in signature(run_pipeline).parameters


def test_append_llm_news_impact_falls_back_on_empty_report(tmp_path):
    def fake_run_empty(inputs):
        report = Path(inputs.output_dir) / "report.json"
        report.write_text(json.dumps({"rows": []}), encoding="utf-8")
        return types.SimpleNamespace(artifact_paths={"report.json": report})

    out = append_llm_news_impact_context(
        _gemma_pred_df(),
        _gemma_context_df(),
        llm_config_path="configs/news_impact.gemma.example.json",
        symbols=["005930.KS"],
        symbol_name_map={"005930.KS": "삼성전자"},
        run_date="2026-06-16",
        _run_daily_pipeline=fake_run_empty,
    )
    # gemma가 유효 row를 못 내면(서버 다운 등) 규칙 기반으로 폴백
    assert "뉴스/공시 영향 점수" in out.columns
