import json
from pathlib import Path

import pandas as pd

from src.domain.signal_policy import recommendation_from_signal
from src.reports.news_impact_context import append_news_impact_context


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
