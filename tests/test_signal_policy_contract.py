import pandas as pd

from src.domain.signal_policy import recommendation_from_signal
from src.reports.news_impact_context import append_generated_news_impact_context
from src.reports.pm_report import build_pm_report


def test_recommendation_depends_only_on_next_day_predicted_return():
    assert recommendation_from_signal(0.99, 1.99, 0.99, 0.0) == "관망"
    assert recommendation_from_signal(0.01, 2.01, 0.01, 1.0) == "매수"
    assert recommendation_from_signal(0.99, -2.0, 0.99, 0.0) == "매도"


def test_generated_news_context_does_not_mutate_policy_columns():
    base = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-03-26"),
                "Symbol": "005930.KS",
                "symbol_name": "삼성전자",
                "predicted_return": 2.5,
                "predicted_close": 70000.0,
                "signal_score": 0.2,
                "recommendation": "매수",
                "portfolio_action": "신규매수",
            }
        ]
    )
    context = pd.DataFrame(
        [
            {
                "Date": "2026-03-26",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 단기 악재 보도",
                "body": "악화 하락 급락",
                "url": "https://example.com/news",
            }
        ]
    )

    out = append_generated_news_impact_context(base, context)

    for column in ["predicted_return", "predicted_close", "signal_score", "recommendation", "portfolio_action"]:
        assert out.loc[0, column] == base.loc[0, column]


def test_pm_report_top_buy_candidates_are_ordered_by_expected_return():
    pred_df = pd.DataFrame(
        [
            {"Symbol": "A", "recommendation": "매수", "predicted_return": 1.0, "confidence_score": 0.9, "signal_score": 0.99},
            {"Symbol": "B", "recommendation": "매수", "predicted_return": 3.0, "confidence_score": 0.5, "signal_score": 0.10},
        ]
    )

    report = build_pm_report(pred_df, report={})

    assert [row["Symbol"] for row in report["top_buy_candidates"]] == ["B", "A"]
