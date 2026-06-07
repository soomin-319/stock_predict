from src.reports.context_policy import evaluate_context_policy


def test_context_date_matches_prediction_policy():
    result = evaluate_context_policy("2026-06-05", "2026-06-07", max_gap_days=3)

    assert result.allowed is True
    assert result.gap_days == 2
    assert result.reason is None


def test_stale_context_is_excluded():
    result = evaluate_context_policy("2023-08-10", "2026-06-07", max_gap_days=3)

    assert result.allowed is False
    assert result.reason == "context_date_gap_exceeded"


def test_missing_context_date_is_excluded():
    result = evaluate_context_policy("2026-06-05", None)

    assert result.allowed is False
    assert result.reason == "missing_context_date"
