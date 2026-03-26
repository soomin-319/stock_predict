from src.domain.signal_policy import recommendation_from_signal


def test_recommendation_uses_hold_between_minus_two_and_plus_two_percent_when_context_missing():
    assert recommendation_from_signal(0.3, 2.2) == "매수"
    assert recommendation_from_signal(0.3, -2.2) == "매도"
    assert recommendation_from_signal(0.3, 2.0) == "관망"
    assert recommendation_from_signal(0.3, -1.8) == "관망"
