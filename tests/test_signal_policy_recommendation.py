from src.domain.signal_policy import recommendation_from_signal


def test_recommendation_based_on_predicted_return_thresholds():
    assert recommendation_from_signal(None, 1.1) == "매수"
    assert recommendation_from_signal(None, 5.0) == "매수"
    assert recommendation_from_signal(None, -1.1) == "매도"
    assert recommendation_from_signal(None, -5.0) == "매도"
    assert recommendation_from_signal(None, 1.0) == "관망"
    assert recommendation_from_signal(None, -1.0) == "관망"
    assert recommendation_from_signal(None, 0.0) == "관망"
    assert recommendation_from_signal(None, 0.5) == "관망"


def test_recommendation_ignores_signal_and_probability_for_decision():
    # signal_score, up_probability, uncertainty_score는 권고에 영향 없음
    assert recommendation_from_signal(0.1, 2.0, up_probability=0.3, uncertainty_score=0.9) == "매수"
    assert recommendation_from_signal(0.9, -2.0, up_probability=0.9, uncertainty_score=0.1) == "매도"


def test_recommendation_returns_hold_when_return_is_missing():
    assert recommendation_from_signal(None, None) == "관망"
