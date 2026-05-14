from src.domain.signal_policy import recommendation_from_signal


def test_recommendation_based_on_predicted_return_thresholds():
    assert recommendation_from_signal(None, 2.1) == "매수"
    assert recommendation_from_signal(None, 5.0) == "매수"
    assert recommendation_from_signal(None, -2.0) == "매도"
    assert recommendation_from_signal(None, -5.0) == "매도"
    assert recommendation_from_signal(None, 2.0) == "관망"
    assert recommendation_from_signal(None, -1.9) == "관망"
    assert recommendation_from_signal(None, 0.0) == "관망"
    assert recommendation_from_signal(None, 1.9) == "관망"


def test_recommendation_uses_signal_probability_and_uncertainty_when_available():
    assert recommendation_from_signal(0.7, 0.8, up_probability=0.7, uncertainty_score=0.2) == "매수"
    assert recommendation_from_signal(0.2, 2.0, up_probability=0.6, uncertainty_score=0.2) == "매도"
    assert recommendation_from_signal(0.5, 1.2, up_probability=0.52, uncertainty_score=0.7) == "관망"


def test_recommendation_returns_hold_when_return_is_missing():
    assert recommendation_from_signal(None, None) == "관망"
