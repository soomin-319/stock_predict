from src.recommendation.close_betting import CloseBettingRecommendation, format_recommendation_message

__all__ = [
    "CloseBettingRecommendation",
    "RealTimeCloseBettingRecommendationService",
    "format_recommendation_message",
]


def __getattr__(name: str):
    if name == "RealTimeCloseBettingRecommendationService":
        from src.recommendation.realtime_close_betting import RealTimeCloseBettingRecommendationService

        return RealTimeCloseBettingRecommendationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
