from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureConfig:
    lookback_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 60])
    moving_average_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 20, 60])
    rsi_period: int = 14


@dataclass
class TrainingConfig:
    min_train_size: int = 252 * 3
    test_size: int = 252
    step_size: int = 126
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    random_state: int = 42


@dataclass
class SignalConfig:
    return_weight: float = 0.45
    up_prob_weight: float = 0.35
    rel_strength_weight: float = 0.20
    uncertainty_penalty: float = 0.25


@dataclass
class AppConfig:
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
