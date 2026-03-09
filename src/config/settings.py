from dataclasses import dataclass, field
from typing import List


@dataclass
class UniverseConfig:
    name: str = "KOSPI200_KOSDAQ150"
    expected_size: int = 350
    default_kospi_count: int = 200
    default_kosdaq_count: int = 150


@dataclass
class FeatureConfig:
    lookback_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 60])
    moving_average_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 20, 60])
    rsi_period: int = 14
    cci_period: int = 20
    stochastic_period: int = 14


@dataclass
class ExternalFeatureConfig:
    enabled: bool = True
    market_symbols: List[str] = field(
        default_factory=lambda: ["^KS11", "^KQ11", "^GSPC", "^IXIC", "^SOX", "^VIX", "KRW=X", "^TNX"]
    )


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
class BacktestConfig:
    top_k: int = 20
    fee_bps: float = 10.0
    slippage_bps: float = 5.0


@dataclass
class AppConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    external: ExternalFeatureConfig = field(default_factory=ExternalFeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
