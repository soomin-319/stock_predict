from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List



@dataclass
class UniverseConfig:
    name: str = "KOSPI200_KOSDAQ150"
    expected_size: int = 350


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
        default_factory=lambda: ["^KS11", "^KQ11", "^GSPC", "^IXIC", "NQ=F", "^SOX", "^VIX", "KRW=X", "^TNX"]
    )


@dataclass
class TrainingConfig:
    min_train_size: int = 252 * 3
    test_size: int = 252
    step_size: int = 126
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    random_state: int = 42
    model_n_jobs: int = -1
    walk_forward_n_jobs: int = 1
    use_gpu: bool = False


@dataclass
class SignalConfig:
    return_weight: float = 0.45
    up_prob_weight: float = 0.35
    rel_strength_weight: float = 0.20
    uncertainty_penalty: float = 0.25


@dataclass
class InvestmentCriteriaConfig:
    top_turnover_rank: int = 15
    high_conviction_net_buy_krw: float = 100_000_000_000.0  # 1,000억
    rsi_buy_watch_low: float = 30.0
    rsi_buy_watch_high: float = 35.0
    rsi_overbought: float = 70.0
    nasdaq_tailwind_threshold: float = 0.01
    nasdaq_headwind_threshold: float = -0.01
    near_52w_distance_threshold: float = 0.03
    leader_top_n: int = 3
    leader_min_co_movers: int = 2
    leader_min_return: float = 0.0


@dataclass
class BacktestConfig:
    top_k: int = 20
    portfolio_value: float = 1_000_000_000.0
    max_daily_participation: float = 0.10
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    dynamic_slippage_bps: float = 10.0
    conservative_slippage_multiplier: float = 1.5
    aggressive_slippage_multiplier: float = 0.75
    min_up_probability: float = 0.50
    min_signal_score: float = 0.0
    turnover_limit: float = 0.5
    min_value_traded: float = 3_000_000_000.0
    min_external_coverage_ratio: float = 0.0
    min_investor_coverage_ratio: float = 0.5
    max_positions_per_market_type: int = 12


@dataclass
class AppConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    external: ExternalFeatureConfig = field(default_factory=ExternalFeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    investment_criteria: InvestmentCriteriaConfig = field(default_factory=InvestmentCriteriaConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _merge_dataclass_config(instance, overrides: dict):
    valid_fields = {f.name: f for f in fields(instance)}
    for key, value in overrides.items():
        if key not in valid_fields:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass_config(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_app_config(config_path: str | Path | None = None, overrides: dict | None = None) -> AppConfig:
    cfg = AppConfig()
    payload: dict = {}
    if config_path:
        path = Path(config_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload:
        _merge_dataclass_config(cfg, payload)
    if overrides:
        _merge_dataclass_config(cfg, overrides)
    return cfg


def app_config_to_dict(cfg: AppConfig) -> dict:
    return asdict(cfg)
