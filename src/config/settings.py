from __future__ import annotations

import difflib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List



@dataclass
class UniverseConfig:
    name: str = "KOSPI200"
    expected_size: int = 200


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
    model_head_n_jobs: int = 1
    walk_forward_n_jobs: int = -1
    use_gpu: bool = False
    early_stopping_rounds: int = 0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    min_child_samples: int = 20
    # Gap between train end and validation start to prevent look-ahead leakage
    # from the next-day target overlapping the validation window.
    purge_gap_days: int = 1
    # Additional buffer after purge to dampen serial-correlation carryover.
    embargo_days: int = 0
    # 최종 모델 학습에 사용할 최근 거래일 수 (0 = 전체 히스토리 사용)
    final_model_lookback_days: int = 252 * 3
    # Walk-forward 학습에 사용할 최근 거래일 수 (0 = 확장창)
    walk_forward_lookback_days: int = 0


@dataclass
class SignalConfig:
    return_weight: float = 0.65  # 과거 rel_strength_weight(0.20)는 norm_return 중복이라 여기로 흡수
    up_prob_weight: float = 0.35
    uncertainty_penalty: float = 0.25
    recommendation_buy_threshold_pct: float = 2.0
    recommendation_sell_threshold_pct: float = -2.0


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
    config_schema_version: int = 1
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    external: ExternalFeatureConfig = field(default_factory=ExternalFeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    investment_criteria: InvestmentCriteriaConfig = field(default_factory=InvestmentCriteriaConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _merge_dataclass_config(instance, overrides: dict, path: str = ""):
    valid_fields = {f.name: f for f in fields(instance)}
    for key, value in overrides.items():
        field_path = f"{path}.{key}" if path else key
        if key not in valid_fields:
            message = f"Unknown configuration key: {field_path}"
            suggestion = difflib.get_close_matches(key, valid_fields.keys(), n=1)
            if suggestion:
                message += f"; did you mean '{suggestion[0]}'?"
            raise ValueError(message)
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section must be an object: {field_path}")
            _merge_dataclass_config(current, value, field_path)
        else:
            setattr(instance, key, value)
    return instance


def _validate_positive(value, path: str, *, allow_zero: bool = False) -> None:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path} must be {comparator}, got {value!r}")


def _validate_ratio(value, path: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise ValueError(f"{path} must be between 0 and 1, got {value!r}")


def _validate_percent(value, path: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100:
        raise ValueError(f"{path} must be between 0 and 100, got {value!r}")


def _validate_positive_int(value, path: str, *, allow_zero: bool = False) -> None:
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        comparator = "non-negative integer" if allow_zero else "positive integer"
        raise ValueError(f"{path} must be a {comparator}, got {value!r}")


def _validate_number(value, path: str, *, allow_zero: bool = False) -> None:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        comparator = "non-negative number" if allow_zero else "positive number"
        raise ValueError(f"{path} must be a {comparator}, got {value!r}")


def _validate_negative_number(value, path: str) -> None:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0
    if not valid:
        raise ValueError(f"{path} must be a negative number, got {value!r}")


def _validate_positive_int_windows(values: list[int], path: str) -> None:
    valid = (
        isinstance(values, list)
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values)
        and len(values) > 0
        and len(set(values)) == len(values)
        and values == sorted(values)
    )
    if not valid:
        raise ValueError(f"{path} must contain unique increasing positive integers, got {values!r}")


def _validate_app_config(cfg: AppConfig) -> None:
    if cfg.config_schema_version != 1:
        raise ValueError(f"config_schema_version must be 1, got {cfg.config_schema_version!r}")
    _validate_positive_int(cfg.universe.expected_size, "universe.expected_size")
    _validate_positive_int_windows(cfg.feature.lookback_windows, "feature.lookback_windows")
    _validate_positive_int_windows(cfg.feature.moving_average_windows, "feature.moving_average_windows")
    _validate_positive_int_windows(cfg.feature.volatility_windows, "feature.volatility_windows")
    for name in ("rsi_period", "cci_period", "stochastic_period"):
        _validate_positive_int(getattr(cfg.feature, name), f"feature.{name}")
    if not isinstance(cfg.external.enabled, bool):
        raise ValueError(f"external.enabled must be a boolean, got {cfg.external.enabled!r}")
    if not isinstance(cfg.external.market_symbols, list) or not all(
        isinstance(symbol, str) and symbol for symbol in cfg.external.market_symbols
    ):
        raise ValueError(f"external.market_symbols must contain non-empty strings, got {cfg.external.market_symbols!r}")

    for name in ("min_train_size", "test_size", "step_size"):
        _validate_positive_int(getattr(cfg.training, name), f"training.{name}")
    if cfg.training.min_train_size <= cfg.training.test_size:
        raise ValueError(
            "training.min_train_size must be greater than training.test_size "
            f"for walk-forward validation, got {cfg.training.min_train_size!r} <= {cfg.training.test_size!r}"
        )
    if cfg.training.step_size > cfg.training.test_size:
        raise ValueError(
            "training.step_size must be less than or equal to training.test_size, "
            f"got {cfg.training.step_size!r} > {cfg.training.test_size!r}"
        )
    for name in (
        "purge_gap_days",
        "embargo_days",
        "final_model_lookback_days",
        "walk_forward_lookback_days",
        "early_stopping_rounds",
    ):
        _validate_positive_int(getattr(cfg.training, name), f"training.{name}", allow_zero=True)
    _validate_positive_int(cfg.training.min_child_samples, "training.min_child_samples")
    for name in ("reg_alpha", "reg_lambda"):
        _validate_number(getattr(cfg.training, name), f"training.{name}", allow_zero=True)
    quantiles = cfg.training.quantiles
    valid_quantile_values = isinstance(quantiles, list) and all(
        isinstance(q, (int, float)) and not isinstance(q, bool) and 0 < q < 1 for q in quantiles
    )
    if (
        not valid_quantile_values
        or len(quantiles) < 3
        or len(set(quantiles)) != len(quantiles)
        or quantiles != sorted(quantiles)
    ):
        raise ValueError(f"training.quantiles must contain at least 3 unique increasing values in (0, 1), got {quantiles!r}")

    for name in ("return_weight", "up_prob_weight", "uncertainty_penalty"):
        _validate_number(getattr(cfg.signal, name), f"signal.{name}", allow_zero=True)
    primary_weight_sum = cfg.signal.return_weight + cfg.signal.up_prob_weight
    if primary_weight_sum <= 0:
        raise ValueError("signal weights must include at least one positive primary weight")
    _validate_number(cfg.signal.recommendation_buy_threshold_pct, "signal.recommendation_buy_threshold_pct")
    _validate_negative_number(
        cfg.signal.recommendation_sell_threshold_pct,
        "signal.recommendation_sell_threshold_pct",
    )
    if cfg.signal.recommendation_sell_threshold_pct >= cfg.signal.recommendation_buy_threshold_pct:
        raise ValueError(
            "signal.recommendation_sell_threshold_pct must be less than "
            "signal.recommendation_buy_threshold_pct"
        )

    _validate_positive_int(cfg.investment_criteria.top_turnover_rank, "investment_criteria.top_turnover_rank")
    _validate_positive(cfg.investment_criteria.high_conviction_net_buy_krw, "investment_criteria.high_conviction_net_buy_krw", allow_zero=True)
    _validate_percent(cfg.investment_criteria.rsi_buy_watch_low, "investment_criteria.rsi_buy_watch_low")
    _validate_percent(cfg.investment_criteria.rsi_buy_watch_high, "investment_criteria.rsi_buy_watch_high")
    _validate_percent(cfg.investment_criteria.rsi_overbought, "investment_criteria.rsi_overbought")
    if cfg.investment_criteria.rsi_buy_watch_low > cfg.investment_criteria.rsi_buy_watch_high:
        raise ValueError(
            "investment_criteria.rsi_buy_watch_low must be less than or equal to "
            "investment_criteria.rsi_buy_watch_high"
        )
    if cfg.investment_criteria.rsi_buy_watch_high > cfg.investment_criteria.rsi_overbought:
        raise ValueError(
            "investment_criteria.rsi_buy_watch_high must be less than or equal to "
            "investment_criteria.rsi_overbought"
        )
    if cfg.investment_criteria.nasdaq_headwind_threshold >= cfg.investment_criteria.nasdaq_tailwind_threshold:
        raise ValueError(
            "investment_criteria.nasdaq_headwind_threshold must be less than "
            "investment_criteria.nasdaq_tailwind_threshold"
        )
    _validate_ratio(cfg.investment_criteria.near_52w_distance_threshold, "investment_criteria.near_52w_distance_threshold")
    _validate_positive_int(cfg.investment_criteria.leader_top_n, "investment_criteria.leader_top_n")
    _validate_positive_int(cfg.investment_criteria.leader_min_co_movers, "investment_criteria.leader_min_co_movers")
    if not isinstance(cfg.investment_criteria.leader_min_return, (int, float)) or isinstance(
        cfg.investment_criteria.leader_min_return, bool
    ):
        raise ValueError(
            f"investment_criteria.leader_min_return must be numeric, got {cfg.investment_criteria.leader_min_return!r}"
        )

    _validate_positive_int(cfg.backtest.top_k, "backtest.top_k")
    _validate_positive(cfg.backtest.portfolio_value, "backtest.portfolio_value")
    _validate_ratio(cfg.backtest.max_daily_participation, "backtest.max_daily_participation")
    for name in ("fee_bps", "slippage_bps", "dynamic_slippage_bps"):
        _validate_positive(getattr(cfg.backtest, name), f"backtest.{name}", allow_zero=True)
    for name in ("conservative_slippage_multiplier", "aggressive_slippage_multiplier"):
        _validate_positive(getattr(cfg.backtest, name), f"backtest.{name}")
    _validate_ratio(cfg.backtest.min_up_probability, "backtest.min_up_probability")
    _validate_ratio(cfg.backtest.turnover_limit, "backtest.turnover_limit")
    _validate_ratio(cfg.backtest.min_external_coverage_ratio, "backtest.min_external_coverage_ratio")
    _validate_ratio(cfg.backtest.min_investor_coverage_ratio, "backtest.min_investor_coverage_ratio")
    _validate_positive_int(cfg.backtest.max_positions_per_market_type, "backtest.max_positions_per_market_type")


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
    _validate_app_config(cfg)
    return cfg


def app_config_to_dict(cfg: AppConfig) -> dict:
    return asdict(cfg)
