from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import app_config_to_dict, load_app_config
from src.data.investor_context import collect_context_raw_events
from src.models.lgbm_heads import MultiHeadStockModel
from src.pipeline import _prepare_pipeline_context


def test_pytest_conftest_does_not_force_repository_shared_temp_directory():
    root = Path(__file__).resolve().parents[1]
    shared_temp = (root / "result" / ".pytest_tmp").resolve()

    assert Path(tempfile.gettempdir()).resolve() != shared_temp


def test_config_rejects_unknown_nested_key():
    with pytest.raises(ValueError, match=r"training\.min_trian_size"):
        load_app_config(overrides={"training": {"min_trian_size": 10}})


def test_config_unknown_key_suggests_close_match():
    with pytest.raises(ValueError, match=r"training\.min_trian_size.*did you mean 'min_train_size'"):
        load_app_config(overrides={"training": {"min_trian_size": 10}})


def test_config_unknown_key_without_close_match_has_plain_error():
    with pytest.raises(ValueError, match=r"Unknown configuration key: training\.zzzz"):
        load_app_config(overrides={"training": {"zzzz": 10}})


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"training": {"min_train_size": 0}}, "training.min_train_size"),
        ({"training": {"min_train_size": 252, "test_size": 252}}, "training.min_train_size"),
        ({"training": {"step_size": 253, "test_size": 252}}, "training.step_size"),
        ({"training": {"quantiles": [0.1, 0.5]}}, "training.quantiles"),
        ({"training": {"config_schema_version": 2}}, "training.config_schema_version"),
        ({"feature": {"lookback_windows": [5, 1]}}, "feature.lookback_windows"),
        ({"feature": {"moving_average_windows": [0, 5]}}, "feature.moving_average_windows"),
        ({"feature": {"volatility_windows": [5, 5]}}, "feature.volatility_windows"),
        ({"feature": {"rsi_period": 0}}, "feature.rsi_period"),
        ({"signal": {"return_weight": -0.1}}, "signal.return_weight"),
        ({"signal": {"return_weight": 0.0, "up_prob_weight": 0.0}}, "signal weights"),
        ({"signal": {"uncertainty_penalty": -0.1}}, "signal.uncertainty_penalty"),
        ({"investment_criteria": {"rsi_buy_watch_low": 40.0, "rsi_buy_watch_high": 35.0}}, "investment_criteria.rsi_buy_watch"),
        ({"investment_criteria": {"rsi_overbought": 101.0}}, "investment_criteria.rsi_overbought"),
        ({"investment_criteria": {"near_52w_distance_threshold": 1.1}}, "investment_criteria.near_52w_distance_threshold"),
        ({"investment_criteria": {"leader_top_n": 0}}, "investment_criteria.leader_top_n"),
        ({"backtest": {"top_k": 0}}, "backtest.top_k"),
        ({"backtest": {"min_up_probability": 1.1}}, "backtest.min_up_probability"),
        ({"backtest": {"fee_bps": -0.1}}, "backtest.fee_bps"),
        ({"backtest": {"slippage_bps": -0.1}}, "backtest.slippage_bps"),
        ({"backtest": {"dynamic_slippage_bps": -0.1}}, "backtest.dynamic_slippage_bps"),
        ({"backtest": {"conservative_slippage_multiplier": 0.0}}, "backtest.conservative_slippage_multiplier"),
        ({"backtest": {"aggressive_slippage_multiplier": 0.0}}, "backtest.aggressive_slippage_multiplier"),
        ({"backtest": {"max_positions_per_market_type": 0}}, "backtest.max_positions_per_market_type"),
    ],
)
def test_config_rejects_invalid_ranges(overrides, expected):
    with pytest.raises(ValueError, match=expected):
        load_app_config(overrides=overrides)


def test_app_config_to_dict_includes_schema_version():
    assert app_config_to_dict(load_app_config())["config_schema_version"] == 1


def test_signal_config_exposes_recommendation_thresholds():
    cfg = load_app_config(
        overrides={
            "signal": {
                "recommendation_buy_threshold_pct": 3.5,
                "recommendation_sell_threshold_pct": -1.5,
            }
        }
    )

    signal_dict = app_config_to_dict(cfg)["signal"]
    assert signal_dict["recommendation_buy_threshold_pct"] == 3.5
    assert signal_dict["recommendation_sell_threshold_pct"] == -1.5


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"signal": {"recommendation_buy_threshold_pct": 0.0}}, "signal.recommendation_buy_threshold_pct"),
        ({"signal": {"recommendation_buy_threshold_pct": -0.1}}, "signal.recommendation_buy_threshold_pct"),
        ({"signal": {"recommendation_buy_threshold_pct": True}}, "signal.recommendation_buy_threshold_pct"),
        ({"signal": {"recommendation_sell_threshold_pct": 0.0}}, "signal.recommendation_sell_threshold_pct"),
        ({"signal": {"recommendation_sell_threshold_pct": 0.1}}, "signal.recommendation_sell_threshold_pct"),
        ({"signal": {"recommendation_sell_threshold_pct": False}}, "signal.recommendation_sell_threshold_pct"),
        (
            {
                "signal": {
                    "recommendation_buy_threshold_pct": 1.0,
                    "recommendation_sell_threshold_pct": 1.0,
                }
            },
            "signal.recommendation_sell_threshold_pct",
        ),
    ],
)
def test_signal_config_rejects_invalid_recommendation_thresholds(overrides, expected):
    with pytest.raises(ValueError, match=expected):
        load_app_config(overrides=overrides)


@pytest.mark.parametrize(
    "quantiles",
    [
        [0.1, 0.5],
        [0.1, 0.5, 0.5],
        [0.5, 0.1, 0.9],
        [0.0, 0.5, 0.9],
        [0.1, 0.5, 1.0],
    ],
)
def test_model_fit_rejects_invalid_quantiles_before_training(quantiles):
    model = MultiHeadStockModel()
    frame = pd.DataFrame({"feature": [1.0], "target_log_return": [0.1], "target_up": [1]})

    with pytest.raises(ValueError, match="quantiles"):
        model.fit(frame, ["feature"], quantiles)


def _context_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-05"]),
            "Symbol": ["005930.KS"],
            "Close": [100.0],
        }
    )


def test_pipeline_context_reports_raw_collection_failure(monkeypatch):
    def _raise(**kwargs):
        raise TimeoutError("context timed out")

    monkeypatch.setattr("src.pipeline.collect_context_raw_events", _raise)
    _, coverage, raw = _prepare_pipeline_context(
        data=_context_frame(),
        use_investor_context=False,
        enable_investor_disclosure=True,
        dart_api_key="key",
        dart_corp_map_csv=None,
        naver_client_id=None,
        naver_client_secret=None,
        context_raw_event_n_jobs=1,
    )

    assert raw.empty
    assert coverage["raw_events"]["status"] == "collection_failed"
    assert coverage["raw_events"]["error_types"] == ["TimeoutError"]
    assert coverage["raw_events"]["failed_symbols"] == ["005930.KS"]


def test_pipeline_context_distinguishes_no_events(monkeypatch):
    monkeypatch.setattr("src.pipeline.collect_context_raw_events", lambda **kwargs: pd.DataFrame())
    _, coverage, raw = _prepare_pipeline_context(
        data=_context_frame(),
        use_investor_context=False,
        enable_investor_disclosure=True,
        dart_api_key="key",
        dart_corp_map_csv=None,
        naver_client_id=None,
        naver_client_secret=None,
        context_raw_event_n_jobs=1,
    )

    assert raw.empty
    assert coverage["raw_events"]["status"] == "no_events"
    assert coverage["raw_events"]["error_types"] == []


def test_raw_context_status_reports_symbol_level_collection_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise TimeoutError("naver timed out")

    monkeypatch.setattr("src.data.investor_context.urlopen", _raise)
    raw, status = collect_context_raw_events(
        symbols=["005930.KS"],
        start="2026-06-05",
        end="2026-06-05",
        symbol_name_map={"005930.KS": "삼성전자"},
        naver_client_id="id",
        naver_client_secret="secret",
        raw_event_n_jobs=1,
        return_status=True,
    )

    assert raw.empty
    assert status["status"] == "collection_failed"
    assert status["failed_symbols"] == ["005930.KS"]
    assert status["error_types"] == ["TimeoutError"]
