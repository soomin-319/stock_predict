from __future__ import annotations

import src.pipeline as pipeline
from src.pipeline_cli import _build_pipeline_overrides, build_cli_parser


def test_build_cli_parser_is_reexported_from_pipeline_module():
    # src.pipeline.build_cli_parser is a documented public surface and must stay
    # importable as the same object so existing callers and monkeypatches keep working.
    assert pipeline.build_cli_parser is build_cli_parser

    parser = build_cli_parser()
    args = parser.parse_args([])

    assert args.input == "data/real_ohlcv.csv"
    assert args.report_json == "pipeline_report.json"
    assert args.disable_external is False


def test_build_pipeline_overrides_only_includes_provided_values():
    overrides = _build_pipeline_overrides(
        min_value_traded=1.0,
        turnover_limit=None,
        min_up_probability=None,
        min_signal_score=None,
        min_external_coverage_ratio=None,
        min_investor_coverage_ratio=None,
        portfolio_value=None,
        max_daily_participation=None,
        max_positions_per_market_type=None,
        walk_forward_n_jobs=2,
        model_n_jobs=None,
        model_head_n_jobs=None,
    )

    assert overrides == {
        "backtest": {"min_value_traded": 1.0},
        "training": {"walk_forward_n_jobs": 2},
    }


def test_build_pipeline_overrides_returns_empty_when_nothing_set():
    overrides = _build_pipeline_overrides(
        min_value_traded=None,
        turnover_limit=None,
        min_up_probability=None,
        min_signal_score=None,
        min_external_coverage_ratio=None,
        min_investor_coverage_ratio=None,
        portfolio_value=None,
        max_daily_participation=None,
        max_positions_per_market_type=None,
        walk_forward_n_jobs=None,
        model_n_jobs=None,
        model_head_n_jobs=None,
    )

    assert overrides == {}
