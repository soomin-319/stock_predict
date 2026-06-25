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


def test_build_cli_parser_accepts_shared_llm_config():
    parser = build_cli_parser()
    args = parser.parse_args(["--llm-config", "configs/news_impact.openai.example.json"])

    assert args.llm_config == "configs/news_impact.openai.example.json"


def test_resolve_effective_llm_options_shared_config_drives_both_consumers(tmp_path, monkeypatch):
    import json

    cfg_path = tmp_path / "llm.json"
    cfg_path.write_text(
        json.dumps(
            {
                "llm_provider": "openai",
                "llm_base_url": "https://api.openai.com/v1",
                "llm_model": "gpt-5-mini",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    resolved = pipeline._resolve_effective_llm_options(
        llm_config=str(cfg_path),
        news_impact_llm_config=None,
        openai_api_key=None,
        openai_model=None,
    )

    assert resolved.issue_summary_provider == "openai"
    assert resolved.issue_summary_base_url == "https://api.openai.com/v1"
    assert resolved.issue_summary_model == "gpt-5-mini"
    assert resolved.issue_summary_api_key == "sk-test"
    assert resolved.news_impact_llm_config == str(cfg_path)


def test_resolve_effective_llm_options_keeps_rule_default_without_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    resolved = pipeline._resolve_effective_llm_options(
        llm_config=None,
        news_impact_llm_config=None,
        openai_api_key=None,
        openai_model=None,
    )

    assert resolved.issue_summary_provider == "openai"
    assert resolved.issue_summary_model is None
    assert resolved.issue_summary_api_key is None
    assert resolved.news_impact_llm_config is None


def test_resolve_effective_llm_options_backcompat_openai_flags_still_work():
    resolved = pipeline._resolve_effective_llm_options(
        llm_config=None,
        news_impact_llm_config=None,
        openai_api_key="sk-legacy",
        openai_model="gpt-test",
    )

    assert resolved.issue_summary_provider == "openai"
    assert resolved.issue_summary_model == "gpt-test"
    assert resolved.issue_summary_api_key == "sk-legacy"
    assert resolved.news_impact_llm_config is None
