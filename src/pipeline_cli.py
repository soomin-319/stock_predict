"""CLI layer for the stock-predict pipeline.

Holds the argparse parser and the CLI/config override mapping so ``src/pipeline.py``
can focus on orchestration. ``src.pipeline`` re-imports these names, so the public
surface ``src.pipeline.build_cli_parser`` is preserved.
"""

from __future__ import annotations

import argparse

from src.data.fetch_real_data import DEFAULT_REAL_START_DATE


def _build_pipeline_overrides(
    min_value_traded: float | None,
    turnover_limit: float | None,
    min_up_probability: float | None,
    min_signal_score: float | None,
    min_external_coverage_ratio: float | None,
    min_investor_coverage_ratio: float | None,
    portfolio_value: float | None,
    max_daily_participation: float | None,
    max_positions_per_market_type: int | None,
    walk_forward_n_jobs: int | None,
    model_n_jobs: int | None,
    model_head_n_jobs: int | None,
) -> dict[str, dict]:
    cfg_overrides: dict[str, dict] = {"backtest": {}, "training": {}}
    if min_value_traded is not None:
        cfg_overrides["backtest"]["min_value_traded"] = float(min_value_traded)
    if turnover_limit is not None:
        cfg_overrides["backtest"]["turnover_limit"] = float(turnover_limit)
    if min_up_probability is not None:
        cfg_overrides["backtest"]["min_up_probability"] = float(min_up_probability)
    if min_signal_score is not None:
        cfg_overrides["backtest"]["min_signal_score"] = float(min_signal_score)
    if min_external_coverage_ratio is not None:
        cfg_overrides["backtest"]["min_external_coverage_ratio"] = float(min_external_coverage_ratio)
    if min_investor_coverage_ratio is not None:
        cfg_overrides["backtest"]["min_investor_coverage_ratio"] = float(min_investor_coverage_ratio)
    if portfolio_value is not None:
        cfg_overrides["backtest"]["portfolio_value"] = float(portfolio_value)
    if max_daily_participation is not None:
        cfg_overrides["backtest"]["max_daily_participation"] = float(max_daily_participation)
    if max_positions_per_market_type is not None:
        cfg_overrides["backtest"]["max_positions_per_market_type"] = int(max_positions_per_market_type)
    if walk_forward_n_jobs is not None:
        cfg_overrides["training"]["walk_forward_n_jobs"] = int(walk_forward_n_jobs)
    if model_n_jobs is not None:
        cfg_overrides["training"]["model_n_jobs"] = int(model_n_jobs)
    if model_head_n_jobs is not None:
        cfg_overrides["training"]["model_head_n_jobs"] = int(model_head_n_jobs)
    return {section: values for section, values in cfg_overrides.items() if values}


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock next-day prediction pipeline")
    parser.add_argument("--input", required=False, default="data/real_ohlcv.csv", help="OHLCV CSV path")
    parser.add_argument(
        "--output",
        default="result_detail.csv",
        help="Legacy option (CSV outputs are always saved as result_detail.csv and result_simple.csv under result/)",
    )
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with Symbol column")
    parser.add_argument("--report-json", default="pipeline_report.json", help="Pipeline summary JSON")
    parser.add_argument("--news-impact-report", default=None, help="Optional stock-news-impact JSON report for display-only context")
    parser.add_argument("--news-impact-llm-config", default=None, help="Optional llama.cpp/gemma LLM config for on-demand news-impact judging")
    parser.add_argument("--fetch-real", action="store_true", help="Fetch real OHLCV from yfinance before running")
    parser.add_argument("--disable-external", action="store_true", help="Disable external market feature download")
    parser.add_argument("--fetch-investor-context", action="store_true", help="Fetch investor flow context features (foreign/institution flows)")
    parser.add_argument("--disable-disclosure-context", action="store_true", help="Disable DART disclosure context")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key for AI news scoring")
    parser.add_argument("--openai-model", default=None, help="OpenAI model for AI news scoring")
    parser.add_argument("--naver-client-id", default=None, help="Naver News Search API client id")
    parser.add_argument("--naver-client-secret", default=None, help="Naver News Search API client secret")
    parser.add_argument("--dart-api-key", default=None, help="Deprecated legacy option kept for compatibility")
    parser.add_argument("--dart-corp-map-csv", default=None, help="Deprecated legacy option kept for compatibility")
    parser.add_argument("--config-json", default=None, help="Optional JSON file overriding nested AppConfig values")
    parser.add_argument("--min-value-traded", type=float, default=None, help="Minimum daily traded value filter for backtest/report")
    parser.add_argument("--turnover-limit", type=float, default=None, help="Override backtest turnover limit")
    parser.add_argument("--min-up-probability", type=float, default=None, help="Override backtest minimum up probability")
    parser.add_argument("--min-signal-score", type=float, default=None, help="Override backtest minimum signal score")
    parser.add_argument(
        "--min-external-coverage-ratio",
        type=float,
        default=None,
        help="Override backtest minimum external feature coverage ratio",
    )
    parser.add_argument(
        "--min-investor-coverage-ratio",
        type=float,
        default=None,
        help="Override backtest minimum investor context coverage ratio",
    )
    parser.add_argument("--portfolio-value", type=float, default=None, help="Backtest portfolio notional for liquidity capacity checks")
    parser.add_argument(
        "--max-daily-participation",
        type=float,
        default=None,
        help="Maximum share of daily traded value allowed per position",
    )
    parser.add_argument(
        "--max-positions-per-market-type",
        type=int,
        default=None,
        help="Maximum number of holdings allowed per market_type bucket",
    )
    parser.add_argument(
        "--issue-summary-symbols",
        nargs="*",
        default=None,
        help="Generate issue summaries only for specific symbols",
    )
    parser.add_argument("--walk-forward-n-jobs", type=int, default=None, help="Walk-forward fold worker count")
    parser.add_argument("--model-n-jobs", type=int, default=None, help="Per-model LightGBM worker count")
    parser.add_argument("--model-head-n-jobs", type=int, default=None, help="Parallel model head worker count")
    parser.add_argument(
        "--context-raw-event-n-jobs",
        type=int,
        default=None,
        help="Worker count for raw news/disclosure collection",
    )
    parser.add_argument("--issue-summary-n-jobs", type=int, default=1, help="Worker count for issue summary generation")
    parser.add_argument(
        "--real-symbols",
        nargs="*",
        default=None,
        help="Symbols used when --fetch-real is enabled (no auto KRX universe)",
    )
    parser.add_argument("--real-start", default=DEFAULT_REAL_START_DATE, help="Start date for real data fetch")
    parser.add_argument(
        "--auto-refresh-real",
        action="store_true",
        help="Refresh data in --input incrementally before running (explicit opt-in)",
    )
    parser.add_argument(
        "--add-symbols",
        nargs="*",
        default=None,
        help="Append user-entered stock codes/symbols into --input CSV (e.g., 005930 000660.KS)",
    )
    return parser
