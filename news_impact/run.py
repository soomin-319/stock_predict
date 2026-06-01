from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
from urllib.error import URLError

from news_impact.backtest_snapshots import (
    build_observations_from_snapshots,
    build_validation_report,
    load_price_bar_snapshot,
    load_signal_snapshot,
    write_validation_report,
)
from news_impact.llm_config import load_llm_config
from news_impact.llm_smoke import check_llama_cpp_prerequisites, run_llm_smoke
from news_impact.performance_validation import PerformanceCriteria
from news_impact.pipeline import DailyPipelineInputs, run_daily_pipeline
from news_impact.report import ReportRow, write_csv_report, write_json_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        return _run_report(args, parser)
    if args.command == "llm-smoke":
        return _run_llm_smoke(args)
    if args.command == "llm-preflight":
        return _run_llm_preflight(args)
    if args.command == "daily-run":
        return _run_daily_run(args)
    if args.command == "backtest-validate":
        return _run_backtest_validate(args)
    parser.error("missing command")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock-news-impact")
    subparsers = parser.add_subparsers(dest="command")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--csv")
    report_parser.add_argument("--json")
    smoke_parser = subparsers.add_parser("llm-smoke")
    smoke_parser.add_argument("--config", required=True)
    preflight_parser = subparsers.add_parser("llm-preflight")
    preflight_parser.add_argument("--config", required=True)
    preflight_parser.add_argument("--model-path")
    preflight_parser.add_argument("--grammar-path")
    daily_parser = subparsers.add_parser("daily-run")
    daily_parser.add_argument("--run-date", required=True)
    daily_parser.add_argument("--watchlist", required=True)
    daily_parser.add_argument("--company-master", required=True)
    daily_parser.add_argument("--input-fixture", required=True)
    daily_parser.add_argument("--output-dir", required=True)
    daily_parser.add_argument(
        "--semantic-clustering",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    daily_parser.add_argument(
        "--no-semantic-clustering",
        dest="semantic_clustering",
        action="store_false",
    )
    daily_parser.set_defaults(semantic_clustering=True)
    daily_parser.add_argument("--llm-config")
    validate_parser = subparsers.add_parser("backtest-validate")
    validate_parser.add_argument("--signals", required=True)
    validate_parser.add_argument("--prices", required=True)
    validate_parser.add_argument("--output", required=True)
    validate_parser.add_argument("--round-trip-cost-bps", type=float, default=26.0)
    validate_parser.add_argument("--min-overall-samples", type=int, default=250)
    validate_parser.add_argument("--min-bucket-samples", type=int, default=30)
    validate_parser.add_argument("--min-rank-ic", type=float, default=0.0)
    validate_parser.add_argument("--min-top-bottom-spread", type=float, default=0.0)
    validate_parser.add_argument("--include-missing", action="store_true")
    return parser


def _run_report(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if not args.csv and not args.json:
        parser.error("report requires --csv and/or --json output")
    rows = _load_report_rows(args.input)
    if args.csv:
        write_csv_report(rows, args.csv)
    if args.json:
        write_json_report(rows, args.json)
    return 0


def _run_llm_smoke(args: argparse.Namespace) -> int:
    config = load_llm_config(args.config)
    try:
        result = run_llm_smoke(config)
    except URLError as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "provider": config.provider,
                    "base_url": config.base_url,
                    "model": config.model,
                    "error": str(error.reason),
                    "hint": "Start llama.cpp server and confirm /v1/models serves the configured alias.",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _run_llm_preflight(args: argparse.Namespace) -> int:
    config = load_llm_config(args.config)
    result = check_llama_cpp_prerequisites(
        config,
        model_path=args.model_path,
        grammar_path=args.grammar_path,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ready" else 1


def _run_daily_run(args: argparse.Namespace) -> int:
    result = run_daily_pipeline(
        DailyPipelineInputs(
            run_date=args.run_date,
            watchlist_path=args.watchlist,
            company_master_path=args.company_master,
            input_fixture_path=args.input_fixture,
            output_dir=args.output_dir,
            semantic_clustering=args.semantic_clustering,
            llm_config_path=args.llm_config,
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": result.run_id,
                "artifacts": {
                    name: str(path)
                    for name, path in sorted(result.artifact_paths.items())
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_backtest_validate(args: argparse.Namespace) -> int:
    observations = build_observations_from_snapshots(
        load_signal_snapshot(args.signals),
        load_price_bar_snapshot(args.prices),
        round_trip_cost_bps=args.round_trip_cost_bps,
        include_missing=args.include_missing,
    )
    report = build_validation_report(
        observations,
        criteria=PerformanceCriteria(
            min_overall_samples=args.min_overall_samples,
            min_bucket_samples=args.min_bucket_samples,
            min_rank_ic=args.min_rank_ic,
            min_top_bottom_spread=args.min_top_bottom_spread,
        ),
    )
    write_validation_report(report, args.output)
    print(
        json.dumps(
            {
                "status": "passed" if report["validation"]["passed"] else "failed",
                "output": str(Path(args.output)),
                "sample_size": report["overall"]["sample_size"],
                "failed_checks": report["validation"]["failed_checks"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["validation"]["passed"] else 1


def _load_report_rows(path: str | Path) -> list[ReportRow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    if not isinstance(raw_rows, list):
        raise ValueError("report input must be a JSON array or object with rows")
    return [ReportRow(**_normalize_row(row)) for row in raw_rows]


def _normalize_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        raise ValueError("report row must be a JSON object")
    normalized = dict(row)
    for key in ("risk_flags", "review_checklist"):
        value = normalized.get(key, [])
        if isinstance(value, str):
            normalized[key] = tuple(item for item in value.split(";") if item)
        else:
            normalized[key] = tuple(value)
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
