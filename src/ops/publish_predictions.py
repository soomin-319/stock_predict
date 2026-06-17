from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from src.ops.published_store import (
    PublishMeta,
    copy_published_set,
    resolve_published_dir,
    update_index,
    write_publish_meta,
)
from src.reports.run_artifacts import resolve_latest_run_dir

_LOGGER = logging.getLogger(__name__)


def publish_artifacts(
    *,
    run_dir: str | Path,
    published_root: str | Path,
    trading_date: str,
    news_mode: str,
    source_run_id: str,
    symbol_count: int,
    git_commit: str | None = None,
    git_branch: str | None = None,
    generated_at_kst: str | None = None,
) -> PublishMeta:
    run_dir = Path(run_dir)
    published_root = Path(published_root)
    meta = PublishMeta(
        generated_at_kst=generated_at_kst
        or datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        trading_date=str(trading_date),
        news_mode=str(news_mode),
        source_run_id=str(source_run_id),
        symbol_count=int(symbol_count),
        git_commit=git_commit,
        git_branch=git_branch,
    )
    for dest in (
        resolve_published_dir(published_root, None),
        resolve_published_dir(published_root, meta.trading_date),
    ):
        copy_published_set(run_dir, dest)
        write_publish_meta(dest, meta)
    update_index(published_root, meta)
    return meta


def infer_trading_date(run_dir: str | Path) -> str:
    detail_path = Path(run_dir) / "csv" / "result_detail.csv"
    if not detail_path.exists():
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    df = pd.read_csv(detail_path, encoding="utf-8-sig")
    if "Date" not in df.columns or df.empty:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if dates.empty:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    return dates.max().strftime("%Y-%m-%d")


def ensure_operational_manifest(manifest: dict | None) -> None:
    ok = bool(
        manifest
        and manifest.get("promoted") is True
        and manifest.get("status") in {"pass", "warning"}
    )
    if not ok:
        raise ValueError(
            f"운영 산출물이 아니어서 publish를 중단합니다 (manifest={manifest}). "
            "파이프라인 상태를 확인하세요."
        )


GEMMA_CONFIG = "configs/news_impact.gemma.example.json"


def _news_config_for_mode(news_mode: str) -> str | None:
    return GEMMA_CONFIG if news_mode == "gemma" else None


def _default_pipeline_fn(project_root: Path) -> Callable[..., dict[str, Any]]:
    from src.data.fetch_real_data import append_real_ohlcv_csv, save_real_ohlcv_csv
    # Imports deferred to call time to avoid importing the heavy pipeline graph
    # at module load and to sidestep circular imports.
    from src.data.cli_refresh import (
        fallback_symbols_from_input_or_default as _fallback_symbols_from_input_or_default,
        resolve_incremental_fetch_start as _resolve_incremental_fetch_start,
    )
    from src.pipeline import run_pipeline

    input_csv = str(project_root / "data" / "real_ohlcv.csv")

    def _run(news_impact_llm_config: str | None, full_refresh: bool, config_json: str | None = None) -> dict[str, Any]:
        symbols = _fallback_symbols_from_input_or_default(input_csv)
        if full_refresh:
            save_real_ohlcv_csv(input_csv, symbols=symbols, start="2020-01-01")
        else:
            start = _resolve_incremental_fetch_start(input_csv, "2020-01-01")
            append_real_ohlcv_csv(input_csv, symbols=symbols, start=start)
        return run_pipeline(
            input_csv=input_csv,
            output_csv="result_detail.csv",
            report_json="pipeline_report.json",
            use_external=False,
            use_investor_context=True,
            news_impact_llm_config=news_impact_llm_config,
            config_json=config_json,
        )

    return _run


def _default_git_fn(project_root: Path) -> Callable[[list[str]], None]:
    def _git(args: list[str]) -> None:
        subprocess.run(["git", *args], cwd=project_root, check=True)

    return _git


def _symbol_count(run_dir: Path) -> int:
    path = Path(run_dir) / "csv" / "result_simple.csv"
    try:
        return int(len(pd.read_csv(path, encoding="utf-8-sig")))
    except Exception:
        return 0


def _effective_news_mode(report: dict[str, Any]) -> str:
    flags = report.get("news_impact") if isinstance(report.get("news_impact"), dict) else {}
    if str(flags.get("mode", "")).lower() in {"rule", "rule_based", "heuristic"}:
        return "rule_based"
    return "gemma"


def run_publish(
    args: "argparse.Namespace",
    *,
    project_root: str | Path,
    pipeline_fn: Callable[..., dict[str, Any]] | None = None,
    git_fn: Callable[[list[str]], None] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    result_root = project_root / "result"
    published_root = project_root / "published"
    pipeline_fn = pipeline_fn or _default_pipeline_fn(project_root)
    git_fn = git_fn or _default_git_fn(project_root)

    news_config = _news_config_for_mode(args.news_mode)
    report = pipeline_fn(
        news_impact_llm_config=news_config,
        full_refresh=bool(args.full_refresh),
        config_json=getattr(args, "config_json", None),
    )
    manifest = report.get("manifest") if isinstance(report, dict) else None
    ensure_operational_manifest(manifest)

    run_dir = resolve_latest_run_dir(result_root)
    if run_dir is None:
        raise FileNotFoundError(f"run dir를 찾을 수 없습니다: {result_root}")
    if not manifest.get("run_id"):
        _LOGGER.warning("manifest에 run_id가 없어 source_run_id가 비어 publish됩니다.")
    trading_date = infer_trading_date(run_dir)
    symbol_count = _symbol_count(run_dir)
    news_mode = "rule_based" if news_config is None else _effective_news_mode(report)

    meta = publish_artifacts(
        run_dir=run_dir,
        published_root=published_root,
        trading_date=trading_date,
        news_mode=news_mode,
        source_run_id=str(manifest.get("run_id", "")),
        symbol_count=symbol_count,
        git_branch="main",
    )

    if not args.no_push and not args.dry_run:
        git_fn(["add", "published"])
        git_fn(["commit", "-m", f"chore(publish): {meta.trading_date} predictions ({meta.news_mode})"])
        git_fn(["push"])

    return meta.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="기본 200종목 예측을 published/에 게시")
    parser.add_argument("--news-mode", choices=["gemma", "rule"], default="gemma")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-json", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    meta = run_publish(args, project_root=project_root)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
