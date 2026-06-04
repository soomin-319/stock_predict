from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

from news_impact.deduper import (
    ClusteredItem,
    assign_cluster_ids,
    dedupe_disclosures,
    dedupe_news_items,
)
from news_impact.impact_judge import (
    LLM_REQUIRED_KEYS,
    NewsAnalysisInput,
    build_news_user_prompt,
    build_system_prompt,
    detect_prompt_injection,
    judgment_to_impact_event,
)
from news_impact.llm_client import FileLLMResponseCache, LLMResponseError, LlamaCppClient
from news_impact.llm_config import LLMConfig, load_llm_config
from news_impact.mapper import MappingCandidate
from news_impact.market_clock import KST
from news_impact.ranking import rank_report_rows
from news_impact.report import ReportRow, write_csv_report, write_json_report
from news_impact.schema import DisclosureItem, ImpactEvent, NewsItem, RunAudit
from news_impact.scorer import aggregate_scores
from news_impact.semantic_clusterer import SemanticClusterLLM, assign_semantic_cluster_ids


SCORING_VERSION = "scoring.v1"
BACKTEST_VERSION = "backtest.v1"
RULE_BASED_FLAGS = {"rule_based_no_llm"}


class ImpactJudgeLLM(Protocol):
    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class DailyPipelineInputs:
    run_date: str
    watchlist_path: str | Path
    company_master_path: str | Path
    input_fixture_path: str | Path
    output_dir: str | Path
    semantic_clustering: bool = True
    llm_config_path: str | Path | None = None
    impact_judge_llm: ImpactJudgeLLM | None = None
    semantic_cluster_llm: SemanticClusterLLM | None = None


@dataclass(frozen=True)
class DailyPipelineResult:
    run_id: str
    artifact_paths: dict[str, Path]
    report_rows: tuple[ReportRow, ...]


def run_daily_pipeline(inputs: DailyPipelineInputs) -> DailyPipelineResult:
    watchlist_path = Path(inputs.watchlist_path)
    company_master_path = Path(inputs.company_master_path)
    fixture_path = Path(inputs.input_fixture_path)
    output_dir = Path(inputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture = _read_json_object(fixture_path)
    run_id = _run_id(inputs.run_date, fixture_path)
    companies = _read_company_master(company_master_path)
    watchlist_tickers = _read_watchlist_tickers(watchlist_path)
    news = dedupe_news_items([_news_item(row) for row in fixture.get("news", [])])
    disclosures = dedupe_disclosures(
        [_disclosure_item(row) for row in fixture.get("disclosures", [])]
    )
    clustered_news = _assign_unique_news_cluster_ids(news)
    clustered_disclosures = assign_cluster_ids(disclosures)
    llm_config = (
        load_llm_config(inputs.llm_config_path)
        if inputs.llm_config_path is not None
        else LLMConfig.default()
    )
    impact_judge_llm = inputs.impact_judge_llm or _build_impact_judge_llm(
        llm_config=llm_config,
        output_dir=output_dir,
    )
    impact_events, llm_failed_count = _build_llm_judged_events(
        run_date=inputs.run_date,
        clustered_news=clustered_news,
        watchlist_tickers=watchlist_tickers,
        companies=companies,
        llm_client=impact_judge_llm,
    )
    if inputs.semantic_clustering and impact_events:
        impact_events = assign_semantic_cluster_ids(
            impact_events,
            inputs.semantic_cluster_llm or _build_semantic_cluster_llm(inputs, output_dir),
        )
    semantic_cluster_metrics = _semantic_cluster_metrics(impact_events)
    fixture_risk_flags = _filter_rule_based_flags(
        tuple(str(flag) for flag in fixture.get("risk_flags", ()))
    )
    audit_risk_flags = _build_audit_risk_flags(
        fixture_risk_flags=fixture_risk_flags,
        impact_event_count=len(impact_events),
        semantic_cluster_failed_count=semantic_cluster_metrics["failed_count"],
        semantic_cluster_failure_rate=semantic_cluster_metrics["failure_rate"],
    )

    audit = RunAudit(
        run_id=run_id,
        run_started_at=datetime.now(tz=KST),
        git_commit=_git_commit(),
        config_hash=_file_sha256(fixture_path),
        watchlist_hash=_file_sha256(watchlist_path),
        company_master_snapshot_id=_file_sha256(company_master_path),
        data_snapshot_id=_file_sha256(fixture_path),
        llm_provider=llm_config.provider,
        llm_model_requested=llm_config.model,
        llm_model_returned=getattr(impact_judge_llm, "last_response_model", None)
        or llm_config.model,
        scoring_version=SCORING_VERSION,
        backtest_version=BACKTEST_VERSION,
    )
    rows = _build_report_rows(
        run_date=inputs.run_date,
        run_id=run_id,
        events=impact_events,
        companies=companies,
        llm_failed_count=llm_failed_count,
        fixture_risk_flags=fixture_risk_flags,
    )
    ranked_rows = tuple(rank_report_rows(rows))

    raw_snapshot = {
        "schema": "stock-news-impact.raw_snapshot.v1",
        "run_id": run_id,
        "input_fixture": str(fixture_path),
        "payload": fixture,
    }
    normalized_snapshot = {
        "schema": "stock-news-impact.normalized_snapshot.v1",
        "run_id": run_id,
        "news": [
            {"cluster_id": clustered.cluster_id, **clustered.item.to_dict()}
            for clustered in clustered_news
        ],
        "disclosures": [
            {"cluster_id": clustered.cluster_id, **clustered.item.to_dict()}
            for clustered in clustered_disclosures
        ],
    }
    impact_events_payload = {
        "schema": "stock-news-impact.impact_events.v1",
        "run_id": run_id,
        "impact_events": [event.to_dict() for event in impact_events],
    }
    audit_payload = _build_audit_payload(
        audit=audit,
        output_dir=output_dir,
        watchlist_ticker_count=len(watchlist_tickers),
        news_count=len(news),
        disclosure_count=len(disclosures),
        impact_event_count=len(impact_events),
        report_row_count=len(ranked_rows),
        llm_failed_count=llm_failed_count,
        semantic_cluster_metrics=semantic_cluster_metrics,
        risk_flags=audit_risk_flags,
    )

    artifact_paths = {
        "raw_snapshot.json": output_dir / "raw_snapshot.json",
        "normalized_snapshot.json": output_dir / "normalized_snapshot.json",
        "impact_events.json": output_dir / "impact_events.json",
        "audit.json": output_dir / "audit.json",
        "report.json": output_dir / "report.json",
        "report.csv": output_dir / "report.csv",
    }
    _write_json(artifact_paths["raw_snapshot.json"], raw_snapshot)
    _write_json(artifact_paths["normalized_snapshot.json"], normalized_snapshot)
    _write_json(artifact_paths["impact_events.json"], impact_events_payload)
    _write_json(artifact_paths["audit.json"], audit_payload)
    write_json_report(list(ranked_rows), artifact_paths["report.json"], audit=audit_payload)
    write_csv_report(list(ranked_rows), artifact_paths["report.csv"])
    return DailyPipelineResult(
        run_id=run_id,
        artifact_paths=artifact_paths,
        report_rows=ranked_rows,
    )


def _build_report_rows(
    run_date: str,
    run_id: str,
    events: list[ImpactEvent],
    companies: dict[str, dict[str, str]],
    llm_failed_count: int,
    fixture_risk_flags: tuple[str, ...],
) -> list[ReportRow]:
    rows: list[ReportRow] = []
    events_by_ticker: dict[str, list[ImpactEvent]] = {}
    for event in events:
        events_by_ticker.setdefault(event.ticker, []).append(event)

    for ticker, ticker_events in sorted(events_by_ticker.items()):
        company = companies.get(ticker, {})
        summary = aggregate_scores(ticker_events, llm_failed_count=llm_failed_count)
        top_event = max(
            ticker_events,
            key=lambda event: (event.impact_strength * event.confidence, event.event_id),
        )
        risk_flags = set(fixture_risk_flags)
        for event in ticker_events:
            risk_flags.update(event.risk_flags)
        if llm_failed_count:
            risk_flags.add("llm_judgment_failed")
        rows.append(
            ReportRow(
                date=run_date,
                run_id=run_id,
                ticker=ticker,
                company=company.get("company", top_event.company),
                market=company.get("market", ""),
                sector=company.get("sector", top_event.sector),
                news_disclosure_score=summary.news_disclosure_score,
                global_proxy_adjustment=0.0,
                sector_neutral_score=summary.news_disclosure_score,
                positive_score=summary.positive_score,
                negative_score=summary.negative_score,
                uncertainty_score=summary.uncertainty_score,
                confidence=max(event.confidence for event in ticker_events),
                event_count=summary.event_count,
                llm_failed_count=summary.llm_failed_count,
                top_event_type=top_event.event_type,
                top_reason=top_event.reason,
                why_may_be_wrong=top_event.why_may_be_wrong,
                risk_flags=tuple(sorted(risk_flags)),
                already_reflected_price_move=top_event.already_reflected_price_move,
                price_change_since_news=0.0,
                volume_change=0.0,
                tradeability_status="unknown",
                review_checklist=(
                    "not_investment_advice",
                    "verify_evidence",
                    "check_tradeability",
                ),
                top_evidence_url=top_event.evidence_urls[0] if top_event.evidence_urls else "",
            )
        )
    return rows


def _build_llm_judged_events(
    run_date: str,
    clustered_news: Iterable[Any],
    watchlist_tickers: list[str],
    companies: dict[str, dict[str, str]],
    llm_client: ImpactJudgeLLM,
) -> tuple[list[ImpactEvent], int]:
    events: list[ImpactEvent] = []
    llm_failed_count = 0
    system_prompt = build_system_prompt()
    for news_index, clustered in enumerate(clustered_news, start=1):
        item = clustered.item
        article_text, input_flags = _llm_article_text_and_flags(item)
        for ticker in watchlist_tickers:
            company = companies.get(ticker, {})
            company_name = company.get("company", "")
            sector = company.get("sector", "")
            analysis_input = NewsAnalysisInput(
                title=item.title,
                article_text=article_text,
                url=item.original_url or item.url,
                publisher_domain=item.publisher_domain,
                risk_flags=input_flags,
                should_call_llm=True,
            )
            user_prompt = _build_targeted_news_prompt(
                run_date=run_date,
                ticker=ticker,
                company=company_name,
                sector=sector,
                analysis_input=analysis_input,
            )
            try:
                judgment = llm_client.chat_json(
                    system_prompt,
                    user_prompt,
                    required_keys=LLM_REQUIRED_KEYS,
                )
            except (LLMResponseError, TimeoutError, OSError):
                llm_failed_count += 1
                continue
            judgment = {
                **judgment,
                "risk_flags": _dedupe_flags(
                    tuple(str(flag) for flag in judgment.get("risk_flags", ()))
                    + input_flags
                ),
            }
            events.append(
                judgment_to_impact_event(
                    judgment=judgment,
                    candidate=MappingCandidate(
                        ticker=ticker,
                        relation_type="watchlist_target",
                        relevance=1.0,
                        confidence=1.0,
                        evidence="watchlist",
                    ),
                    event_id=f"news-{news_index:03d}-{ticker}",
                    cluster_id=clustered.cluster_id,
                    company=company_name,
                    sector=sector,
                    evidence_urls=tuple(
                        url
                        for url in (item.original_url, item.url)
                        if isinstance(url, str) and url
                    ),
                )
            )
    return events, llm_failed_count


def _assign_unique_news_cluster_ids(news: Iterable[NewsItem]) -> list[ClusteredItem]:
    return [
        ClusteredItem(item=item, cluster_id=f"cluster-news-{index:03d}")
        for index, item in enumerate(news, start=1)
    ]


def _llm_article_text_and_flags(item: NewsItem) -> tuple[str, tuple[str, ...]]:
    base_flags = tuple(str(flag) for flag in item.quality_flags)
    if item.raw_text:
        text = item.raw_text
        flags = base_flags + detect_prompt_injection(text)
    else:
        text = item.summary
        flags = base_flags + ("summary_only_no_full_text", "needs_full_text_review")
    return text, _dedupe_flags(flags)


def _build_targeted_news_prompt(
    run_date: str,
    ticker: str,
    company: str,
    sector: str,
    analysis_input: NewsAnalysisInput,
) -> str:
    return "\n".join(
        (
            f"date: {run_date}",
            f"ticker: {ticker}",
            f"company: {company}",
            f"sector: {sector}",
            build_news_user_prompt(analysis_input),
        )
    )


def _build_impact_judge_llm(llm_config: LLMConfig, output_dir: Path) -> LlamaCppClient:
    cache = FileLLMResponseCache(output_dir / "llm_cache" / "impact_judgments")
    return LlamaCppClient(llm_config, cache=cache)


def _build_semantic_cluster_llm(
    inputs: DailyPipelineInputs,
    output_dir: Path,
) -> LlamaCppClient:
    config = (
        load_llm_config(inputs.llm_config_path)
        if inputs.llm_config_path is not None
        else LLMConfig.default()
    )
    cache = FileLLMResponseCache(output_dir / "llm_cache" / "semantic_clusters")
    return LlamaCppClient(config, cache=cache)


def _news_item(row: object) -> NewsItem:
    if not isinstance(row, dict):
        raise ValueError("news fixture rows must be objects")
    data = dict(row)
    for key in ("published_at", "collected_at", "signal_at"):
        data[key] = _parse_datetime(data[key])
    return NewsItem(**data)


def _disclosure_item(row: object) -> DisclosureItem:
    if not isinstance(row, dict):
        raise ValueError("disclosure fixture rows must be objects")
    data = dict(row)
    for key in ("disclosure_at", "collected_at", "signal_at"):
        data[key] = _parse_datetime(data[key])
    return DisclosureItem(**data)


def _impact_event(row: object) -> ImpactEvent:
    if not isinstance(row, dict):
        raise ValueError("impact event fixture rows must be objects")
    return ImpactEvent(**dict(row))


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("datetime fixture values must be strings")
    return datetime.fromisoformat(value)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input fixture must be a JSON object")
    return payload


def _read_company_master(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return {row["ticker"]: row for row in csv.DictReader(file)}


def _read_watchlist_tickers(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row["ticker"] for row in csv.DictReader(file)]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failure_causes(
    llm_failed_count: int,
    impact_event_count: int,
    semantic_cluster_failed_count: int,
) -> list[dict[str, Any]]:
    causes: list[dict[str, Any]] = []
    if llm_failed_count:
        causes.append(
            {
                "component": "llm",
                "cause": "llm_judgment_failed",
                "count": llm_failed_count,
                "audit_field": "counts.llm_failed_count",
            }
        )
    if impact_event_count == 0:
        causes.append(
            {
                "component": "impact_judge",
                "cause": "missing_impact_events",
                "count": 0,
                "audit_field": "risk_flags",
            }
        )
    if semantic_cluster_failed_count:
        causes.append(
            {
                "component": "semantic_clusterer",
                "cause": "semantic_cluster_failed",
                "count": semantic_cluster_failed_count,
                "audit_field": "counts.semantic_cluster_failed_count",
            }
        )
    return causes


def _build_audit_payload(
    audit: RunAudit,
    output_dir: Path,
    watchlist_ticker_count: int,
    news_count: int,
    disclosure_count: int,
    impact_event_count: int,
    report_row_count: int,
    llm_failed_count: int,
    semantic_cluster_metrics: dict[str, Any],
    risk_flags: list[str],
) -> dict[str, Any]:
    audit_base = audit.to_dict()
    return {
        **audit_base,
        "schema": "stock-news-impact.audit.v1",
        "counts": {
            "watchlist_tickers": watchlist_ticker_count,
            "news": news_count,
            "disclosures": disclosure_count,
            "impact_events": impact_event_count,
            "report_rows": report_row_count,
            "llm_failed_count": llm_failed_count,
            "semantic_cluster_failed_count": semantic_cluster_metrics["failed_count"],
        },
        "risk_flags": sorted(set(risk_flags)),
        "failure_causes": _failure_causes(
            llm_failed_count=llm_failed_count,
            impact_event_count=impact_event_count,
            semantic_cluster_failed_count=semantic_cluster_metrics["failed_count"],
        ),
        "semantic_cluster_failure_causes": semantic_cluster_metrics["failure_causes"],
        "semantic_cluster_failure_rate": semantic_cluster_metrics["failure_rate"],
        "replay": {
            "run_id": audit.run_id,
            "config_hash": audit_base["config_hash"],
            "watchlist_hash": audit_base["watchlist_hash"],
            "company_master_snapshot_id": audit_base["company_master_snapshot_id"],
            "data_snapshot_id": audit_base["data_snapshot_id"],
            "reproducible_report_scope": "rows_excluding_audit_timestamps_and_output_paths",
        },
        "artifacts": {
            name: str(output_dir / name)
            for name in _artifact_names(include_reports=False)
        },
    }


def _build_audit_risk_flags(
    fixture_risk_flags: tuple[str, ...],
    impact_event_count: int,
    semantic_cluster_failed_count: int,
    semantic_cluster_failure_rate: float,
) -> list[str]:
    risk_flags = list(fixture_risk_flags)
    if impact_event_count == 0:
        risk_flags.append("missing_impact_events")
    if semantic_cluster_failed_count:
        risk_flags.append("semantic_cluster_failed")
    if semantic_cluster_failed_count and semantic_cluster_failure_rate >= 0.2:
        risk_flags.append("semantic_clustering_degraded")
    return risk_flags


def _run_id(run_date: str, fixture_path: Path) -> str:
    return f"daily-{run_date}-{_file_sha256(fixture_path)[:8]}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dedupe_flags(flags: tuple[str, ...]) -> tuple[str, ...]:
    deduped: list[str] = []
    for flag in flags:
        if flag not in deduped:
            deduped.append(flag)
    return tuple(deduped)


def _filter_rule_based_flags(flags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(flag for flag in flags if flag not in RULE_BASED_FLAGS)


def _semantic_cluster_metrics(events: list[ImpactEvent]) -> dict[str, Any]:
    failed_count = _semantic_cluster_failed_count(events)
    failure_rate = round(failed_count / len(events), 6) if events else 0.0
    return {
        "failed_count": failed_count,
        "failure_causes": _semantic_cluster_failure_causes(events),
        "failure_rate": failure_rate,
    }


def _semantic_cluster_failed_count(events: Iterable[ImpactEvent]) -> int:
    return sum(1 for event in events if "semantic_cluster_failed" in event.risk_flags)


def _semantic_cluster_failure_causes(events: Iterable[ImpactEvent]) -> list[dict[str, Any]]:
    prefix = "semantic_cluster_failed:"
    counts: dict[str, int] = {}
    for event in events:
        for flag in event.risk_flags:
            if flag.startswith(prefix):
                cause = flag[len(prefix) :]
                counts[cause] = counts.get(cause, 0) + 1
    return [{"cause": cause, "count": counts[cause]} for cause in sorted(counts)]


def _artifact_names(include_reports: bool = True) -> tuple[str, ...]:
    names = (
        "raw_snapshot.json",
        "normalized_snapshot.json",
        "impact_events.json",
        "audit.json",
    )
    if include_reports:
        return names + ("report.json", "report.csv")
    return names
