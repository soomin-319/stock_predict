from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.news_impact.market_clock import KST


@dataclass(frozen=True)
class NewsImpactFixtureBundle:
    fixture_path: Path
    watchlist_path: Path
    company_master_path: Path


def build_news_impact_fixture(
    *,
    context_raw_df: pd.DataFrame,
    symbols: Iterable[str],
    symbol_name_map: dict[str, str],
    run_date: str,
    output_dir: str | Path,
) -> NewsImpactFixtureBundle:
    """Convert chatbot ``context_raw_df`` into the fixture/watchlist/company-master
    files that ``src.news_impact.pipeline.run_daily_pipeline`` consumes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = {str(s) for s in symbols if str(s).strip()}

    df = context_raw_df.copy()
    df["Symbol"] = df["Symbol"].astype(str)
    if target:
        df = df[df["Symbol"].isin(target)]
    df["source_type"] = df["source_type"].astype(str)

    news_rows: list[dict] = []
    disclosure_rows: list[dict] = []
    for _, row in df.iterrows():
        published_at = _resolve_datetime(row.get("published_at"), run_date)
        signal_at = published_at  # collected_at == published_at -> signal_at == max(...)
        source_type = str(row.get("source_type"))
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        if source_type == "news":
            news_rows.append(
                {
                    "source": str(row.get("provider") or "naver"),
                    "title": title,
                    "summary": "",
                    "url": str(row.get("url") or ""),
                    "original_url": str(row.get("url") or "") or None,
                    "publisher_domain": None,
                    "publisher_domain_source": None,
                    "publisher_confidence": 0.0,
                    "published_at": published_at.isoformat(),
                    "timestamp_source": "naver_pubDate" if _has_value(row.get("published_at")) else "manual",
                    "collected_at": published_at.isoformat(),
                    "signal_at": signal_at.isoformat(),
                    "market_session": "regular",
                    "raw_text": None,
                    "storage_policy": "metadata_only",
                    "quality_flags": ["title_only"],
                }
            )
        elif source_type == "disclosure":
            ticker = _ticker(str(row.get("Symbol")))
            if ticker is None:
                continue
            disclosure_rows.append(
                {
                    "source": str(row.get("provider") or "dart"),
                    "receipt_no": str(row.get("raw_id") or f"synthetic-{ticker}-{len(disclosure_rows)}"),
                    "corp_code": "",
                    "ticker": ticker,
                    "disclosure_title": title,
                    "disclosure_at": published_at.isoformat(),
                    "collected_at": published_at.isoformat(),
                    "signal_at": signal_at.isoformat(),
                    "is_correction": "정정" in title,
                    "original_receipt_no": None,
                    "url": str(row.get("url") or ""),
                    "quality_flags": [],
                }
            )

    fixture_path = out / "fixture.json"
    fixture_path.write_text(
        json.dumps({"news": news_rows, "disclosures": disclosure_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    watchlist_path = out / "watchlist.csv"
    company_master_path = out / "company_master.csv"
    tickers = [t for t in (_ticker(str(s)) for s in (target or df["Symbol"].unique())) if t]
    unique_tickers = list(dict.fromkeys(tickers))
    _write_csv(watchlist_path, ["ticker"], [{"ticker": t} for t in unique_tickers])
    _write_csv(
        company_master_path,
        ["ticker", "company", "market", "sector"],
        [
            {
                "ticker": t,
                "company": symbol_name_map.get(_symbol_for_ticker(t, target), t),
                "market": _market(_symbol_for_ticker(t, target)),
                "sector": "",
            }
            for t in unique_tickers
        ],
    )
    return NewsImpactFixtureBundle(fixture_path, watchlist_path, company_master_path)


def _has_value(value: object) -> bool:
    return bool(str(value or "").strip())


def _resolve_datetime(value: object, run_date: str) -> datetime:
    text = str(value or "").strip()
    if text:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=KST)
    base = datetime.fromisoformat(f"{run_date}T00:00:00")
    return datetime.combine(base.date(), time(9, 0), tzinfo=KST)


def _ticker(symbol: str) -> str | None:
    head = symbol.split(".", 1)[0]
    return head if len(head) == 6 and head.isdigit() else None


def _symbol_for_ticker(ticker: str, target: set[str]) -> str:
    for symbol in target:
        if symbol.split(".", 1)[0] == ticker:
            return symbol
    return ticker


def _market(symbol: str) -> str:
    return "KOSDAQ" if symbol.upper().endswith(".KQ") else "KOSPI"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
