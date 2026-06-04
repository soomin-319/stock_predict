from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPORT_COLUMNS = (
    "date",
    "run_id",
    "ticker",
    "company",
    "market",
    "sector",
    "news_disclosure_score",
    "global_proxy_adjustment",
    "sector_neutral_score",
    "positive_score",
    "negative_score",
    "uncertainty_score",
    "confidence",
    "event_count",
    "llm_failed_count",
    "top_event_type",
    "top_reason",
    "why_may_be_wrong",
    "risk_flags",
    "already_reflected_price_move",
    "price_change_since_news",
    "volume_change",
    "tradeability_status",
    "review_checklist",
    "top_evidence_url",
)

REPORT_DISCLAIMER = "This report is for research review only and is not investment advice."


@dataclass(frozen=True)
class ReportRow:
    date: str
    run_id: str
    ticker: str
    company: str
    market: str
    sector: str
    news_disclosure_score: float
    global_proxy_adjustment: float
    sector_neutral_score: float
    positive_score: float
    negative_score: float
    uncertainty_score: float
    confidence: float
    event_count: int
    llm_failed_count: int
    top_event_type: str
    top_reason: str
    why_may_be_wrong: str
    risk_flags: tuple[str, ...] | list[str]
    already_reflected_price_move: float
    price_change_since_news: float
    volume_change: float
    tradeability_status: str
    review_checklist: tuple[str, ...] | list[str]
    top_evidence_url: str

    def __post_init__(self) -> None:
        _require_ticker(self.ticker)
        for field_name in (
            "news_disclosure_score",
            "global_proxy_adjustment",
            "sector_neutral_score",
            "positive_score",
            "negative_score",
        ):
            _require_score_range(field_name, float(getattr(self, field_name)))
        _require_probability("confidence", self.confidence)
        _require_score_range("uncertainty_score", self.uncertainty_score, minimum=0.0)
        if self.event_count < 0:
            raise ValueError("event_count must be non-negative")
        if self.llm_failed_count < 0:
            raise ValueError("llm_failed_count must be non-negative")
        object.__setattr__(self, "risk_flags", tuple(self.risk_flags))
        object.__setattr__(self, "review_checklist", tuple(self.review_checklist))

    def to_raw_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        raw = self.to_raw_dict()
        raw["risk_flags"] = _join_list(self.risk_flags)
        raw["review_checklist"] = _join_list(self.review_checklist)
        return {column: raw[column] for column in REPORT_COLUMNS}


def write_csv_report(rows: list[ReportRow] | tuple[ReportRow, ...], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def write_json_report(
    rows: list[ReportRow] | tuple[ReportRow, ...],
    output_path: str | Path,
    audit: Any | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stock-news-impact.report.v1",
        "disclaimer": REPORT_DISCLAIMER,
        "columns": list(REPORT_COLUMNS),
        "rows": [row.to_dict() for row in rows],
    }
    if audit is not None:
        payload["audit"] = audit.to_dict() if hasattr(audit, "to_dict") else dict(audit)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _join_list(values: tuple[str, ...] | list[str]) -> str:
    return ";".join(values)


def _require_ticker(value: str) -> None:
    if len(value) != 6 or not value.isdigit():
        raise ValueError("ticker must be a six-character numeric string")


def _require_score_range(field_name: str, value: float, minimum: float = -100.0) -> None:
    if not minimum <= value <= 100.0:
        raise ValueError(f"{field_name} must be between {minimum} and 100.0")


def _require_probability(field_name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
