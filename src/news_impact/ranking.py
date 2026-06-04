from __future__ import annotations

from src.news_impact.report import ReportRow


def compute_sector_neutral_score(
    news_disclosure_score: float,
    market_effect: float,
    beta: float = 1.0,
) -> float:
    return _round_score(_clamp(news_disclosure_score - beta * market_effect, -100.0, 100.0))


def rank_report_rows(
    rows: list[ReportRow] | tuple[ReportRow, ...],
    include_untradable: bool = True,
) -> list[ReportRow]:
    rankable = list(rows)
    if not include_untradable:
        rankable = [
            row
            for row in rankable
            if row.tradeability_status not in {"halted", "limit_locked"}
        ]
    return sorted(
        rankable,
        key=lambda row: (
            row.news_disclosure_score,
            row.confidence,
            -row.uncertainty_score,
            row.ticker,
        ),
        reverse=True,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _round_score(value: float) -> float:
    return round(value, 6)
