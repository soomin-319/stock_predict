from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NEWS_IMPACT_COLUMNS = {
    "run_id": "news_impact_run_id",
    "final_score": "news_impact_final_score",
    "sector_neutral_score": "news_impact_sector_neutral_score",
    "uncertainty_score": "news_impact_uncertainty_score",
    "top_event_type": "news_impact_top_event_type",
    "top_reason": "news_impact_top_reason",
    "why_may_be_wrong": "news_impact_why_may_be_wrong",
    "risk_flags": "news_impact_risk_flags",
    "tradeability_status": "news_impact_tradeability_status",
    "review_checklist": "news_impact_review_checklist",
    "top_evidence_url": "news_impact_top_evidence_url",
}


def append_news_impact_context(pred_df: pd.DataFrame, report_path: str | Path | None) -> pd.DataFrame:
    if pred_df.empty or not report_path:
        return pred_df
    path = Path(report_path)
    if not path.exists():
        return pred_df
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pred_df
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return pred_df
    context = pd.DataFrame(rows)
    if not {"date", "ticker"}.issubset(set(context.columns)):
        return pred_df
    out = pred_df.copy()
    join = context.copy()
    join["Date"] = pd.to_datetime(join["date"], errors="coerce").dt.normalize()
    join["Symbol"] = join["ticker"].astype(str).map(_ticker_to_symbol)
    keep = ["Date", "Symbol"]
    for source, target in NEWS_IMPACT_COLUMNS.items():
        if source in join.columns:
            join[target] = join[source]
            keep.append(target)
    join = join[keep].dropna(subset=["Date", "Symbol"]).drop_duplicates(subset=["Date", "Symbol"], keep="last")
    if join.empty:
        return out
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    return out.merge(join, on=["Date", "Symbol"], how="left")


def _ticker_to_symbol(ticker: str) -> str:
    value = str(ticker).strip()
    if "." in value:
        return value
    if len(value) == 6 and value.isdigit():
        return f"{value}.KS"
    return value
