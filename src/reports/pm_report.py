from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.utils.atomic_files import atomic_write_text

COMMON_METADATA_FIELDS = (
    "schema_version",
    "run_id",
    "environment",
    "data_mode",
    "generated_at",
    "input_as_of_date",
    "prediction_for_date",
    "context_as_of_date",
    "git_commit",
    "config_hash",
    "status",
    "blocking_reasons",
)


def build_pm_report(pred_df: pd.DataFrame, report: dict) -> dict:
    work = pred_df.copy()
    if "recommendation" in work.columns:
        top_buys = work[work["recommendation"].astype(str) == "매수"].copy()
    else:
        top_buys = work.head(0).copy()
    top_buys = top_buys.sort_values(["predicted_return", "confidence_score"], ascending=[False, False]).head(10)
    focus_columns = [
        "Symbol",
        "symbol_name",
        "recommendation",
        "portfolio_action",
        "trading_gate",
        "risk_flag",
        "predicted_return",
        "up_probability",
        "prediction_reason",
    ]
    focus_columns = [column for column in focus_columns if column in top_buys.columns]
    top_buy_payload = top_buys[focus_columns].to_dict(orient="records")

    up_probability = (
        pd.to_numeric(work["up_probability"], errors="coerce").fillna(0.5)
        if "up_probability" in work.columns
        else pd.Series(0.5, index=work.index)
    )
    horizon_summary = {
        "1d": {
            "avg_predicted_return_pct": float(pd.to_numeric(work["predicted_return"], errors="coerce").fillna(0.0).mean()),
            "positive_signal_count": int((pd.to_numeric(work["predicted_return"], errors="coerce").fillna(0.0) > 0).sum()),
            "avg_up_probability": float(up_probability.mean()),
        }
    }

    coverage_gate = report.get("coverage_gate", {})
    risk_counts = work["risk_flag"].astype(str).value_counts(dropna=False).to_dict() if "risk_flag" in work.columns else {}

    return {
        **{key: report.get(key) for key in COMMON_METADATA_FIELDS},
        "coverage_gate": coverage_gate,
        "pm_summary": report.get("pm_summary", {}),
        "risk_flag_counts": risk_counts,
        "horizon_summary": horizon_summary,
        "top_buy_candidates": top_buy_payload,
    }


def save_pm_report(pm_report: dict, out_path: Path) -> Path:
    atomic_write_text(out_path, json.dumps(pm_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


__all__ = ["build_pm_report", "save_pm_report"]
