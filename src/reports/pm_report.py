from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_pm_report(pred_df: pd.DataFrame, report: dict) -> dict:
    work = pred_df.copy()
    if "recommendation" in work.columns:
        top_buys = work[work["recommendation"].astype(str) == "매수"].copy()
    else:
        top_buys = work.head(0).copy()
    top_buys = top_buys.sort_values(["signal_score", "confidence_score"], ascending=[False, False]).head(10)
    focus_columns = [
        "Symbol",
        "symbol_name",
        "recommendation",
        "portfolio_action",
        "trading_gate",
        "risk_flag",
        "predicted_return",
        "predicted_return_5d",
        "predicted_return_20d",
        "up_probability",
        "up_probability_5d",
        "up_probability_20d",
        "prediction_reason",
    ]
    focus_columns = [column for column in focus_columns if column in top_buys.columns]
    top_buy_payload = top_buys[focus_columns].to_dict(orient="records")

    horizon_summary = {}
    for horizon in (1, 5, 20):
        return_col = "predicted_return" if horizon == 1 else f"predicted_return_{horizon}d"
        prob_col = "up_probability" if horizon == 1 else f"up_probability_{horizon}d"
        if return_col in work.columns:
            prob_source = work[prob_col] if prob_col in work.columns else pd.Series(0.5, index=work.index)
            horizon_summary[f"{horizon}d"] = {
                "avg_predicted_return_pct": float(pd.to_numeric(work[return_col], errors="coerce").fillna(0.0).mean()),
                "positive_signal_count": int((pd.to_numeric(work[return_col], errors="coerce").fillna(0.0) > 0).sum()),
                "avg_up_probability": float(pd.to_numeric(prob_source, errors="coerce").fillna(0.5).mean()),
            }

    coverage_gate = report.get("coverage_gate", {})
    risk_counts = work["risk_flag"].astype(str).value_counts(dropna=False).to_dict() if "risk_flag" in work.columns else {}

    return {
        "coverage_gate": coverage_gate,
        "pm_summary": report.get("pm_summary", {}),
        "risk_flag_counts": risk_counts,
        "horizon_summary": horizon_summary,
        "top_buy_candidates": top_buy_payload,
    }


def save_pm_report(pm_report: dict, out_path: Path) -> Path:
    out_path.write_text(json.dumps(pm_report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


__all__ = ["build_pm_report", "save_pm_report"]
