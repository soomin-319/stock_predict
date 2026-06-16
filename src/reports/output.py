from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.domain.signal_policy import build_prediction_policy_frame
from src.reports.result_formatter import (
    build_result_simple as format_result_simple,
    print_prediction_console_summary as format_prediction_console_summary,
)
from src.utils.atomic_files import _tmp_path


def project_result_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_output_path(output_csv: str, is_windows: bool | None = None) -> Path:
    """Force all file outputs under project-local ./result directory."""
    _ = (os.name == "nt") if is_windows is None else is_windows
    requested = Path(output_csv)
    result_dir = project_result_dir()

    try:
        if requested.is_absolute() and requested.resolve().is_relative_to(result_dir.resolve()):
            output_path = requested
        else:
            output_path = result_dir / requested.name
    except Exception:
        output_path = result_dir / requested.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_output_dir(output_dir: str) -> Path:
    requested = Path(output_dir)
    result_dir = project_result_dir()
    try:
        if requested.is_absolute() and requested.resolve().is_relative_to(result_dir.resolve()):
            out_dir = requested
        else:
            out_dir = result_dir / requested.name
    except Exception:
        out_dir = result_dir / requested.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_to_csv(df: pd.DataFrame, path: Path, *, allow_fallback: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(path)
        return path
    except PermissionError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        if not allow_fallback:
            raise
        fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
        fallback_tmp = _tmp_path(fallback)
        try:
            df.to_csv(fallback_tmp, index=False, encoding="utf-8-sig")
            fallback_tmp.replace(fallback)
        finally:
            try:
                if fallback_tmp.exists():
                    fallback_tmp.unlink()
            except OSError:
                pass
        print(f"[경고] 파일이 열려 있어 기본 경로에 저장하지 못했습니다. 대체 경로로 저장: {fallback}")
        return fallback
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def build_pipeline_result_simple(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if "confidence_score" not in out.columns:
        if "예측 신뢰도" in out.columns:
            out["confidence_score"] = pd.to_numeric(out["예측 신뢰도"], errors="coerce").fillna(0.5)
        else:
            out["confidence_score"] = 0.5
    if "history_direction_accuracy" not in out.columns:
        out["history_direction_accuracy"] = 0.5
    required = {"recommendation", "portfolio_action", "trading_gate", "risk_flag", "confidence_label"}
    if not required.issubset(set(out.columns)):
        out = build_prediction_policy_frame(out)
    return format_result_simple(out)


def print_pipeline_prediction_console_summary(pred_df: pd.DataFrame) -> None:
    out = pred_df.copy()
    required = {"recommendation", "portfolio_action", "trading_gate", "risk_flag", "confidence_label"}
    if not required.issubset(set(out.columns)):
        out = build_prediction_policy_frame(out)
    if "confidence_score" not in out.columns:
        out["confidence_score"] = 0.5
    if "history_direction_accuracy" not in out.columns:
        out["history_direction_accuracy"] = 0.5
    format_prediction_console_summary(out)


def drop_empty_detail_columns(
    detail_df: pd.DataFrame,
    *,
    prune_empty_optional: bool = False,
) -> pd.DataFrame:
    """Keep detail schema stable; optionally prune empty optional columns for legacy callers."""
    if not prune_empty_optional:
        return detail_df
    optional_cols = [
        "foreign_net_buy",
        "institution_net_buy",
        "disclosure_score",
        "news_sentiment",
        "news_relevance_score",
        "news_impact_score",
        "news_article_count",
        "rsi_pullback_buy_flag",
        "rsi_overbought_sell_flag",
        "foreign_buy_signal",
        "institution_buy_signal",
        "smart_money_buy_signal",
        "foreign_buy_ratio",
        "institution_buy_ratio",
        "smart_money_strength",
        "foreign_net_buy_z20",
        "institution_net_buy_z20",
        "foreign_net_buy_3d",
        "foreign_net_buy_5d",
        "institution_net_buy_3d",
        "institution_net_buy_5d",
        "news_positive_signal",
        "news_negative_signal",
        "near_52w_high_flag",
        "breakout_52w_flag",
        "leader_confirmation_flag",
        "investor_event_score",
        "target_up",
    ]
    drop_cols: list[str] = []
    for col in optional_cols:
        if col not in detail_df.columns:
            continue
        series = detail_df[col]
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().sum() == 0:
                drop_cols.append(col)
            continue
        normalized = series.astype(str).str.strip()
        non_empty = normalized[~normalized.isin({"", "-", "nan", "NaN", "None", "<NA>", "NA", "null"})]
        if non_empty.empty:
            drop_cols.append(col)
    if not drop_cols:
        return detail_df
    return detail_df.drop(columns=drop_cols, errors="ignore")


def build_issue_summary_snapshot(pred_df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "Symbol",
        "symbol_name",
        "오늘 종목 이슈 한줄 요약",
        "공시 요약",
        "뉴스 요약",
        "종합 판단",
        "주의사항",
    ]
    available = [c for c in summary_cols if c in pred_df.columns]
    if "Symbol" not in available:
        return pd.DataFrame(columns=summary_cols)
    snapshot = pred_df[available].copy()
    if "symbol_name" not in snapshot.columns:
        snapshot["symbol_name"] = snapshot["Symbol"].astype(str)
    for col in summary_cols:
        if col not in snapshot.columns:
            snapshot[col] = "-"
    return snapshot[summary_cols]


def build_combined_symbol_results(pred_df: pd.DataFrame, summary_csv: str | None, out_path: Path) -> str | None:
    if pred_df.empty or not summary_csv:
        return None
    try:
        summary = pd.read_csv(summary_csv, encoding="utf-8-sig")
    except Exception:
        return None

    if summary.empty or "Symbol" not in summary.columns:
        return None

    extra_cols = [c for c in summary.columns if c not in pred_df.columns]
    combined = pred_df.merge(summary[["Symbol", *extra_cols]], on="Symbol", how="left")
    saved = safe_to_csv(combined, out_path)
    return str(saved)
