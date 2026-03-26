from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config.settings import app_config_to_dict, load_app_config
from src.data.cleaners import clean_ohlcv
from src.data.fetch_real_data import append_real_ohlcv_csv, normalize_user_symbols, save_real_ohlcv_csv
from src.data.krx_universe import get_symbol_name_map
from src.data.loaders import load_ohlcv_csv
from src.data.investor_context import InvestorContextConfig, add_investor_context_with_coverage, collect_context_raw_events
from src.data.universe import filter_by_universe, load_default_universe_symbols, load_universe_symbols
from src.domain.signal_policy import (
    build_prediction_policy_frame,
    recommendation_from_signal as domain_recommendation_from_signal,
    vectorized_event_signal_boost,
)
from src.features.external_features import add_external_market_features_with_coverage
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.models.lgbm_heads import MultiHeadPrediction, MultiHeadStockModel
from src.pipeline_support import (
    PredictionFrameContext,
    build_scored_prediction_frame,
    build_symbol_history_accuracy,
    finalize_latest_prediction_frame,
)
from src.reports.pm_report import build_pm_report, save_pm_report
from src.reports.issue_summary import append_issue_summary_columns
from src.reports.result_formatter import (
    build_result_simple as formatter_build_result_simple,
    format_percentage_text as formatter_format_percentage_text,
    print_prediction_console_summary as formatter_print_prediction_console_summary,
)
from src.reports.visualize import (
    save_actual_vs_predicted_plot,
    save_actual_vs_predicted_price_plot,
    save_diagnostic_figures,
    save_symbol_level_comparison_figures,
    save_backtest_figures,
    save_signal_histogram,
    save_symbol_summary_artifacts,
)
from src.validation.backtest import run_long_only_topk_backtest
from src.validation.baselines import evaluate_baselines
from src.validation.signal_tuning import tune_signal_weights
from src.validation.walk_forward import walk_forward_validate_with_oof
from src.validation.metrics import probability_calibration_metrics


def _fallback_symbols_from_input_or_default(input_csv: str) -> list[str]:
    """Return the repo-managed default fetch universe used when no explicit fetch universe is provided."""
    _ = input_csv
    return load_default_universe_symbols()


def _project_result_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    out = root / "result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_output_path(output_csv: str, is_windows: bool | None = None) -> Path:
    """Force all file outputs under project-local ./result directory."""
    _ = (os.name == "nt") if is_windows is None else is_windows
    requested = Path(output_csv)
    result_dir = _project_result_dir()

    # keep explicit paths already inside result dir; otherwise redirect by filename
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
    result_dir = _project_result_dir()
    try:
        if requested.is_absolute() and requested.resolve().is_relative_to(result_dir.resolve()):
            out_dir = requested
        else:
            out_dir = result_dir / requested.name
    except Exception:
        out_dir = result_dir / requested.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _feature_columns(df: pd.DataFrame) -> list[str]:
    base = {
        "daily_return",
        "gap_return",
        "intraday_return",
        "range_pct",
        "vol_ratio_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "stoch_k",
        "stoch_d",
        "cci_20",
        "obv",
        "obv_change_5d",
        "value_traded",
        "turnover_rank_daily",
        "is_top_turnover_3",
        "is_top_turnover_10",
        "market_type_kospi",
        "market_type_kosdaq",
        "market_type_konex",
        "venue_krx",
        "venue_nxt",
        "session_regular",
        "session_premarket",
        "session_aftermarket",
        "session_offhours",
        "days_since_listing",
        "is_newly_listed",
        "is_newly_listed_60d",
        "individual_net_buy",
        "foreign_net_buy",
        "institution_net_buy",
        "foreign_ownership_ratio",
        "program_trading_flow",
        "disclosure_score",
        "news_sentiment",
        "news_relevance_score",
        "news_impact_score",
        "news_article_count",
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
        "close_to_52w_high",
        "near_52w_high_flag",
        "breakout_52w_flag",
        "leader_confirmation_flag",
        "rsi_pullback_buy_flag",
        "rsi_overbought_sell_flag",
        "investor_event_score",
        "limit_hit_up_flag",
        "limit_hit_down_flag",
        "limit_event_flag",
        "pbr",
        "per",
        "roe",
        "dividend_yield",
        "buyback_flag",
        "share_cancellation_flag",
        "shareholder_return_score",
        "short_sell_event_score",
    }
    return [
        c
        for c in df.columns
        if c.startswith(("ret_", "ma_", "close_to_ma_", "vol_", "ks", "kq", "gspc", "ixic", "nq_f", "sox", "vix", "krw", "tnx"))
        or c in base
    ]




def _adaptive_training_cfg(cfg, feat: pd.DataFrame):
    tuned = cfg.training
    uniq = len(feat["Date"].unique())
    tuned.min_train_size = min(tuned.min_train_size, max(60, int(uniq * 0.6)))
    tuned.test_size = min(tuned.test_size, max(20, int(uniq * 0.2)))
    tuned.step_size = min(tuned.step_size, max(20, tuned.test_size // 2))
    return tuned


def _format_percentage_text(value, digits: int = 1, unit_interval: bool = False) -> str:
    return formatter_format_percentage_text(value, digits=digits, unit_interval=unit_interval)


def _build_result_simple(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    if "confidence_score" not in out.columns:
        if "예측 신뢰도" in out.columns:
            out["confidence_score"] = pd.to_numeric(out["예측 신뢰도"], errors="coerce").fillna(0.5)
        else:
            out["confidence_score"] = 0.5
    if "history_direction_accuracy" not in out.columns:
        out["history_direction_accuracy"] = 0.5
    if not {"recommendation", "portfolio_action", "trading_gate", "risk_flag", "prediction_reason", "confidence_label"}.issubset(set(out.columns)):
        out = build_prediction_policy_frame(out)
    return formatter_build_result_simple(out)


def _backtest_summary_fields(backtest: dict) -> dict[str, float]:
    keys = [
        "days",
        "cum_return",
        "avg_daily_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "avg_selected_count",
        "benchmark_cum_return",
        "excess_cum_return",
        "halted_days",
        "liquidity_blocked_days",
        "avg_market_type_count",
    ]
    out = {}
    for k in keys:
        v = backtest.get(k, 0.0)
        out[f"backtest_{k}"] = float(v) if isinstance(v, (int, float)) else 0.0
    return out


def _print_backtest_console_summary(backtest: dict):
    print("\n=== Backtest Summary ===")
    print(f"Days: {int(backtest.get('days', 0))}")
    print(f"CumReturn: {float(backtest.get('cum_return', 0.0)):.3f}")
    print(f"AvgDailyReturn: {float(backtest.get('avg_daily_return', 0.0)):.4f}")
    print(f"Sharpe: {float(backtest.get('sharpe', 0.0)):.3f}")
    print(f"MaxDrawdown: {float(backtest.get('max_drawdown', 0.0)):.3f}")
    print(f"AvgTurnover: {float(backtest.get('avg_turnover', 0.0)):.3f}")
    print(f"AvgSelectedCount: {float(backtest.get('avg_selected_count', 0.0)):.2f}")


def _print_prediction_console_summary(pred_df: pd.DataFrame):
    out = pred_df.copy()
    required = {"recommendation", "portfolio_action", "trading_gate", "risk_flag", "prediction_reason", "confidence_label"}
    if not required.issubset(set(out.columns)):
        out = build_prediction_policy_frame(out)
    if "confidence_score" not in out.columns:
        out["confidence_score"] = 0.5
    if "history_direction_accuracy" not in out.columns:
        out["history_direction_accuracy"] = 0.5
    formatter_print_prediction_console_summary(out)


def _recommendation_from_signal(
    signal_score: float | int | None,
    predicted_return: float | int | None,
    up_probability: float | int | None = None,
    uncertainty_score: float | int | None = None,
) -> str:
    """Backward-compatible wrapper for tests/importers."""
    return domain_recommendation_from_signal(signal_score, predicted_return, up_probability, uncertainty_score)


def _policy_recommendation(row: pd.Series) -> str:
    """Backward-compatible policy helper kept for test compatibility."""
    return _recommendation_from_signal(
        row.get("signal_score"),
        row.get("predicted_return"),
        row.get("up_probability"),
        row.get("uncertainty_score"),
    ) if not (
        pd.to_numeric(pd.Series([row.get("nq_f_ret_1d")]), errors="coerce").iloc[0] <= -0.01
        or pd.to_numeric(pd.Series([row.get("rsi_14")]), errors="coerce").iloc[0] >= 70.0
    ) else "매도"


def _apply_event_signal_boost(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for event signal boost."""
    return vectorized_event_signal_boost(pred_df)


def _coverage_gate_status(cfg, external_coverage_ratio: float, investor_coverage_ratio: float) -> str:
    if cfg.backtest.min_external_coverage_ratio > 0 and external_coverage_ratio < cfg.backtest.min_external_coverage_ratio:
        return "halt"
    if cfg.backtest.min_investor_coverage_ratio > 0 and investor_coverage_ratio < cfg.backtest.min_investor_coverage_ratio:
        return "halt"
    if external_coverage_ratio < max(0.7, cfg.backtest.min_external_coverage_ratio) or investor_coverage_ratio < max(
        0.7, cfg.backtest.min_investor_coverage_ratio
    ):
        return "caution"
    return "normal"

def _split_oof_for_tuning_and_eval(scored_oof: pd.DataFrame, tune_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(scored_oof["Date"]).dropna().unique())
    if len(dates) < 10:
        return scored_oof.copy(), scored_oof.copy()

    split_idx = max(1, min(len(dates) - 1, int(len(dates) * tune_ratio)))
    tune_dates = set(dates[:split_idx])
    eval_dates = set(dates[split_idx:])

    tune_df = scored_oof[scored_oof["Date"].isin(tune_dates)].copy()
    eval_df = scored_oof[scored_oof["Date"].isin(eval_dates)].copy()
    if tune_df.empty or eval_df.empty:
        return scored_oof.copy(), scored_oof.copy()
    return tune_df, eval_df


def _prediction_from_oof_df(oof: pd.DataFrame) -> MultiHeadPrediction:
    horizon_predicted_return = {}
    horizon_up_probability = {}
    for horizon in (5, 20):
        pred_col = f"predicted_return_{horizon}d"
        prob_col = f"up_probability_{horizon}d"
        if pred_col in oof.columns:
            horizon_predicted_return[horizon] = oof[pred_col].values
        if prob_col in oof.columns:
            horizon_up_probability[horizon] = oof[prob_col].values
    return MultiHeadPrediction(
        predicted_return=oof["predicted_return"].values,
        up_probability=oof["up_probability"].values,
        quantile_low=oof["quantile_low"].values,
        quantile_mid=oof["quantile_mid"].values,
        quantile_high=oof["quantile_high"].values,
        horizon_predicted_return=horizon_predicted_return,
        horizon_up_probability=horizon_up_probability,
    )

def _print_progress(step: int, total: int, message: str):
    print(f"[{step}/{total}] {message}")


def _round_floats(obj, digits: int = 3):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    return obj


def _compute_oof_diagnostics(scored_oof: pd.DataFrame) -> dict:
    if scored_oof.empty:
        return {}

    req = {"target_log_return", "rel_strength", "norm_return", "predicted_log_return", "uncertainty_score", "uncertainty_width"}
    if not req.issubset(set(scored_oof.columns)):
        return {}

    df = scored_oof[list(req)].copy().dropna()
    if df.empty:
        return {}

    actual_up = (df["target_log_return"] > 0).astype(int)

    rel_dir_acc = float(((df["rel_strength"] > 0).astype(int) == actual_up).mean())
    norm_dir_acc = float(((df["norm_return"] > 0).astype(int) == actual_up).mean())
    pred_dir_acc = float(((df["predicted_log_return"] > 0).astype(int) == actual_up).mean())

    abs_error = (df["predicted_log_return"] - df["target_log_return"]).abs()

    return {
        "direction_accuracy": {
            "predicted_log_return": pred_dir_acc,
            "rel_strength": rel_dir_acc,
            "norm_return": norm_dir_acc,
        },
        "uncertainty_diagnostics": {
            "corr_uncertainty_vs_abs_error": float(df["uncertainty_width"].corr(abs_error)),
            "corr_uncertainty_score_vs_abs_error": float(df["uncertainty_score"].corr(abs_error)),
            "uncertainty_score_zero_ratio": float((df["uncertainty_score"] == 0).mean()),
            "uncertainty_score_mean": float(df["uncertainty_score"].mean()),
        },
    }




def _ensure_universe_size(symbols: list[str], expected_size: int) -> list[str]:
    """Backward-compatible helper retained for older tests/import paths."""
    uniq = list(dict.fromkeys(str(s) for s in symbols))
    if len(uniq) >= expected_size:
        return uniq[:expected_size]
    pads = [f"NO_DATA_{i:03d}" for i in range(1, expected_size - len(uniq) + 1)]
    return uniq + pads

def _expand_predictions_to_universe(pred_df: pd.DataFrame, universe_symbols: list[str] | None) -> pd.DataFrame:
    if not universe_symbols:
        return pred_df

    universe = set(str(s) for s in universe_symbols)
    return pred_df[pred_df["Symbol"].astype(str).isin(universe)].copy()





def _calibrate_up_probability(oof_df: pd.DataFrame, up_probs: pd.Series | pd.Index | list | tuple | pd.Series) -> pd.Series:
    if oof_df.empty or "up_probability" not in oof_df.columns or "target_log_return" not in oof_df.columns:
        return pd.Series(up_probs, dtype=float)

    cal = oof_df[["up_probability", "target_log_return"]].copy().dropna()
    if cal.empty or cal["up_probability"].nunique() < 3:
        return pd.Series(up_probs, dtype=float)

    y = (cal["target_log_return"] > 0).astype(int)
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal["up_probability"].astype(float).values, y.values)
        return pd.Series(iso.predict(pd.Series(up_probs, dtype=float).values), dtype=float).clip(0.0, 1.0)
    except Exception:
        return pd.Series(up_probs, dtype=float)


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_fallback{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"[경고] 파일이 열려있어 기본 경로에 저장하지 못했습니다. 대체 경로로 저장: {fallback}")
        return fallback


def _build_combined_symbol_results(pred_df: pd.DataFrame, summary_csv: str | None, out_path: Path) -> str | None:
    if pred_df.empty or not summary_csv:
        return None
    try:
        summary = pd.read_csv(summary_csv)
    except Exception:
        return None

    if summary.empty:
        return None

    if "Symbol" not in summary.columns:
        return None

    extra_cols = [c for c in summary.columns if c not in pred_df.columns]
    combined = pred_df.merge(summary[["Symbol", *extra_cols]], on="Symbol", how="left")
    saved = _safe_to_csv(combined, out_path)
    return str(saved)


def run_pipeline(
    input_csv: str,
    output_csv: str,
    universe_csv: str | None = None,
    report_json: str | None = None,
    figure_dir: str = "reports/figures",
    use_external: bool = True,
    use_investor_context: bool = False,
    dart_api_key: str | None = None,
    dart_corp_map_csv: str | None = None,
    config_json: str | None = None,
    enable_investor_flow: bool = True,
    enable_investor_disclosure: bool = True,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    naver_client_id: str | None = None,
    naver_client_secret: str | None = None,
    min_value_traded: float | None = None,
    turnover_limit: float | None = None,
    min_up_probability: float | None = None,
    min_signal_score: float | None = None,
    min_external_coverage_ratio: float | None = None,
    min_investor_coverage_ratio: float | None = None,
    portfolio_value: float | None = None,
    max_daily_participation: float | None = None,
    max_positions_per_market_type: int | None = None,
    enable_issue_summary: bool = True,
    issue_summary_symbols: list[str] | None = None,
):
    effective_openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    effective_openai_model = openai_model or os.getenv("OPENAI_MODEL") or ("gpt-5" if effective_openai_api_key else None)
    total_steps = 13
    _print_progress(1, total_steps, "Loading app configuration")
    cfg_overrides: dict[str, dict[str, float]] = {"backtest": {}}
    if min_value_traded is not None:
        cfg_overrides["backtest"]["min_value_traded"] = float(min_value_traded)
    if turnover_limit is not None:
        cfg_overrides["backtest"]["turnover_limit"] = float(turnover_limit)
    if min_up_probability is not None:
        cfg_overrides["backtest"]["min_up_probability"] = float(min_up_probability)
    if min_signal_score is not None:
        cfg_overrides["backtest"]["min_signal_score"] = float(min_signal_score)
    if min_external_coverage_ratio is not None:
        cfg_overrides["backtest"]["min_external_coverage_ratio"] = float(min_external_coverage_ratio)
    if min_investor_coverage_ratio is not None:
        cfg_overrides["backtest"]["min_investor_coverage_ratio"] = float(min_investor_coverage_ratio)
    if portfolio_value is not None:
        cfg_overrides["backtest"]["portfolio_value"] = float(portfolio_value)
    if max_daily_participation is not None:
        cfg_overrides["backtest"]["max_daily_participation"] = float(max_daily_participation)
    if max_positions_per_market_type is not None:
        cfg_overrides["backtest"]["max_positions_per_market_type"] = int(max_positions_per_market_type)
    if not cfg_overrides["backtest"]:
        cfg_overrides = {}
    cfg = load_app_config(config_json, overrides=cfg_overrides or None)

    _print_progress(2, total_steps, f"Loading input data: {input_csv}")

    raw = load_ohlcv_csv(input_csv)
    cleaned = clean_ohlcv(raw)

    _print_progress(3, total_steps, "Applying data cleaning and universe filter")
    requested_universe_symbols = None
    if universe_csv:
        universe = load_universe_symbols(universe_csv)
        requested_universe_symbols = list(universe)
        data = filter_by_universe(cleaned, universe)
    else:
        requested_universe_symbols = sorted(cleaned["Symbol"].astype(str).unique().tolist())
        data = cleaned.copy()

    _print_progress(4, total_steps, "Adding investor context")
    investor_context_coverage = {
        "enabled": False,
        "flow": {"requested": 0, "successful": 0, "failed": 0},
        "disclosure": {"requested": 0, "successful": 0, "failed": 0},
        "news": {"requested": 0, "successful": 0, "failed": 0},
    }
    context_raw_df = pd.DataFrame(
        columns=["Date", "Symbol", "source_type", "title", "published_at", "provider", "url", "raw_id"]
    )
    if use_investor_context:
        data, investor_context_coverage = add_investor_context_with_coverage(
            data,
            InvestorContextConfig(
                enabled=True,
                enable_flow=enable_investor_flow,
                enable_disclosure=enable_investor_disclosure,
                dart_api_key=dart_api_key,
                dart_corp_map_csv=dart_corp_map_csv,
            ),
        )

    should_collect_context_raw = bool(enable_issue_summary and (dart_api_key or (naver_client_id and naver_client_secret)))
    if should_collect_context_raw:
        try:
            context_symbols = sorted(data["Symbol"].dropna().astype(str).unique().tolist())
            context_symbol_name_map = get_symbol_name_map(context_symbols)
            context_raw_df = collect_context_raw_events(
                symbols=context_symbols,
                start=data["Date"].min().strftime("%Y-%m-%d"),
                end=data["Date"].max().strftime("%Y-%m-%d"),
                dart_api_key=dart_api_key,
                dart_corp_map_csv=dart_corp_map_csv,
                symbol_name_map=context_symbol_name_map,
                naver_client_id=naver_client_id,
                naver_client_secret=naver_client_secret,
            )
        except Exception:
            context_raw_df = pd.DataFrame(
                columns=["Date", "Symbol", "source_type", "title", "published_at", "provider", "url", "raw_id"]
            )

    _print_progress(5, total_steps, "Building price features")
    feat = build_features(data, cfg.feature)
    _print_progress(6, total_steps, "Adding external market features")
    external_coverage = {"requested": 0, "successful": 0, "failed": 0, "fallback_used": 0, "details": []}
    if cfg.external.enabled and use_external:
        feat, external_coverage = add_external_market_features_with_coverage(feat, cfg.external.market_symbols)
    feat = annotate_market_regime(feat)
    feat = feat.dropna(subset=["target_log_return"]).copy()
    feature_columns = _feature_columns(feat)

    _print_progress(7, total_steps, "Running walk-forward validation")
    folds, oof = walk_forward_validate_with_oof(feat, feature_columns, cfg.training)
    effective_cfg = cfg.training
    if not folds:
        effective_cfg = _adaptive_training_cfg(cfg, feat)
        folds, oof = walk_forward_validate_with_oof(feat, feature_columns, effective_cfg)

    wf_summary = pd.DataFrame([f.metrics for f in folds]).mean().to_dict() if folds else {}

    _print_progress(8, total_steps, "Evaluating baselines")
    baseline_summary = evaluate_baselines(feat)

    _print_progress(9, total_steps, "Using walk-forward OOF predictions")
    if oof.empty:
        raise RuntimeError("OOF predictions are empty. Increase data length or adjust training window.")

    external_coverage_ratio = (
        float(external_coverage.get("successful", 0)) / float(external_coverage.get("requested", 1) or 1)
        if use_external and external_coverage.get("requested", 0)
        else 1.0
    )
    investor_requested = 0
    investor_successful = 0
    for key in ("flow", "disclosure", "news"):
        investor_requested += int(investor_context_coverage.get(key, {}).get("requested", 0))
        investor_successful += int(investor_context_coverage.get(key, {}).get("successful", 0))
    investor_coverage_ratio = float(investor_successful) / float(investor_requested or 1) if investor_requested else 1.0
    coverage_gate_status = _coverage_gate_status(cfg, external_coverage_ratio, investor_coverage_ratio)
    prediction_context = PredictionFrameContext(
        external_coverage_ratio=external_coverage_ratio,
        investor_coverage_ratio=investor_coverage_ratio,
        min_liquidity_threshold=cfg.backtest.min_value_traded,
    )

    oof_pred = _prediction_from_oof_df(oof)
    oof_pred.up_probability = _calibrate_up_probability(oof, oof_pred.up_probability).values
    scored_oof = build_scored_prediction_frame(oof, oof_pred, cfg.signal, prediction_context)
    scored_oof["coverage_gate_status"] = coverage_gate_status

    tune_df, eval_df = _split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7)

    _print_progress(10, total_steps, "Tuning signal weights (train split)")
    tuned = tune_signal_weights(tune_df)
    cfg.signal.return_weight = tuned["return_weight"]
    cfg.signal.up_prob_weight = tuned["up_prob_weight"]
    cfg.signal.rel_strength_weight = tuned["rel_strength_weight"]
    cfg.signal.uncertainty_penalty = tuned["uncertainty_penalty"]

    scored_oof = build_scored_prediction_frame(oof, oof_pred, cfg.signal, prediction_context)
    scored_oof["coverage_gate_status"] = coverage_gate_status
    tune_df, eval_df = _split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7)

    _print_progress(11, total_steps, "Running backtest on holdout split and creating figures")
    backtest_input = eval_df if not eval_df.empty else scored_oof
    backtest = run_long_only_topk_backtest(backtest_input, cfg.backtest)
    backtest_series = pd.DataFrame(backtest.get("series", []))
    figure_dir_path = resolve_output_dir(figure_dir)
    fig_paths = save_backtest_figures(backtest_series, str(figure_dir_path))
    signal_hist = save_signal_histogram(scored_oof, str(figure_dir_path))
    actual_vs_pred = save_actual_vs_predicted_plot(scored_oof, str(figure_dir_path))
    actual_vs_pred_price = save_actual_vs_predicted_price_plot(scored_oof, str(figure_dir_path))
    diagnostic_figs = save_diagnostic_figures(scored_oof, str(figure_dir_path))
    symbol_level_figs = save_symbol_level_comparison_figures(scored_oof, str(figure_dir_path))

    _print_progress(12, total_steps, "Training final model and creating latest predictions")
    train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(
        random_state=cfg.training.random_state,
        n_jobs=cfg.training.model_n_jobs,
        use_gpu=cfg.training.use_gpu,
    )
    model.fit(train_df, feature_columns, cfg.training.quantiles)

    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
    latest_pred = model.predict(latest)
    latest_pred.up_probability = _calibrate_up_probability(scored_oof, latest_pred.up_probability).values
    pred_df = build_scored_prediction_frame(latest, latest_pred, cfg.signal, prediction_context)
    pred_df["coverage_gate_status"] = coverage_gate_status
    symbol_name_map = get_symbol_name_map(pred_df["Symbol"].dropna().astype(str).tolist())
    sym_acc = build_symbol_history_accuracy(scored_oof)
    pred_df = pred_df.merge(sym_acc, on="Symbol", how="left")
    pred_df["history_direction_accuracy"] = pred_df["history_direction_accuracy"].fillna(0.5)
    pred_df = finalize_latest_prediction_frame(pred_df, symbol_name_map)
    if enable_issue_summary:
        pred_df = append_issue_summary_columns(
            pred_df,
            context_raw_df=context_raw_df,
            openai_api_key=effective_openai_api_key,
            openai_model=effective_openai_model,
            summarize_symbols=issue_summary_symbols,
        )
    else:
        pred_df = append_issue_summary_columns(pred_df, summarize_symbols=[])
    pred_df["예측 신뢰도"] = pred_df["confidence_score"].map(lambda v: _format_percentage_text(v, digits=1, unit_interval=True))
    pred_df["예측 이유"] = pred_df["prediction_reason"]
    pred_df["권고"] = pred_df["recommendation"]

    bt_summary_cols = _backtest_summary_fields(backtest)
    for k, v in bt_summary_cols.items():
        pred_df[k] = v

    symbol_summary_artifacts = save_symbol_summary_artifacts(pred_df, scored_oof, str(figure_dir_path))
    oof_diagnostics = _compute_oof_diagnostics(scored_oof)

    _print_progress(13, total_steps, "Saving artifacts")
    pred_detail = pred_df.drop(columns=["Close"], errors="ignore")
    overlap_cols = [c for c in pred_detail.columns if c in latest.columns and c not in {"Date", "Symbol"}]
    detail_df = latest.merge(pred_detail.drop(columns=overlap_cols, errors="ignore"), on=["Date", "Symbol"], how="left")
    detail_numeric_cols = detail_df.select_dtypes(include=["number"]).columns
    detail_df.loc[:, detail_numeric_cols] = detail_df.loc[:, detail_numeric_cols].round(3)
    detail_df["내일 예상 종가"] = detail_df["predicted_close"].map(lambda v: "-" if pd.isna(v) else f"{float(v):,.0f}원")
    detail_df["상승확률(%)"] = detail_df["up_probability"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    detail_df["predicted_return_display"] = detail_df["predicted_return"].map(lambda v: _format_percentage_text(v, digits=3))
    detail_df["up_probability_display"] = detail_df["up_probability"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    if "predicted_return_5d" in detail_df.columns:
        detail_df["5일 예상 수익률(%)"] = detail_df["predicted_return_5d"].map(lambda v: _format_percentage_text(v, digits=3))
    if "predicted_return_20d" in detail_df.columns:
        detail_df["20일 예상 수익률(%)"] = detail_df["predicted_return_20d"].map(lambda v: _format_percentage_text(v, digits=3))
    if "up_probability_5d" in detail_df.columns:
        detail_df["5일 상승확률(%)"] = detail_df["up_probability_5d"].map(
            lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
        )
    if "up_probability_20d" in detail_df.columns:
        detail_df["20일 상승확률(%)"] = detail_df["up_probability_20d"].map(
            lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
        )
    detail_df["confidence_score_display"] = detail_df["confidence_score"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    detail_df["history_direction_accuracy_display"] = detail_df["history_direction_accuracy"].map(
        lambda v: _format_percentage_text(v, digits=1, unit_interval=True)
    )
    simple_df = _build_result_simple(detail_df)

    detail_path = resolve_output_path("result_detail.csv")
    detail_path = _safe_to_csv(detail_df, detail_path)

    simple_path = resolve_output_path("result_simple.csv")
    simple_path = _safe_to_csv(simple_df, simple_path)
    export_context_df = context_raw_df.copy()
    if not export_context_df.empty and "Date" in export_context_df.columns:
        export_context_df["Date"] = pd.to_datetime(export_context_df["Date"], errors="coerce").dt.normalize()
        today_kst = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
        export_context_df = export_context_df[export_context_df["Date"] == today_kst].copy()
    news_path = resolve_output_path("result_news.csv")
    news_df = (
        export_context_df[export_context_df["source_type"].astype(str) == "news"].copy()
        if "source_type" in export_context_df.columns
        else pd.DataFrame()
    )
    news_path = _safe_to_csv(news_df, news_path)
    disclosure_path = resolve_output_path("result_disclosure.csv")
    disclosure_df = (
        export_context_df[export_context_df["source_type"].astype(str) == "disclosure"].copy()
        if "source_type" in export_context_df.columns
        else pd.DataFrame()
    )
    disclosure_path = _safe_to_csv(disclosure_df, disclosure_path)

    report = {
        "universe_name": cfg.universe.name,
        "universe_size_used": int(data["Symbol"].nunique()),
        "feature_count": len(feature_columns),
        "config": app_config_to_dict(cfg),
        "walk_forward": wf_summary,
        "baselines": baseline_summary,
        "tuned_signal": tuned,
        "tuning_samples": int(len(tune_df)),
        "backtest_samples": int(len(backtest_input)),
        "backtest": {k: v for k, v in backtest.items() if k != "series"},
        "external_feature_coverage": external_coverage,
        "investor_context_coverage": investor_context_coverage,
        "coverage_gate": {
            "external_coverage_ratio": external_coverage_ratio,
            "investor_coverage_ratio": investor_coverage_ratio,
            "min_value_traded": cfg.backtest.min_value_traded,
            "status": coverage_gate_status,
        },
        "oof_diagnostics": oof_diagnostics,
        "probability_calibration": probability_calibration_metrics(
            (scored_oof["target_log_return"] > 0).astype(int).values,
            scored_oof["up_probability"].astype(float).values,
        ),
        "prediction_coverage": {
            "requested_universe_size": int(len(set(requested_universe_symbols))) if requested_universe_symbols else None,
            "predictions_row_count": int(len(pred_df)),
            "available_prediction_count": int(pred_df["predicted_return"].notna().sum()) if "predicted_return" in pred_df.columns else 0,
            "missing_prediction_count": int(pred_df["predicted_return"].isna().sum()) if "predicted_return" in pred_df.columns else 0,
            "tradable_prediction_count": int((pd.to_numeric(pred_df.get("value_traded", 0), errors="coerce").fillna(0.0) >= cfg.backtest.min_value_traded).sum())
            if "value_traded" in pred_df.columns
            else int(len(pred_df)),
        },
        "pm_summary": {
            "portfolio_action_counts": pred_df["portfolio_action"].value_counts(dropna=False).to_dict() if "portfolio_action" in pred_df.columns else {},
            "risk_flag_counts": pred_df["risk_flag"].value_counts(dropna=False).to_dict() if "risk_flag" in pred_df.columns else {},
        },
        "visualization_note": "시각화는 전체 집계 + 종목별 + 종목별 최근1개월 비교 그래프(OOF 기준)를 제공합니다.",
        "artifacts": {
            "result_detail_csv": str(detail_path),
            "result_simple_csv": str(simple_path),
            "result_news_csv": str(news_path),
            "result_disclosure_csv": str(disclosure_path),
            "figure_dir": str(figure_dir_path),
            **fig_paths,
            "signal_hist": signal_hist,
            "actual_vs_predicted": actual_vs_pred,
            "actual_vs_predicted_price": actual_vs_pred_price,
            **diagnostic_figs,
            **symbol_level_figs,
            **symbol_summary_artifacts,
        },
    }

    pm_report_path = resolve_output_path("pm_report.json")
    pm_report = build_pm_report(pred_df, report)
    save_pm_report(pm_report, pm_report_path)
    report["artifacts"]["pm_report_json"] = str(pm_report_path)

    if report_json:
        report_path = resolve_output_path(report_json)
        report_path.write_text(json.dumps(_round_floats(report, 3), indent=2, ensure_ascii=False))

    _print_prediction_console_summary(pred_df)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock next-day prediction pipeline")
    parser.add_argument("--input", required=False, default="data/real_ohlcv.csv", help="OHLCV CSV path")
    parser.add_argument(
        "--output",
        default="result_detail.csv",
        help="Legacy option (CSV outputs are always saved as result_detail.csv and result_simple.csv under result/)",
    )
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with Symbol column")
    parser.add_argument("--report-json", default="pipeline_report.json", help="Pipeline summary JSON")
    parser.add_argument("--figure-dir", default="figures", help="Directory for generated charts")
    parser.add_argument("--fetch-real", action="store_true", help="Fetch real OHLCV from yfinance before running")
    parser.add_argument("--disable-external", action="store_true", help="Disable external market feature download")
    parser.add_argument("--fetch-investor-context", action="store_true", help="Fetch investor flow context features (foreign/institution flows)")
    parser.add_argument("--disable-investor-flow", action="store_true", help="Disable pykrx investor flow context")
    parser.add_argument("--disable-disclosure-context", action="store_true", help="Disable DART disclosure context")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key for AI news scoring")
    parser.add_argument("--openai-model", default=None, help="OpenAI model for AI news scoring")
    parser.add_argument("--naver-client-id", default=None, help="Naver News Search API client id")
    parser.add_argument("--naver-client-secret", default=None, help="Naver News Search API client secret")
    parser.add_argument("--dart-api-key", default=None, help="Deprecated legacy option kept for compatibility")
    parser.add_argument("--dart-corp-map-csv", default=None, help="Deprecated legacy option kept for compatibility")
    parser.add_argument("--config-json", default=None, help="Optional JSON file overriding nested AppConfig values")
    parser.add_argument("--min-value-traded", type=float, default=None, help="Minimum daily traded value filter for backtest/report")
    parser.add_argument("--turnover-limit", type=float, default=None, help="Override backtest turnover limit")
    parser.add_argument("--min-up-probability", type=float, default=None, help="Override backtest minimum up probability")
    parser.add_argument("--min-signal-score", type=float, default=None, help="Override backtest minimum signal score")
    parser.add_argument(
        "--min-external-coverage-ratio",
        type=float,
        default=None,
        help="Override backtest minimum external feature coverage ratio",
    )
    parser.add_argument(
        "--min-investor-coverage-ratio",
        type=float,
        default=None,
        help="Override backtest minimum investor context coverage ratio",
    )
    parser.add_argument("--portfolio-value", type=float, default=None, help="Backtest portfolio notional for liquidity capacity checks")
    parser.add_argument(
        "--max-daily-participation",
        type=float,
        default=None,
        help="Maximum share of daily traded value allowed per position",
    )
    parser.add_argument(
        "--max-positions-per-market-type",
        type=int,
        default=None,
        help="Maximum number of holdings allowed per market_type bucket",
    )
    parser.add_argument("--disable-issue-summary", action="store_true", help="Disable news/disclosure summary generation")
    parser.add_argument(
        "--issue-summary-symbols",
        nargs="*",
        default=None,
        help="Generate issue summaries only for specific symbols",
    )
    parser.add_argument(
        "--real-symbols",
        nargs="*",
        default=None,
        help="Symbols used when --fetch-real is enabled (no auto KRX universe)",
    )
    parser.add_argument("--real-start", default="2018-01-01", help="Start date for real data fetch")
    parser.add_argument(
        "--add-symbols",
        nargs="*",
        default=None,
        help="Append user-entered stock codes/symbols into --input CSV (e.g., 005930 000660.KS)",
    )
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    input_csv = args.input
    if args.add_symbols:
        symbols_to_add = normalize_user_symbols(args.add_symbols)
        if symbols_to_add:
            append_real_ohlcv_csv(input_csv, symbols=symbols_to_add, start=args.real_start)
            print(f"Added symbols to {input_csv}: {len(symbols_to_add)}")
    if args.fetch_real:
        symbols = args.real_symbols
        if not symbols and args.universe_csv:
            try:
                symbols = load_universe_symbols(args.universe_csv)
                print(f"Loaded symbols from universe CSV: {len(symbols)}")
            except Exception as exc:
                print(f"[경고] universe CSV 로드 실패: {exc}")

        if not symbols:
            symbols = _fallback_symbols_from_input_or_default(input_csv)

        save_real_ohlcv_csv(input_csv, symbols=symbols, start=args.real_start)

    run_pipeline(
        args.input,
        args.output,
        args.universe_csv,
        args.report_json,
        args.figure_dir,
        use_external=not args.disable_external,
        use_investor_context=args.fetch_investor_context,
        dart_api_key=args.dart_api_key,
        dart_corp_map_csv=args.dart_corp_map_csv,
        config_json=args.config_json,
        enable_investor_flow=not args.disable_investor_flow,
        enable_investor_disclosure=not args.disable_disclosure_context,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        naver_client_id=args.naver_client_id,
        naver_client_secret=args.naver_client_secret,
        min_value_traded=args.min_value_traded,
        turnover_limit=args.turnover_limit,
        min_up_probability=args.min_up_probability,
        min_signal_score=args.min_signal_score,
        min_external_coverage_ratio=args.min_external_coverage_ratio,
        min_investor_coverage_ratio=args.min_investor_coverage_ratio,
        portfolio_value=args.portfolio_value,
        max_daily_participation=args.max_daily_participation,
        max_positions_per_market_type=args.max_positions_per_market_type,
        enable_issue_summary=not args.disable_issue_summary,
        issue_summary_symbols=args.issue_summary_symbols,
    )


if __name__ == "__main__":
    main()
