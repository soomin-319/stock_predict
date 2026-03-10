from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config.settings import AppConfig
from src.data.cleaners import clean_ohlcv
from src.data.fetch_real_data import append_real_ohlcv_csv, normalize_user_symbols, save_real_ohlcv_csv
from src.data.krx_universe import get_kospi200_kosdaq150_symbols, get_symbol_name_map, save_universe_csv
from src.data.loaders import load_ohlcv_csv
from src.data.universe import filter_by_universe, load_universe_symbols
from src.features.external_features import add_external_market_features_with_coverage
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.inference.predict import build_prediction_frame
from src.models.lgbm_heads import MultiHeadPrediction, MultiHeadStockModel
from src.reports.visualize import (
    save_actual_vs_predicted_plot,
    save_actual_vs_predicted_price_plot,
    save_diagnostic_figures,
    save_backtest_figures,
    save_signal_histogram,
    save_symbol_summary_artifacts,
)
from src.validation.backtest import run_long_only_topk_backtest
from src.validation.baselines import evaluate_baselines
from src.validation.signal_tuning import tune_signal_weights
from src.validation.walk_forward import walk_forward_oof_predictions, walk_forward_validate


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
    }
    return [
        c
        for c in df.columns
        if c.startswith(("ret_", "ma_", "close_to_ma_", "vol_", "ks", "kq", "gspc", "ixic", "sox", "vix", "krw", "tnx"))
        or c in base
    ]




def _adaptive_training_cfg(cfg, feat: pd.DataFrame):
    tuned = cfg.training
    uniq = len(feat["Date"].unique())
    tuned.min_train_size = min(tuned.min_train_size, max(60, int(uniq * 0.6)))
    tuned.test_size = min(tuned.test_size, max(20, int(uniq * 0.2)))
    tuned.step_size = min(tuned.step_size, max(20, tuned.test_size // 2))
    return tuned


def _display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
    return width


def _pad_display(text: str, width: int, align: str = "left") -> str:
    s = str(text)
    pad = max(0, width - _display_width(s))
    if align == "right":
        return " " * pad + s
    return s + " " * pad


def _recommendation_from_signal(signal_score: float | int | None, predicted_return: float | int | None) -> str:
    if pd.isna(predicted_return) or pd.isna(signal_score):
        return "관망"

    ret = float(predicted_return)
    score = float(signal_score)

    if score >= 0 and ret > 0:
        return "매수"
    if score < 0 and ret < 0:
        return "매도"
    return "관망"


def _confidence_label(confidence_score: float | int | None) -> str:
    if pd.isna(confidence_score):
        return "신뢰도 보통"
    c = float(confidence_score)
    if c >= 0.67:
        return "신뢰도 높음"
    if c >= 0.34:
        return "신뢰도 보통"
    return "신뢰도 낮음"


def _build_prediction_reason(row: pd.Series) -> str:
    direction = "상승" if float(row.get("predicted_return", 0.0)) >= 0 else "하락"
    up_prob = row.get("up_probability")
    conf = row.get("confidence_score")
    acc = row.get("history_direction_accuracy")

    up_prob_txt = "-" if pd.isna(up_prob) else f"{float(up_prob):.3f}"
    conf_txt = "-" if pd.isna(conf) else f"{float(conf):.3f}"
    acc_txt = "-" if pd.isna(acc) else f"{float(acc):.3f}"
    return f"예측방향:{direction}, 상승확률:{up_prob_txt}, 신뢰도:{conf_txt}, 과거방향정확도:{acc_txt}"


def _print_prediction_console_summary(pred_df: pd.DataFrame):
    if pred_df.empty:
        print("\n=== Predictions (all symbols, by confidence) ===")
        print("(no rows)")
        return

    out = pred_df.copy()
    if "symbol_name" not in out.columns:
        out["symbol_name"] = out["Symbol"]

    out["confidence_score"] = (1 - out["uncertainty_score"].fillna(1)).clip(lower=0, upper=1)
    out["recommendation"] = out.apply(
        lambda r: _recommendation_from_signal(r.get("signal_score"), r.get("predicted_return")), axis=1
    )
    out["confidence_label"] = out["confidence_score"].map(_confidence_label)
    out["prediction_reason"] = out.apply(_build_prediction_reason, axis=1)
    out["predicted_close_int"] = out["predicted_close"].abs().round(0).astype("Int64")
    out = out.sort_values(["confidence_score", "signal_score"], ascending=[False, False]).copy()

    rows = []
    for _, r in out.iterrows():
        ret_text = "-" if pd.isna(r.get("predicted_return")) else f"{float(r['predicted_return']):,.3f}"
        pred_close_text = "-" if pd.isna(r.get("predicted_close_int")) else f"{int(r['predicted_close_int']):,}"
        conf_text = "-" if pd.isna(r.get("confidence_score")) else f"{float(r['confidence_score']):.3f}"
        rows.append(
            {
                "심볼": str(r.get("Symbol", "")),
                "종목명": str(r["symbol_name"]),
                "권고": str(r["recommendation"]),
                "신뢰도": conf_text,
                "예상 수익률(%)": ret_text,
                "내일 예측 종가": pred_close_text,
                "예측 사유": str(r["prediction_reason"]),
            }
        )

    headers = ["심볼", "종목명", "권고", "신뢰도", "예상 수익률(%)", "내일 예측 종가", "예측 사유"]
    col_widths = {h: max(_display_width(h), *(_display_width(row[h]) for row in rows)) for h in headers}

    print("\n=== Predictions (all symbols, by confidence) ===")
    print("  ".join(_pad_display(h, col_widths[h], "left") for h in headers))
    for row in rows:
        print(
            "  ".join(
                [
                    _pad_display(row["심볼"], col_widths["심볼"], "left"),
                    _pad_display(row["종목명"], col_widths["종목명"], "left"),
                    _pad_display(row["권고"], col_widths["권고"], "left"),
                    _pad_display(row["신뢰도"], col_widths["신뢰도"], "right"),
                    _pad_display(row["예상 수익률(%)"], col_widths["예상 수익률(%)"], "right"),
                    _pad_display(row["내일 예측 종가"], col_widths["내일 예측 종가"], "right"),
                    _pad_display(row["예측 사유"], col_widths["예측 사유"], "left"),
                ]
            )
        )

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
    return MultiHeadPrediction(
        predicted_return=oof["predicted_return"].values,
        up_probability=oof["up_probability"].values,
        quantile_low=oof["quantile_low"].values,
        quantile_mid=oof["quantile_mid"].values,
        quantile_high=oof["quantile_high"].values,
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


def _expand_predictions_to_universe(pred_df: pd.DataFrame, universe_symbols: list[str] | None) -> pd.DataFrame:
    if not universe_symbols:
        return pred_df

    universe = set(str(s) for s in universe_symbols)
    return pred_df[pred_df["Symbol"].astype(str).isin(universe)].copy()



def _ensure_universe_size(symbols: list[str], expected_size: int) -> list[str]:
    """Backward-compatible helper retained for older tests/import paths."""
    uniq = list(dict.fromkeys(str(s) for s in symbols))
    if len(uniq) >= expected_size:
        return uniq[:expected_size]
    pads = [f"NO_DATA_{i:03d}" for i in range(1, expected_size - len(uniq) + 1)]
    return uniq + pads

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


def _expand_predictions_to_universe(pred_df: pd.DataFrame, universe_symbols: list[str] | None) -> pd.DataFrame:
    if not universe_symbols:
        return pred_df

    universe = set(str(s) for s in universe_symbols)
    return pred_df[pred_df["Symbol"].astype(str).isin(universe)].copy()



def _ensure_universe_size(symbols: list[str], expected_size: int) -> list[str]:
    """Backward-compatible helper retained for older tests/import paths."""
    uniq = list(dict.fromkeys(str(s) for s in symbols))
    if len(uniq) >= expected_size:
        return uniq[:expected_size]
    pads = [f"NO_DATA_{i:03d}" for i in range(1, expected_size - len(uniq) + 1)]
    return uniq + pads

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


def _expand_predictions_to_universe(pred_df: pd.DataFrame, universe_symbols: list[str] | None) -> pd.DataFrame:
    if not universe_symbols:
        return pred_df

    universe = set(str(s) for s in universe_symbols)
    return pred_df[pred_df["Symbol"].astype(str).isin(universe)].copy()

def run_pipeline(
    input_csv: str,
    output_csv: str,
    universe_csv: str | None = None,
    report_json: str | None = None,
    figure_dir: str = "reports/figures",
    use_external: bool = True,
):
    total_steps = 12
    _print_progress(1, total_steps, "Loading app configuration")
    cfg = AppConfig()

    _print_progress(2, total_steps, f"Loading input data: {input_csv}")

    raw = load_ohlcv_csv(input_csv)
    cleaned = clean_ohlcv(raw)

    _print_progress(3, total_steps, "Applying data cleaning and universe filter")
    requested_universe_symbols = None
    if universe_csv:
        universe = load_universe_symbols(universe_csv, cfg.universe)
        requested_universe_symbols = list(universe)
        data = filter_by_universe(cleaned, universe)
    else:
        requested_universe_symbols = sorted(cleaned["Symbol"].astype(str).unique().tolist())
        data = cleaned.copy()

    _print_progress(4, total_steps, "Building price features")
    feat = build_features(data, cfg.feature)
    _print_progress(5, total_steps, "Adding external market features")
    external_coverage = {"requested": 0, "successful": 0, "failed": 0, "fallback_used": 0, "details": []}
    if cfg.external.enabled and use_external:
        feat, external_coverage = add_external_market_features_with_coverage(feat, cfg.external.market_symbols)
    feat = annotate_market_regime(feat)
    feat = feat.dropna(subset=["target_log_return"]).copy()
    feature_columns = _feature_columns(feat)

    _print_progress(6, total_steps, "Running walk-forward validation")
    folds = walk_forward_validate(feat, feature_columns, cfg.training)
    effective_cfg = cfg.training
    if not folds:
        effective_cfg = _adaptive_training_cfg(cfg, feat)
        folds = walk_forward_validate(feat, feature_columns, effective_cfg)

    wf_summary = pd.DataFrame([f.metrics for f in folds]).mean().to_dict() if folds else {}

    _print_progress(7, total_steps, "Evaluating baselines")
    baseline_summary = evaluate_baselines(feat)

    _print_progress(8, total_steps, "Generating OOF predictions")
    oof = walk_forward_oof_predictions(feat, feature_columns, effective_cfg)
    if oof.empty:
        raise RuntimeError("OOF predictions are empty. Increase data length or adjust training window.")

    oof_pred = _prediction_from_oof_df(oof)
    scored_oof = build_prediction_frame(oof, oof_pred, cfg.signal)
    scored_oof["target_log_return"] = oof["target_log_return"].values

    tune_df, eval_df = _split_oof_for_tuning_and_eval(scored_oof, tune_ratio=0.7)

    _print_progress(9, total_steps, "Tuning signal weights (train split)")
    tuned = tune_signal_weights(tune_df)
    cfg.signal.return_weight = tuned["return_weight"]
    cfg.signal.up_prob_weight = tuned["up_prob_weight"]
    cfg.signal.rel_strength_weight = tuned["rel_strength_weight"]
    cfg.signal.uncertainty_penalty = tuned["uncertainty_penalty"]

    scored_oof = build_prediction_frame(oof, oof_pred, cfg.signal)
    scored_oof["target_log_return"] = oof["target_log_return"].values

    _print_progress(10, total_steps, "Running backtest on holdout split and creating figures")
    backtest_input = eval_df if not eval_df.empty else scored_oof
    backtest = run_long_only_topk_backtest(backtest_input, cfg.backtest)
    backtest_series = pd.DataFrame(backtest.get("series", []))
    figure_dir_path = resolve_output_dir(figure_dir)
    fig_paths = save_backtest_figures(backtest_series, str(figure_dir_path))
    signal_hist = save_signal_histogram(scored_oof, str(figure_dir_path))
    actual_vs_pred = save_actual_vs_predicted_plot(scored_oof, str(figure_dir_path))
    actual_vs_pred_price = save_actual_vs_predicted_price_plot(scored_oof, str(figure_dir_path))
    diagnostic_figs = save_diagnostic_figures(scored_oof, str(figure_dir_path))

    _print_progress(11, total_steps, "Training final model and creating latest predictions")
    train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(random_state=cfg.training.random_state)
    model.fit(train_df, feature_columns, cfg.training.quantiles)

    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
    latest_pred = model.predict(latest)
    pred_df = build_prediction_frame(latest, latest_pred, cfg.signal)
    symbol_name_map = get_symbol_name_map(pred_df["Symbol"].dropna().astype(str).tolist())
    pred_df["symbol_name"] = pred_df["Symbol"].astype(str).map(symbol_name_map).fillna(pred_df["Symbol"].astype(str))
    pred_df["confidence_score"] = (1 - pred_df["uncertainty_score"].fillna(1)).clip(lower=0, upper=1)
    pred_df["signal_label"] = pred_df["confidence_score"].map(_confidence_label)

    sym_acc = pd.DataFrame(columns=["Symbol", "history_direction_accuracy"])
    if {"Symbol", "target_log_return", "predicted_log_return"}.issubset(set(scored_oof.columns)):
        tmp_acc = scored_oof[["Symbol", "target_log_return", "predicted_log_return"]].copy()
        tmp_acc["history_direction_accuracy"] = (
            (tmp_acc["target_log_return"] > 0).astype(int) == (tmp_acc["predicted_log_return"] > 0).astype(int)
        ).astype(float)
        sym_acc = tmp_acc.groupby("Symbol", as_index=False)["history_direction_accuracy"].mean()
    pred_df = pred_df.merge(sym_acc, on="Symbol", how="left")
    pred_df["prediction_reason"] = pred_df.apply(_build_prediction_reason, axis=1)

    symbol_summary_artifacts = save_symbol_summary_artifacts(pred_df, scored_oof, str(figure_dir_path))
    oof_diagnostics = _compute_oof_diagnostics(scored_oof)

    _print_progress(12, total_steps, "Saving artifacts")
    pred_numeric_cols = pred_df.select_dtypes(include=["number"]).columns
    pred_df.loc[:, pred_numeric_cols] = pred_df.loc[:, pred_numeric_cols].round(3)
    oof_numeric_cols = scored_oof.select_dtypes(include=["number"]).columns
    scored_oof.loc[:, oof_numeric_cols] = scored_oof.loc[:, oof_numeric_cols].round(3)

    output_path = resolve_output_path(output_csv)
    pred_df.to_csv(output_path, index=False)

    oof_path = resolve_output_path("oof_predictions.csv")
    scored_oof.to_csv(oof_path, index=False)

    report = {
        "universe_name": cfg.universe.name,
        "universe_size_used": int(data["Symbol"].nunique()),
        "feature_count": len(feature_columns),
        "walk_forward": wf_summary,
        "baselines": baseline_summary,
        "tuned_signal": tuned,
        "tuning_samples": int(len(tune_df)),
        "backtest_samples": int(len(backtest_input)),
        "backtest": {k: v for k, v in backtest.items() if k != "series"},
        "external_feature_coverage": external_coverage,
        "oof_diagnostics": oof_diagnostics,
        "prediction_coverage": {
            "requested_universe_size": int(len(set(requested_universe_symbols))) if requested_universe_symbols else None,
            "predictions_row_count": int(len(pred_df)),
            "available_prediction_count": int(pred_df["predicted_return"].notna().sum()) if "predicted_return" in pred_df.columns else 0,
            "missing_prediction_count": int(pred_df["predicted_return"].isna().sum()) if "predicted_return" in pred_df.columns else 0,
        },
        "visualization_note": "진단 그래프는 전체 종목/전체 OOF 샘플을 집계한 결과입니다.",
        "artifacts": {
            "predictions_csv": str(output_path),
            "oof_predictions_csv": str(oof_path),
            "figure_dir": str(figure_dir_path),
            **fig_paths,
            "signal_hist": signal_hist,
            "actual_vs_predicted": actual_vs_pred,
            "actual_vs_predicted_price": actual_vs_pred_price,
            **diagnostic_figs,
            **symbol_summary_artifacts,
        },
    }

    if report_json:
        report_path = resolve_output_path(report_json)
        report_path.write_text(json.dumps(_round_floats(report, 3), indent=2, ensure_ascii=False))
        print(f"Saved report to {report_path}")

    print("[안내] 시각자료(그래프)는 개별 종목별 차트가 아니라 전체 종목 샘플을 집계한 요약 진단입니다.")
    _print_prediction_console_summary(pred_df)
    print(f"Saved inference output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stock next-day prediction pipeline")
    parser.add_argument("--input", required=False, default="data/real_ohlcv.csv", help="OHLCV CSV path")
    parser.add_argument("--output", default=r"C:\Users\카운\Desktop\result\predictions_direct.csv", help="Output CSV path")
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with Symbol column")
    parser.add_argument("--report-json", default=r"C:\Users\카운\Desktop\result\pipeline_report.json", help="Pipeline summary JSON")
    parser.add_argument("--figure-dir", default=r"C:\Users\카운\Desktop\result\figures", help="Directory for generated charts")
    parser.add_argument("--fetch-real", action="store_true", help="Fetch real OHLCV from yfinance before running")
    parser.add_argument("--disable-external", action="store_true", help="Disable external market feature download")
    parser.add_argument(
        "--real-symbols",
        nargs="*",
        default=None,
        help="Symbols used when --fetch-real is enabled (default: auto KOSPI200+KOSDAQ150)",
    )
    parser.add_argument("--real-start", default="2018-01-01", help="Start date for real data fetch")
    parser.add_argument(
        "--add-symbols",
        nargs="*",
        default=None,
        help="Append user-entered stock codes/symbols into --input CSV (e.g., 005930 000660.KS)",
    )
    args = parser.parse_args()

    input_csv = args.input
    if args.add_symbols:
        symbols_to_add = normalize_user_symbols(args.add_symbols)
        if symbols_to_add:
            append_real_ohlcv_csv(input_csv, symbols=symbols_to_add, start=args.real_start)
            print(f"Added symbols to {input_csv}: {len(symbols_to_add)}")
    if args.fetch_real:
        symbols = args.real_symbols
        if not symbols:
            symbols = get_kospi200_kosdaq150_symbols()
            print(f"Auto-loaded KOSPI200+KOSDAQ150 symbols: {len(symbols)}")
            if args.universe_csv is None:
                auto_universe_csv = str(resolve_output_path(r"C:\Users\카운\Desktop\result\universe_kospi200_kosdaq150.csv"))
                save_universe_csv(auto_universe_csv, symbols)
                args.universe_csv = auto_universe_csv
                print(f"Saved universe CSV to {auto_universe_csv}")

        save_real_ohlcv_csv(input_csv, symbols=symbols, start=args.real_start)
        print(f"Fetched real market data to {input_csv}")

    run_pipeline(args.input, args.output, args.universe_csv, args.report_json, args.figure_dir, use_external=not args.disable_external)


if __name__ == "__main__":
    main()
