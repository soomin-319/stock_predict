from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config.settings import AppConfig
from src.data.cleaners import clean_ohlcv
from src.data.fetch_real_data import save_real_ohlcv_csv
from src.data.krx_universe import get_kospi200_kosdaq150_symbols, save_universe_csv
from src.data.loaders import load_ohlcv_csv
from src.data.universe import filter_by_universe, load_universe_symbols
from src.features.external_features import add_external_market_features_with_coverage
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.inference.predict import build_prediction_frame
from src.models.lgbm_heads import MultiHeadPrediction, MultiHeadStockModel
from src.reports.visualize import save_backtest_figures, save_signal_histogram
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


def _print_prediction_console_summary(pred_df: pd.DataFrame, top_n: int = 10):
    cols = ["Symbol", "predicted_return", "up_probability", "uncertainty_band", "signal_score", "signal_label"]
    top = pred_df.sort_values("signal_score", ascending=False).head(top_n)[cols].copy()
    print("\n=== Top predictions ===")
    print(top.to_string(index=False))


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
    if universe_csv:
        universe = load_universe_symbols(universe_csv, cfg.universe)
        data = filter_by_universe(cleaned, universe)
    else:
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

    _print_progress(11, total_steps, "Training final model and creating latest predictions")
    train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(random_state=cfg.training.random_state)
    model.fit(train_df, feature_columns, cfg.training.quantiles)

    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
    latest_pred = model.predict(latest)
    pred_df = build_prediction_frame(latest, latest_pred, cfg.signal)

    _print_progress(12, total_steps, "Saving artifacts")
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
        "artifacts": {
            "predictions_csv": str(output_path),
            "oof_predictions_csv": str(oof_path),
            "figure_dir": str(figure_dir_path),
            **fig_paths,
            "signal_hist": signal_hist,
        },
    }

    if report_json:
        report_path = resolve_output_path(report_json)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Saved report to {report_path}")

    _print_prediction_console_summary(pred_df, top_n=min(10, len(pred_df)))
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
    args = parser.parse_args()

    input_csv = args.input
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
