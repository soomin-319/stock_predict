from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Support both:
# 1) python -m src.pipeline
# 2) python src/pipeline.py (IDE/direct execution)
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config.settings import AppConfig
from src.data.cleaners import clean_ohlcv
from src.data.loaders import load_ohlcv_csv
from src.data.universe import filter_by_universe, load_universe_symbols
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.inference.predict import build_prediction_frame
from src.models.lgbm_heads import MultiHeadStockModel
from src.validation.backtest import run_long_only_topk_backtest
from src.validation.baselines import evaluate_baselines
from src.validation.signal_tuning import tune_signal_weights
from src.validation.walk_forward import walk_forward_validate


def resolve_output_path(output_csv: str, is_windows: bool | None = None) -> Path:
    win = (os.name == "nt") if is_windows is None else is_windows
    if win and output_csv.startswith("/tmp/"):
        output_path = Path(tempfile.gettempdir()) / output_csv[len("/tmp/") :]
    else:
        output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith(("ret_", "ma_", "close_to_ma_", "vol_"))
        or c in {"daily_return", "gap_return", "intraday_return", "range_pct", "vol_ratio_20", "rsi_14"}
    ]


def run_pipeline(input_csv: str, output_csv: str, universe_csv: str | None = None, report_json: str | None = None):
    cfg = AppConfig()

    # Stage 1) data realism: clean + fixed universe
    raw = load_ohlcv_csv(input_csv)
    cleaned = clean_ohlcv(raw)
    if universe_csv:
        universe = load_universe_symbols(universe_csv, cfg.universe)
        data = filter_by_universe(cleaned, universe)
    else:
        universe = set(cleaned["Symbol"].astype(str).unique())
        data = cleaned.copy()

    feat = annotate_market_regime(build_features(data, cfg.feature))
    feat = feat.dropna(subset=["target_log_return"]).copy()
    feature_columns = _feature_columns(feat)

    # Stage 2) walk-forward + baseline comparison
    folds = walk_forward_validate(feat, feature_columns, cfg.training)
    if not folds:
        tuned_training = cfg.training
        tuned_training.min_train_size = min(252, max(60, int(len(feat["Date"].unique()) * 0.6)))
        tuned_training.test_size = min(126, max(20, int(len(feat["Date"].unique()) * 0.2)))
        tuned_training.step_size = max(20, tuned_training.test_size // 2)
        folds = walk_forward_validate(feat, feature_columns, tuned_training)
    wf_summary = pd.DataFrame([f.metrics for f in folds]).mean().to_dict() if folds else {}
    baseline_summary = evaluate_baselines(feat)

    # fit final model
    train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(random_state=cfg.training.random_state)
    model.fit(train_df, feature_columns, cfg.training.quantiles)

    # in-sample prediction frame for stage 3/4 analysis
    pred_train = model.predict(train_df)
    pred_train_df = build_prediction_frame(train_df, pred_train, cfg.signal)
    pred_train_df["target_log_return"] = train_df["target_log_return"].values

    # Stage 3) tune signal score weights
    tuned = tune_signal_weights(pred_train_df)
    cfg.signal.return_weight = tuned["return_weight"]
    cfg.signal.up_prob_weight = tuned["up_prob_weight"]
    cfg.signal.rel_strength_weight = tuned["rel_strength_weight"]
    cfg.signal.uncertainty_penalty = tuned["uncertainty_penalty"]

    pred_train_df = build_prediction_frame(train_df, pred_train, cfg.signal)
    pred_train_df["target_log_return"] = train_df["target_log_return"].values

    # Stage 4) cost-aware backtest
    backtest = run_long_only_topk_backtest(pred_train_df, cfg.backtest)

    # latest inference output
    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
    latest_pred = model.predict(latest)
    pred_df = build_prediction_frame(latest, latest_pred, cfg.signal)

    output_path = resolve_output_path(output_csv)
    pred_df.to_csv(output_path, index=False)

    report = {
        "universe_name": cfg.universe.name,
        "universe_size_used": int(data["Symbol"].nunique()),
        "walk_forward": wf_summary,
        "baselines": baseline_summary,
        "tuned_signal": tuned,
        "backtest": backtest,
    }

    if report_json:
        report_path = resolve_output_path(report_json)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Saved report to {report_path}")

    print("Pipeline summary:", json.dumps(report, ensure_ascii=False))
    print(f"Saved inference output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stock next-day prediction pipeline")
    parser.add_argument("--input", required=True, help="OHLCV CSV path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV with Symbol column")
    parser.add_argument("--report-json", default="reports/pipeline_report.json", help="Pipeline summary JSON")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.universe_csv, args.report_json)


if __name__ == "__main__":
    main()
