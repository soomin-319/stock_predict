from __future__ import annotations

import argparse
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
from src.data.loaders import load_ohlcv_csv
from src.features.price_features import build_features
from src.features.regime_features import annotate_market_regime
from src.inference.predict import build_prediction_frame
from src.models.lgbm_heads import MultiHeadStockModel
from src.validation.walk_forward import walk_forward_validate


def resolve_output_path(output_csv: str, is_windows: bool | None = None) -> Path:
    """Normalize output path and create parent directory if needed.

    On Windows, '/tmp/...' style paths are mapped to the OS temp directory
    to avoid '\\tmp' non-existent-root issues.
    """
    win = (os.name == "nt") if is_windows is None else is_windows

    if win and output_csv.startswith("/tmp/"):
        output_path = Path(tempfile.gettempdir()) / output_csv[len("/tmp/") :]
    else:
        output_path = Path(output_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def run_pipeline(input_csv: str, output_csv: str):
    cfg = AppConfig()
    raw = load_ohlcv_csv(input_csv)
    feat = build_features(raw, cfg.feature)
    feat = annotate_market_regime(feat)

    feature_columns = [
        c
        for c in feat.columns
        if c.startswith(("ret_", "ma_", "close_to_ma_", "vol_"))
        or c in {"daily_return", "gap_return", "intraday_return", "range_pct", "vol_ratio_20", "rsi_14"}
    ]

    folds = walk_forward_validate(feat.dropna(subset=["target_log_return"]), feature_columns, cfg.training)
    if folds:
        summary = pd.DataFrame([f.metrics for f in folds]).mean().to_dict()
        print("Walk-forward metrics:", {k: round(v, 6) for k, v in summary.items()})

    train_df = feat.dropna(subset=feature_columns + ["target_log_return", "target_up"])
    model = MultiHeadStockModel(random_state=cfg.training.random_state)
    model.fit(train_df, feature_columns, cfg.training.quantiles)

    latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
    pred = model.predict(latest)
    pred_df = build_prediction_frame(latest, pred, cfg.signal)

    output_path = resolve_output_path(output_csv)
    pred_df.to_csv(output_path, index=False)
    print(f"Saved inference output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stock next-day prediction pipeline")
    parser.add_argument("--input", required=True, help="OHLCV CSV path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
