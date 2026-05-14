# Architecture

## Pipeline Flow

1. Load and clean OHLCV input from `src/data/loaders.py` and `src/data/cleaners.py`.
2. Optionally refresh real OHLCV with `src/data/fetch_real_data.py`; CLI refresh helpers live in `src/data/cli_refresh.py`.
3. Filter the requested universe from `data/` or a user-provided universe CSV.
4. Add investor context, disclosure/news raw events, external market features, and market-regime annotations.
5. Build price and event features in `src/features/price_features.py` and `src/features/investment_signals.py`.
6. Run walk-forward validation and OOF prediction generation in `src/validation/walk_forward.py`.
7. Calibrate probabilities, split OOF rows for tuning/evaluation, and compute diagnostics with `src/validation/support.py`.
8. Tune signal weights, run top-k backtests, and train the final multi-head model.
9. Build latest predictions with `src/inference/predict.py` and policy fields with `src/domain/signal_policy.py`.
10. Save CSV, JSON, and figure artifacts through `src/reports/`.

## Stable Interfaces

- `src.pipeline.run_pipeline(...)`
- `src.pipeline.build_cli_parser()`
- console scripts: `stock-predict`, `stock-predict-kakao`
- output filenames: `result_detail.csv`, `result_simple.csv`, `result_news.csv`, `result_disclosure.csv`, `pm_report.json`

Compatibility wrappers remain in `src/pipeline.py` for tests and older imports, but new helper logic should live in the relevant `data`, `validation`, or `reports` module.

## Recommendation Policy

The current signal recommendation policy is:

- `predicted_return > 2.0`: buy
- `predicted_return <= -2.0`: sell
- values between those thresholds: hold
- when `signal_score`, `up_probability`, and `uncertainty_score` are present, strong score/probability with low uncertainty can upgrade to buy, and weak score with low uncertainty can downgrade to sell

Market headwind and overbought RSI guards are applied by the pipeline policy wrapper for console/report compatibility.
