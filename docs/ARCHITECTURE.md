# Architecture

## Pipeline Flow

1. Load and clean OHLCV input from `src/data/loaders.py` and `src/data/cleaners.py`.
2. Optionally refresh real OHLCV with `src/data/fetch_real_data.py`; CLI refresh helpers live in `src/data/cli_refresh.py`.
3. Filter the requested universe from `data/` or a user-provided universe CSV.
4. Add investor context, display-only disclosure/news raw events, external market features, and market-regime annotations.
5. Build price and event features in `src/features/price_features.py` and `src/features/investment_signals.py`.
6. Run walk-forward validation and OOF prediction generation in `src/validation/walk_forward.py`.
7. Calibrate probabilities, split OOF rows for tuning/evaluation, and compute diagnostics with `src/validation/support.py`.
8. Tune ranking weights, run top-k backtests, and train the final multi-head model.
9. Build latest predictions with `src/inference/predict.py` and policy fields with `src/domain/signal_policy.py`.
10. Save CSV, JSON, and figure artifacts through `src/reports/`.

## Stable Interfaces

- `src.pipeline.run_pipeline(...)`
- `src.pipeline.build_cli_parser()`
- console scripts: `stock-predict`, `stock-predict-kakao`, `stock-news-impact`
- output filenames: `result_detail.csv`, `result_simple.csv`, `result_news.csv`, `result_disclosure.csv`, `pm_report.json`

Compatibility wrappers remain in `src/pipeline.py` for tests and older imports, but new helper logic should live in the relevant `data`, `validation`, or `reports` module.

## News Impact Package Boundary

`news_impact/` is a vendored top-level package migrated from `stock-news-impact`.
It has its own console entry point, config template, watchlist examples, collectors,
deduplication, scoring, LLM client, stock-factor classifier, backtesting helpers, and
JSON report generation.

Within `stock_predict`, this package remains an external context producer. The main
pipeline can read a generated report with `--news-impact-report` and append
`news_impact_*` columns through `src/reports/news_impact_context.py`, but those
columns are display/review metadata only.

## Recommendation Policy

The current signal recommendation policy is:

- `predicted_return > 2.0`: buy
- `predicted_return <= -2.0`: sell
- values between those thresholds: hold

The buy/sell/hold decision must use only the next-day expected return (`predicted_return`). `signal_score`, `up_probability`, uncertainty, market context, news, disclosures, and `news_impact_*` report columns can be reported as supporting diagnostics, ranking context, or user-facing explanations, but they must not override the buy/sell/hold label.

News, disclosure, and news-impact data are display-only issue context. They can appear in `result_news.csv`, `result_disclosure.csv`, `result_simple.csv`, `result_detail.csv`, and KakaoTalk responses, but they must not affect the expected return or the recommendation decision.

## GitHub/Colab/Kakao Runtime

The intended client runtime is:

1. Project code is stored in GitHub.
2. Google Colab pulls or clones the GitHub repository and runs the pipeline/chatbot entry point.
3. The Colab runtime exposes the Flask webhook, normally through ngrok.
4. KakaoTalk sends user messages to that webhook and receives cached or newly generated prediction summaries.
