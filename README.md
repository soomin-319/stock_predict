# Stock Predict

Python stock prediction pipeline for next-day and multi-horizon return signals. It builds price, market, and investor-flow features, validates with walk-forward OOF predictions, runs a long-only top-k backtest, and writes CSV/JSON/figure artifacts under `result/`.

This project is for research and operations support. The client uses its outputs as one reference material for investment decisions; it is not an investment-advice or automated trading system. Buy/sell/hold signals are based only on the next-day expected return (`predicted_return`). News and disclosures are collected and summarized only for user display, and must not change the expected return or the buy/sell decision.

## Install

Use the same Python environment for install, CLI runs, and tests. In this workspace, the dependency-installed interpreter is Python 3.12 under the user-local Python installation; the global `python` on `PATH` may point elsewhere.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Run

Sample data, no live market downloads:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json --figure-dir figures_smoke
```

Installed console entry point:

```powershell
stock-predict --input data/sample_ohlcv.csv --disable-external
```

Real OHLCV refresh before running:

```powershell
python src/pipeline.py --fetch-real --input data/real_ohlcv.csv --real-symbols 005930.KS 000660.KS
```

Append symbols to an existing real-data CSV:

```powershell
python src/pipeline.py --input data/real_ohlcv.csv --add-symbols 005930 000660.KS --real-start 2024-01-01
```

## Main CLI Options

- `--input`: OHLCV input CSV. Default: `data/real_ohlcv.csv`.
- `--output`: legacy compatibility option. CSV outputs are always normalized under `result/`.
- `--universe-csv`: optional CSV with a `Symbol` column.
- `--report-json`: pipeline report JSON filename. Default: `pipeline_report.json`.
- `--figure-dir`: generated figure directory under `result/`. Default: `figures`.
- `--fetch-real`: download OHLCV before the run.
- `--real-symbols`: explicit symbols for `--fetch-real`.
- `--real-start`: start date for real-data fetch. Default: `2020-01-01`.
- `--auto-refresh-real`: incrementally append latest OHLCV when not doing a full refresh.
- `--add-symbols`: normalize and append user-entered stock codes/symbols to the input CSV.
- `--disable-external`: skip external market features.
- `--fetch-investor-context`: enable investor flow/disclosure/news context.
- `--disable-disclosure-context`: skip DART disclosure context.
- `--openai-api-key`, `--openai-model`: optional issue/news summary settings.
- `--naver-client-id`, `--naver-client-secret`: optional Naver News Search credentials.
- `--config-json`: nested `AppConfig` override file.
- `--min-value-traded`, `--turnover-limit`, `--min-up-probability`, `--min-signal-score`: backtest/report overrides.
- `--min-external-coverage-ratio`, `--min-investor-coverage-ratio`: coverage gate overrides.
- `--portfolio-value`, `--max-daily-participation`, `--max-positions-per-market-type`: liquidity and portfolio capacity controls.
- `--issue-summary-symbols`: restrict issue-summary generation to selected symbols.

Deprecated but still accepted for compatibility: `--dart-api-key`, `--dart-corp-map-csv`.

## Inputs

Required columns:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

Optional but recommended:

- `Symbol`
- investor/context columns such as `foreign_net_buy`, `institution_net_buy`
- display-only issue context such as disclosure/news summaries; these are shown to users but are not inputs to the expected-return decision policy
- market structure fields such as `market_type`, `venue`, `session`, `listing_date`

Use sample data for deterministic local testing. Avoid live network calls in tests.

## Outputs

All generated outputs are written under `result/`:

- `result_detail.csv`: full latest prediction rows and feature context.
- `result_simple.csv`: user-facing summary used by Kakao chatbot.
- `result_news.csv`, `result_disclosure.csv`: live/raw context or generated issue-summary snapshots.
- `pm_report.json`: portfolio-manager style summary.
- `pipeline_report.json` or the `--report-json` filename.
- figure files under the selected `--figure-dir`.

CSV files are saved with `utf-8-sig` for Excel/Windows compatibility.

## Tests

```powershell
pytest
pytest tests/test_pipeline_smoke.py
pytest tests/test_kakao_colab_bot.py
```

Pytest cache and temporary files are configured under ignored `result/` paths.

## Kakao Bot

The production usage pattern is GitHub -> Google Colab -> KakaoTalk:

1. Store the project code in GitHub.
2. Load the repository in a Colab runtime and run the pipeline/chatbot entry point there.
3. Expose the Colab Flask app, typically through ngrok, so the user can communicate with the service through a KakaoTalk chatbot.

The Kakao/Colab integration lives in `src/chatbot/kakao_colab_bot.py` and can be started through:

```powershell
stock-predict-kakao
```

It reads cached predictions from `result/result_simple.csv`, starts background prediction jobs when a symbol is missing, and can prewarm the prediction cache for default symbols. Chatbot responses may include news/disclosure summaries, but those summaries are display-only context separate from the expected-return signal.

## Environment Variables

- `OPENAI_API_KEY`, `OPENAI_MODEL`: issue/news summary generation.
- `DART_API_KEY`, `DART_CORP_MAP_CSV`: disclosure context.
- `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`: Naver news context.

Do not commit API keys, credentials, or private market data.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Operations](docs/OPERATIONS.md)
- [Roadmap](docs/ROADMAP.md)
- [Feature overview](docs/PROJECT_FEATURES_OVERVIEW.md)
- [External data integration](docs/EXTERNAL_DATA_INTEGRATION_GUIDE.md)
- [Prediction formulas](docs/PREDICTION_FORMULAS.md)
