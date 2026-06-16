# Repository Guidelines

## Project Purpose & Guardrails

This is a Python stock prediction pipeline for next-day and multi-horizon return signals. It builds price, market, and investor-flow features, validates with walk-forward OOF predictions, runs long-only top-k backtests, and writes CSV/JSON artifacts under `result/`. Treat outputs as research and operations support only, not investment advice or an automated trading system. Buy/sell/hold decisions must be based on next-day expected return (`predicted_return`); news and disclosures are display-only context and must not change expected return or signal decisions.

## Project Structure & Module Organization

Core code lives in `src/`: `data/` loaders and market context, `features/` feature engineering, `models/` model heads, `validation/` backtests and metrics, `inference/` prediction logic, `reports/` outputs, `domain/` signal policy, `recommendation/` recommendation helpers, `config/` configuration objects, and `chatbot/` Kakao/Colab integration. Tests live in `tests/` with `test_*.py` files. Configuration presets are in `configs/`, sample and universe CSVs are in `data/`, generated reports are under `result/`, Colab helpers are in `colab/`, and project notes are in `docs/`.

## Build, Test, and Development Commands

- `python -m pip install -r requirements.txt`: install runtime dependencies.
- `python -m pip install -e .`: install locally and expose console scripts.
- `pytest`: run the full test suite.
- `pytest tests/test_pipeline_smoke.py`: run a focused smoke test.
- `pytest tests/test_kakao_colab_bot.py`: run the chatbot integration tests.
- `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`: run against bundled sample data.
- `stock-predict --input data/sample_ohlcv.csv --disable-external`: run via the installed console entry point.
- `python src/pipeline.py --fetch-real --input data/real_ohlcv.csv --real-symbols 005930.KS 000660.KS`: refresh real OHLCV before running.
- `python src/pipeline.py --input data/real_ohlcv.csv --add-symbols 005930 000660.KS --real-start 2024-01-01`: append symbols to an existing real-data CSV.
- `stock-predict-kakao`: start the Kakao/Colab chatbot entry point.

Use the same Python environment for install, CLI runs, and tests. This project requires Python 3.10+; this workspace commonly uses Python 3.12 from the user-local installation.

## Coding Style & Naming Conventions

Use existing PEP 8 style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and uppercase constants. Prefer typed, small functions that keep data preparation, modeling, validation, reporting, and chatbot concerns separated. Reuse helpers in `src/config`, `src/data`, and `src/pipeline_support` before adding abstractions. No formatter is configured in `pyproject.toml`; keep diffs minimal.

## Testing Guidelines

Tests use pytest, with cache and temporary files configured under ignored `result/` paths. Add or update tests in `tests/` for behavior changes, especially data fetching fallbacks, signal policy, calibration, reports, external context, recommendation logic, and chatbot cache behavior. Name files `test_<feature>.py` and test functions `test_<expected_behavior>`. Prefer sample CSVs and deterministic fixtures over live network calls. Run `pytest` before submitting, or at minimum impacted tests plus `tests/test_pipeline_smoke.py`.


## News Impact Scoring Module

The vendored `src.news_impact` package is the migrated `stock-news-impact` news/disclosure scoring module. Keep the original news collection principle: Korean news first, Korean company/industry/search terms by default, and non-Korean or overseas media only when explicitly needed. Within `stock_predict`, news impact data is display/review context only and must not alter `predicted_return`, expected-return ranking, recommendation, or automated signal decisions.

## Data, Outputs, and External Integrations

All generated CSV and JSON outputs should stay under `result/`, including `result_detail.csv`, `result_simple.csv`, `result_news.csv`, `result_disclosure.csv`, `pm_report.json`, and `pipeline_report.json`. CSV outputs should remain `utf-8-sig` for Excel/Windows compatibility. Sample and universe inputs belong in `data/`; avoid adding large or private market data. Live integrations include yfinance, DART, Naver News, OpenAI summaries, Flask, and pyngrok. Keep tests deterministic by mocking or disabling these integrations unless the change explicitly targets a live-fetch path.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, often with PR references, for example `Refresh stale cached predictions from detail date in bot handler (#207)`. Keep commits focused. If any repository change is made, the agent MUST commit, push, and create a pull request before final handoff unless the user explicitly says not to. Do not stop after local changes only. Pull requests should include a summary, test results, linked issues when relevant, and artifact paths when user-facing outputs change. Note new config keys, data files, or external API requirements.

## Security & Configuration Tips

Do not commit API keys, ngrok tokens, or private market data. Pass secrets such as `OPENAI_API_KEY`, `DART_API_KEY`, `NAVER_CLIENT_ID`, and `NAVER_CLIENT_SECRET` through environment-specific tooling or local arguments (`--openai-api-key`, `--openai-model`, `--naver-client-id`, `--naver-client-secret`). Treat `result/` as generated output and avoid checking in large or stale artifacts unless intentional.
