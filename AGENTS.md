# Repository Guidelines

## Project Structure & Module Organization

This is a Python stock prediction pipeline. Core code lives in `src/`: `data/` loaders and market context, `features/` feature engineering, `models/` model heads, `validation/` backtests and metrics, `inference/` prediction logic, `reports/` outputs, `domain/` signal policy, and `chatbot/` Kakao integration. Tests live in `tests/` with `test_*.py` files. Configuration presets are in `configs/`, sample and universe CSVs are in `data/`, generated reports and figures are under `result/`, and project notes are in `docs/`.

## Build, Test, and Development Commands

- `python -m pip install -r requirements.txt`: install dependencies.
- `python -m pip install -e .`: install locally and expose console scripts.
- `pytest`: run the full test suite.
- `pytest tests/test_pipeline_smoke.py`: run a focused smoke test.
- `pytest tests/test_kakao_colab_bot.py`: run the chatbot integration tests.
- `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json --figure-dir figures_smoke`: run against bundled sample data.
- `stock-predict --input data/sample_ohlcv.csv --disable-external`: run via the installed console entry point.
- `stock-predict-kakao`: start the Kakao/Colab chatbot entry point.

## Coding Style & Naming Conventions

Use Python 3.10+ and existing PEP 8 style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and uppercase constants. Prefer typed, small functions that keep data preparation, modeling, validation, and reporting concerns separated. Reuse helpers in `src/config`, `src/data`, and `src/pipeline_support` before adding abstractions. No formatter is configured in `pyproject.toml`; keep diffs minimal.

## Testing Guidelines

Tests use pytest, with cache and temporary files configured under ignored `result/` paths. Add or update tests in `tests/` for behavior changes, especially data fetching fallbacks, signal policy, calibration, reports, and chatbot cache behavior. Name files `test_<feature>.py` and test functions `test_<expected_behavior>`. Prefer sample CSVs and deterministic fixtures over live network calls. Run `pytest` before submitting, or at minimum impacted tests plus `tests/test_pipeline_smoke.py`.

## Data, Outputs, and External Integrations

All generated CSV, JSON, and figure outputs should stay under `result/`, including `result_detail.csv`, `result_simple.csv`, `pm_report.json`, `pipeline_report.json`, and selected figure directories. Sample and universe inputs belong in `data/`; avoid adding large or private market data. Live integrations include yfinance/pykrx, DART, Naver News, OpenAI summaries, Flask, and pyngrok. Keep tests deterministic by mocking or disabling these integrations unless the change explicitly targets a live-fetch path.

## Commit & Pull Request Guidelines

Recent history uses short imperative commit subjects, often with PR references, for example `Refresh stale cached predictions from detail date in bot handler (#207)`. Keep commits focused. Pull requests should include a summary, test results, linked issues when relevant, and screenshots or artifact paths when figures or user-facing outputs change. Note new config keys, data files, or external API requirements.

## Security & Configuration Tips

Do not commit API keys, ngrok tokens, or private market data. Pass secrets such as `--openai-api-key`, `DART_API_KEY`, `NAVER_CLIENT_ID`, and `NAVER_CLIENT_SECRET` through environment-specific tooling or local arguments. Treat `result/` as generated output and avoid checking in large or stale artifacts unless intentional.
